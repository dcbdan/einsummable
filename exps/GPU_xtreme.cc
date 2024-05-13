#include "../src/base/args.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/reference.h"
#include <cstdint>
#include "../src/misc/update.h"

using tensor_t = graph_writer_t::tensor_t;

struct datum_t;

struct xtreme_dist_t {
  xtreme_dist_t(
    communicator_t& comm,
    server_base_t* s)
    : communicator(comm), server(s), register_cmd(s->get_registered_cmd())
  {}

  void insert_random(int gid, relation_t const& rel);

  void insert_labels(
    int gid,
    dtype_t d, placement_t const& pl,
    vector<datum_t> const& data);
  void insert_labels(int gid, relation_t const& rel, vector<datum_t> const& data);

  void insert_features(
    int gid,
    dtype_t d, placement_t const& pl,
    vector<datum_t> const& data);
  void insert_features(int gid, relation_t const& rel, vector<datum_t> const& data);

  void client_insert_random();
  void client_insert_labels();
  void client_insert_features();

  static string insert_random_cmd()   { return "xtreme_dist_t/insert_random";   }
  static string insert_labels_cmd()   { return "xtreme_dist_t/insert_labels";   }
  static string insert_features_cmd() { return "xtreme_dist_t/insert_features"; }

private:
  communicator_t& communicator;
  server_base_t* server;
  string register_cmd;

  void _insert_random(relation_t const& rel);
  void _insert_labels(
    relation_t       const& rel,
    vector<int>      const& exclusive_sum,
    vector<uint64_t> const& labels);
  void _insert_features(
    relation_t       const& rel,
    vector<int>      const& exclusive_sum,
    vector<uint64_t> const& features,
    vector<double>   const& scores);

  relation_t make_fresh_rel(dtype_t d, placement_t const& pl);
};

void main_rank_zero(
  gpu_mg_server_t* server,
  random_inserter_t& random_inserter,
  args_t& pargs,
  autoplace_config_t config);

struct random_inserter_t {
  random_inserter_t(gpu_mg_server_t* server, communicator_t& comm)
    : server(server), comm(comm), cmd("random_inserter")
  {}

  void client() {
    auto rel = relation_t::from_wire(comm.recv_string(0));
    vector<scalar_t> vs = comm.recv_vector<scalar_t>(0);
    _local(rel, vs.at(0), vs.at(1));
  }

  void operator()(int gid, placement_t const& pl, scalar_t lower, scalar_t upper) 
  {
    relation_t rel {
      .dtype = lower.dtype,
      .placement = pl,
      .tids = vtensor_t<int>(pl.block_shape())
    };
    {
      vector<int>& tids = rel.tids.get();
      int next_tid = 1 + server->get_max_tid();
      std::iota(tids.begin(), tids.end(), next_tid);
    }
    
    int world_size = comm.get_world_size();
    string registered_cmd = server->get_registered_cmd();
    for(int dst = 1; dst != world_size; ++dst) {
      comm.send_string(dst, registered_cmd);
      comm.send_string(dst, cmd);
      comm.send_string(dst, rel.to_wire());
      comm.send_vector(dst, vector<scalar_t>{lower, upper});
    }
   
    _local(rel, lower, upper);
    // DOUT("gid: " << gid);
    server->insert_gid_without_data(gid, rel);
  }

  void _local(relation_t const& rel, scalar_t lower, scalar_t upper) {
    // DLINE;
    int this_rank = comm.get_this_rank();
    int nbid = rel.placement.num_parts();
    vector<int> const& locs = rel.placement.locations.get();
    vector<int> const& tids = rel.tids.get();
    vector<uint64_t> block_sizes = rel.placement.partition.all_block_sizes().get();

    map<int, tuple<int, buffer_t>> data;
    for(int bid = 0; bid != nbid; ++bid) {
      int const& loc = locs.at(bid);
      if(server->is_local_gpu(loc)) {
        dbuffer_t d = make_dbuffer(rel.dtype, block_sizes.at(bid));
        d.random(lower, upper);
        // d.zeros();
        int const& tid = tids.at(bid);
        data.insert({tid, {loc, d.data}});
      }
    }

    server->local_insert_tensors(data);
  }

  gpu_mg_server_t* server;
  communicator_t& comm;
  string cmd;
};

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);

  int world_size = 1;
  bool is_rank_zero = true;

  communicator_t communicator("0.0.0.0", is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  uint64_t mem_size = 10lu * 1000lu * 1000lu * 1000lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < 4; ++i){
    buffer_sizes.push_back(mem_size);
  }

  auto gpu_ptr = new gpu_mg_server_t(communicator, buffer_sizes);
  gpu_ptr->set_use_storage(false);
  gpu_ptr->set_split_off_inputs(true);

  std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(gpu_ptr);

  random_inserter_t random_inserter(gpu_ptr, communicator);

  if(!is_rank_zero) {
    server->register_listen(
      random_inserter.cmd, 
      [&] { random_inserter.client(); });

    server->listen();

    return 0;
  }

  args_t args(argc, argv);
  args.set_default("config_threads", 1);
  int n_locs = 4;

  int num_config = args.get<int>("config_threads");
  DOUT("num compute per gpu " << num_config);
  autoplace_config_t config = autoplace_config_t::make_default01(
    n_locs, num_config);

  // for (int i = 0; i < 3; i++){
  //   main_rank_zero(gpu_ptr, random_inserter, args, config);
  // }
  main_rank_zero(gpu_ptr, random_inserter, args, config);

  server->shutdown();

  return 0;
}

struct rms_norm_t {
  rms_norm_t(
    graph_writer_t& w,
    uint64_t dim)
    : writer(w), eps(1e-6), dtype(default_dtype())
  {
    weight = writer.input(vector<uint64_t>{dim}, dtype);
  }

  tensor_t norm(tensor_t x) {
    dtype_t d = x.get_dtype();
  
    if(d == dtype_t::f16) {
      throw std::runtime_error("rms_norm_t::norm needs >16 precision");
    }
  
    auto x_shape = x.get_shape()();
    int out_rank = x_shape.size();
    if(out_rank <= 1) {
      throw std::runtime_error("rms_norm: not a big enough output rank");
    }
  
    scalarop_t inverse_sqrt = scalarop_t::make_inverse_sqrt(d);
    scalarop_t square       = scalarop_t::make_square(d);
    scalarop_t mul          = scalarop_t::make_mul(d);
  
    scalar_t _e(d, write_with_ss(eps));
    scalar_t _a(d, write_with_ss(1.0/double(double(1.0)*x_shape.back())));
    scalarop_t scale_then_add_eps = scalarop_t::combine(
      scalarop_t::make_add(d),
      {
        scalarop_t::make_scale(_a),
        scalarop_t::make_constant(_e)
      });
  
    string ijk(out_rank, ' ');
    std::iota(ijk.begin(), ijk.end(), 'a');
  
    string ij(out_rank-1, ' ');
    std::iota(ij.begin(), ij.end(), 'a');
  
    string ijk_to_ij     = ijk + "->" + ij;
    string ijk_ij_to_ijk = ijk + ","  + ij + "->" + ijk;
  
    // z = x * np.power(np.mean(np.square(x), axis=-1, keepdims=True) + eps, -0.5);
    // y = np.mean(np.square(x), axis=-1) + eps

    tensor_t y;
    y = writer.ew(square,                            x);
    y = writer.reduction(ijk_to_ij, castable_t::add, y);
    y = writer.ew(scale_then_add_eps,                y);
    y = writer.ew(inverse_sqrt,                      y);
  
    // x * y
    return writer.ew(ijk_ij_to_ijk, mul, x, y);
  }

  tensor_t forward(tensor_t x) {
    if(dtype != x.get_dtype()) {
      throw std::runtime_error("invalid input dtype rms norm t forward");
    }
  
    // compute output with a minimum precision of 32
    tensor_t output;
    if(dtype == dtype_t::f16) {
      output = norm(x.to_dtype(dtype_t::f32)).to_dtype(dtype);
    } else {
      output = norm(x);
    }
  
    int out_rank = x.rank();
  
    string ijk(out_rank, ' ');
    std::iota(ijk.begin(), ijk.end(), 'a');
    string k(1, char('a' + (out_rank-1)));
    string str = ijk + "," + k + "->" + ijk;
  
    scalarop_t mul = scalarop_t::make_mul(dtype);
  
    return writer.ew(str, mul, output, weight);
  }

  graph_writer_t& writer;
  float eps;
  string name;
  dtype_t dtype;
  tensor_t weight;
};

struct ff_t {
  ff_t(
    graph_writer_t& writer,
    uint64_t d_inn,
    uint64_t d_out,
    scalarop_t activation)
    : writer(writer), activation(activation)
  {
    weight = writer.input({d_inn, d_out});
  }

  tensor_t forward(tensor_t x) {
    x = writer.matmul(x, weight);
    if(activation.is_identity()) {
      return x;
    } else {
      return writer.ew(activation, x);
    }
  }

  graph_writer_t& writer;
  scalarop_t activation;
  tensor_t weight;
};

struct sigmoid_out_t {
  sigmoid_out_t() {}

  sigmoid_out_t(
    graph_writer_t& w,
    uint64_t d_inn,
    uint64_t d_out)
    : writer(&w), sigmoid(scalarop_t::make_sigmoid())
  {
    weight = writer->input({d_inn, d_out});    
  }

  tensor_t forward(tensor_t x) {
    x = writer->matmul(x, weight);
    x = writer->ew(sigmoid, x);
    return x;
  }
 
  graph_writer_t* writer;
  scalarop_t sigmoid;
  tensor_t weight;
};

struct model_config_t {
  uint64_t num_features;
  uint64_t num_labels;
  vector<uint64_t> dim_hidden;
};

struct model_t {
  model_t(graph_writer_t& writer, model_config_t const& config)
    : writer(writer)
  {
    vector<uint64_t> szs;
    szs.push_back(config.num_features);
    vector_concatenate_into(szs, config.dim_hidden);
    szs.push_back(config.num_labels);

    vector<scalarop_t> scalarops(szs.size() - 1, scalarop_t::make_silu());

    int nszs = szs.size();
    for(int i = 0; i != nszs - 2; ++i) {
      ffs.emplace_back(writer, szs[i], szs[i+1], scalarops[i]);
      norms.emplace_back(writer, szs[i+1]);
    }

    out_layer = sigmoid_out_t(writer, szs[nszs-2], szs[nszs-1]);
  }

  tensor_t forward(tensor_t data)
  {
    tensor_t x = data;
    for(int i = 0; i != ffs.size(); ++i) {
      auto& ff   = ffs[i];
      auto& norm = norms[i];
      x = norm.forward(ff.forward(x));
    }
    x = out_layer.forward(x);
    return x;
  }

  vector<tensor_t> get_ff_weights() const {
    auto ret = vector_from_each_member(ffs, tensor_t, weight);
    ret.push_back(out_layer.weight);
    return ret;
  }
  vector<tensor_t> get_norm_weights() const {
    return vector_from_each_member(norms, tensor_t, weight);
  }
  tensor_t get_mean_logits() const {
    return out_layer.mean_logits;
  }

  graph_writer_t& writer;

  vector<ff_t> ffs;
  vector<rms_norm_t> norms;
  sigmoid_out_t out_layer;
};

tensor_t compute_mse(graph_writer_t& writer, tensor_t sampling, tensor_t x, tensor_t y)
{
  scalarop_t difference = scalarop_t::combine(
    scalarop_t::make_square(),
    { scalarop_t::make_sub () });

  tensor_t loss = writer.straight_bew(difference, x, y);

  loss = writer.ew("bl,l->bl", scalarop_t::make_mul(), loss, sampling);

  uint64_t nelem = product(loss.get_shape()());
  loss = loss.scale(write_with_ss(1.0/double(nelem)));

  return loss;
}

struct xtreme_t {
  struct train_t {
    // the graph inputs are
    // (1) inn_data
    // (2) out_data,
    // (3) sampling                    <- "constants"
    // (4) ff_weights, norm_weights    <- "trainables"
    // (5) vector_mapfst(init_fills)   <- "constants"
    graph_t graph;
    int inn_data;
    int out_data;
    int sampling;
    int mean_logits;
    int mse;
    int regular;
    int loss;
    vector<int> ff_weights;
    vector<int> norm_weights;
    vector<int> grads;
    vector<tuple<int, fill_t>> init_fills;

    map<int, relation_t> inn_rels;
    taskgraph_t taskgraph;
    map<int, relation_t> out_rels;
    vector<tuple<int, int>> next_iter_remap;

    placement_t get_placement(int inn_gid) const {
      return inn_rels.at(inn_gid).placement;
    }
  };

  struct validate_t {
    graph_t graph;
    int inn_data;
    vector<int> ff_weights;
    vector<int> norm_weights;
    vector<tuple<int, int>> constants_tid_to_vid;
    int predictions;

    map<int, relation_t> inn_rels;
    taskgraph_t taskgraph;
    map<int, relation_t> out_rels;

    placement_t get_placement(int inn_gid) const {
      return inn_rels.at(inn_gid).placement;
    }
  };

  vector<tuple<int, int>> remap_to_validate() const {
    vector<tuple<int, int>> ret = validate.constants_tid_to_vid;
    for(int i = 0; i != train.ff_weights.size(); ++i) {
      ret.emplace_back(train.ff_weights[i], validate.ff_weights[i]);
    }
    for(int i = 0; i != train.norm_weights.size(); ++i) {
      ret.emplace_back(train.norm_weights[i], validate.norm_weights[i]);
    }
    return ret;
  }

  vector<tuple<int, int>> remap_from_validate() const {
    vector<tuple<int, int>> ret;
    for(auto const& [t_id, v_id]: remap_to_validate()) {
      ret.emplace_back(v_id, t_id);
    }
    return ret;
  }

  train_t train;
  validate_t validate;

  static
  xtreme_t make(
    uint64_t batch_train,
    uint64_t batch_validate,
    float regularize_scale,
    model_config_t const& model_config,
    updater_desc_t const& updater_desc,
    autoplace_config_t const& autoplace_config);

  static
  train_t make_train_info(
    uint64_t batch,
    float regularize_scale,
    model_config_t const& model_config,
    updater_desc_t const& updater_desc,
    autoplace_config_t const& autoplace_config);

  static
  validate_t make_validate_info(
    uint64_t batch,
    vector<tuple<int, vector<uint64_t>, dtype_t>> const& constants,
    int train_mean_logits,
    model_config_t const& model_config,
    autoplace_config_t const& autoplace_config);
};

map<int, relation_t> make_rels(
  graph_t const& graph,
  vector<placement_t> const& pls,
  map<int, vtensor_t<int>> const& xs)
{
  map<int, relation_t> ret;
  for(auto const& [gid, tids]: xs) {
    auto const& node = graph.nodes[gid];
    ret.insert({gid, relation_t {
      .dtype     = node.op.out_dtype(),
      .placement = pls[gid],
      .tids = tids
    }});
  }
  return ret;
}

xtreme_t::train_t xtreme_t::make_train_info(
  uint64_t batch,
  float lambda,
  model_config_t const& model_config,
  updater_desc_t const& updater_desc,
  autoplace_config_t const& autoplace_config)
{
  graph_writer_t writer;
  model_t model(writer, model_config);

  tensor_t mean_logits = model.get_mean_logits();
  mean_logits.save_inplace();

  tensor_t inn_data = writer.input({batch, model_config.num_features});
  tensor_t predictions = model.forward(inn_data);
  tensor_t out_data = writer.input({batch, model_config.num_labels});

  tensor_t sampling = writer.input(vector<uint64_t>{model_config.num_labels});
  sampling.save_inplace();

  tensor_t mse = compute_mse(writer, sampling, out_data, predictions);
  mse = mse.sum_to_unit();
  mse.save_inplace();

  // Add a weight regularization term
  vector<tensor_t> ff_weights = model.get_ff_weights();
  auto square = scalarop_t::make_square(ff_weights[0].get_dtype());
  tensor_t regular = writer.ew(square, ff_weights[0]).sum_to_unit();
  uint64_t nffelem = 0;
  for(int i = 1; i != ff_weights.size(); ++i) {
    tensor_t t = writer.ew(square, ff_weights[i]).sum_to_unit();
    regular = writer.add(regular, t);
    nffelem += product(ff_weights[i].get_shape()());
  }
  lambda *= 1.0e5;
  lambda /= double(nffelem);
  regular = regular.scale(scalar_t(lambda).convert(default_dtype()));
  regular.save_inplace();

  tensor_t loss = writer.add(mse, regular);
  loss.save_inplace();

  vector<tensor_t> weights = vector_concatenate(
    ff_weights, model.get_norm_weights());
  vector<tensor_t> grads = writer.backprop(loss, weights);

  for(auto& grad: grads) {
    grad.save_inplace();
  }

  xtreme_t::train_t ret {
    .graph = writer.get_graph(),
    .inn_data = inn_data.get_id(),
    .out_data = out_data.get_id(),
    .sampling = sampling.get_id(),
    .mean_logits = mean_logits.get_id(),
    .mse = mse.get_id(),
    .regular = regular.get_id(),
    .loss = loss.get_id(),
    .ff_weights = vector_from_each_method(ff_weights, int, get_id),
    .norm_weights = vector_from_each_method(model.get_norm_weights(), int, get_id),
    .grads   = vector_from_each_method(grads, int, get_id)
  };

  vector<tuple<int, int>> old_news;
  ret.init_fills = update_weights(
    updater_desc, ret.graph, old_news, 
    vector_concatenate(ret.ff_weights, ret.norm_weights),
    vector_from_each_method(grads, int, get_id));

  for(auto const& [old_id, new_id]: old_news) {
    ret.graph.nodes[new_id].op.set_save(true);
    ret.next_iter_remap.emplace_back(new_id, old_id);
  }
  // At this point, next_iter_remap includes
  // (1) weights
  // (2) states added by the updater
  ret.next_iter_remap.emplace_back(sampling.get_id(), sampling.get_id());
  ret.next_iter_remap.emplace_back(mean_logits.get_id(), mean_logits.get_id());
  // Now also make sure to include sampling, mean_logits so that it won't get deleted

  auto pls = autoplace01(ret.graph, autoplace_config);

  auto [inn_tids, out_tids, taskgraph] = taskgraph_t::make(ret.graph, pls);

  ret.inn_rels = make_rels(ret.graph, pls, inn_tids);
  ret.taskgraph = taskgraph;
  ret.out_rels = make_rels(ret.graph, pls, out_tids);

  return ret;
}

xtreme_t::validate_t xtreme_t::make_validate_info(
  uint64_t batch,
  vector<tuple<int, vector<uint64_t>, dtype_t>> const& constants,
  int train_mean_logits,
  model_config_t const& model_config,
  autoplace_config_t const& autoplace_config)
{
  graph_writer_t writer;
  model_t model(writer, model_config);

  tensor_t mean_logits = model.get_mean_logits();
  mean_logits.save_inplace();

  tensor_t inn_data = writer.input({batch, model_config.num_features});
  tensor_t predictions = model.forward(inn_data);
  predictions.save_inplace();

  for(auto& w: model.get_ff_weights()) {
    w.save_inplace();
  }
  for(auto& w: model.get_norm_weights()) {
    w.save_inplace();
  }

  vector<tuple<int, int>> constant_ids;
  for(auto const& [train_id, shape, dtype]: constants) {
    tensor_t t = writer.input(shape, dtype);
    t.save_inplace();
    constant_ids.emplace_back(train_id, t.get_id());
  }
  constant_ids.emplace_back(train_mean_logits, mean_logits.get_id());

  xtreme_t::validate_t ret {
    .graph = writer.get_graph(),
    .inn_data = inn_data.get_id(),
    .ff_weights = vector_from_each_method(model.get_ff_weights(), int, get_id),
    .norm_weights = vector_from_each_method(model.get_norm_weights(), int, get_id),
    .constants_tid_to_vid = constant_ids,
    .predictions = predictions.get_id(),
  };

  auto pls = autoplace01(ret.graph, autoplace_config);

  auto [inn_tids, out_tids, taskgraph] = taskgraph_t::make(ret.graph, pls);

  ret.inn_rels = make_rels(ret.graph, pls, inn_tids);
  ret.taskgraph = taskgraph;
  ret.out_rels = make_rels(ret.graph, pls, out_tids);

  return ret;
}

xtreme_t xtreme_t::make(
  uint64_t batch_train,
  uint64_t batch_validate,
  float regularize_scale,
  model_config_t const& model_config,
  updater_desc_t const& updater_desc,
  autoplace_config_t const& autoplace_config)
{
  auto train = make_train_info(
    batch_train, regularize_scale, model_config, updater_desc, autoplace_config);

  vector<tuple<int, vector<uint64_t>, dtype_t>> state_shapes;
  for(auto const& [train_id, _]: train.init_fills) {
    auto const& node = train.graph.nodes[train_id];
    state_shapes.emplace_back(
      train_id,
      node.op.out_shape(),
      node.op.out_dtype());
  }

  {
    auto const& node = train.graph.nodes[train.sampling];
    state_shapes.emplace_back(
      train.sampling,
      node.op.out_shape(),
      node.op.out_dtype());
  }

  auto validate = make_validate_info(
    batch_validate, state_shapes, train.mean_logits, model_config, autoplace_config);

  return xtreme_t {
    .train = std::move(train),
    .validate = std::move(validate)
  };
}

tensor_t compute_mse(graph_writer_t& writer, tensor_t x, tensor_t y)
{
  scalarop_t difference = scalarop_t::combine(
    scalarop_t::make_square(),
    { scalarop_t::make_sub () });

  tensor_t loss = writer.straight_bew(difference, x, y);

  uint64_t nelem = product(loss.get_shape()());
  loss = loss.scale(write_with_ss(1.0/double(nelem)));

  return loss;
}

int graph_insert_einsummable_ew(
  graph_t& graph,
  scalarop_t join,
  vector<int> const& inns)
{
  vector<uint64_t> shape = graph.out_shape(inns[0]);
  int rank = shape.size();

  einsummable_t e(
    shape,
    vector<vector<int>>(inns.size(), vector_iota<int>(rank)),
    rank,
    join);

  return graph.insert_einsummable(e, inns);
}

// void main_rank_zero(
//   gpu_mg_server_t* server,
//   random_inserter_t& random_inserter,
//   args_t& args,
//   autoplace_config_t config)
// {
//   uint64_t batch = args.get<uint64_t>("batch");

//   model_config_t model_config; 
//   model_config.num_features = args.get<uint64_t>("num_features");
//   model_config.num_labels   = args.get<uint64_t>("num_labels");
//   model_config.dim_hidden   = args.get<vector<uint64_t>>("dim_hidden");

//   graph_writer_t writer;
//   int label_id;
//   {
//     model_t model(writer, model_config);

//     tensor_t inn_data = writer.input({batch, model_config.num_features});
//     tensor_t predictions = model.forward(inn_data);
//     predictions.save_inplace();
//     // DOUT("predictions: " << predictions.get_id());
//     tensor_t out_data = writer.input({batch, model_config.num_labels});
//     label_id = out_data.get_id();

//     tensor_t mse = compute_mse(writer, out_data, predictions);
//     mse.save_inplace();
//     // DOUT("mse: " << mse.get_id());

//     vector<tensor_t> weights = vector_concatenate(
//       model.get_ff_weights(), model.get_norm_weights());
//     vector<tensor_t> grads = writer.backprop(mse, weights);

//     scalarop_t grad_update = scalarop_t::combine(
//       scalarop_t::make_sub(),
//       {
//         scalarop_t::make_identity(default_dtype()),
//         scalarop_t::make_scale(scalar_t(default_dtype(), write_with_ss(1e-6)))
//       }
//     );

//     for(auto const& [w,g]: vector_zip(weights, grads)) {
//       tensor_t new_w = writer.straight_bew(grad_update, w, g);
//       new_w.save_inplace();
//       // DOUT("new_w: " << new_w.get_id());
//     }
//   }

//   auto const& graph = writer.get_graph();
//   // print the graph
//   std::ofstream f("extreme_graph.gv");
//   graph.print_graphviz(f);
//   DOUT("printed extreme_graph.gv");
//   // {
//   //   map<int, dbuffer_t> inn_data;
//   //   for(int gid = 0; gid != graph.nodes.size(); ++gid){
//   //     auto const& node = graph.nodes[gid];
//   //     if (node.op.is_input()){
//   //       uint64_t nelem = product(node.op.out_shape());
//   //       dbuffer_t d = make_dbuffer(node.op.out_dtype(), nelem);
//   //       d.zeros();
//   //       inn_data.insert({gid, d});
//   //     }
//   //   }
//   //   auto out_data = reference_compute_graph(graph, inn_data);
//   //   for (auto const& [gid, d]: out_data){
//   //     auto nelem = product(graph.nodes[gid].op.out_shape());
//   //     DOUT("gid: " << gid << " Value: " << d.sum_to_f64() << " Num elements: " << nelem);
//   //   }
//   // }
//   /////////////////////////

//   vector<placement_t> pls = autoplace01(graph, config);
//   // for(auto const pl: pls) {
//   //   DOUT(pl.partition);
//   // }

//   // DLINE;
//   scalar_t lower(default_dtype(), "-0.0000001");
//   scalar_t upper(default_dtype(),  "0.0000001");
//   for(int gid = 0; gid != graph.nodes.size(); ++gid) {
//     auto const& node = graph.nodes[gid];
//     if(node.op.is_input()) {
//       if (gid == label_id){
//         random_inserter(gid, pls.at(gid), 
//           scalar_t::zero(default_dtype()), scalar_t::one(default_dtype()));
//       } else {
//         random_inserter(gid, pls.at(gid), lower, upper);
//       }
//     }
//   }
//   // DLINE;
//   server->execute_graph(graph, pls);

//   // for (int gid: server->get_gids()) {
//   //   double value = server->get_tensor_from_gid(gid).sum_to_f64();
//   //   DOUT("gid: " << gid << " Value: " << value);
//   // }
// }

void main_rank_zero(
  int world_size,
  server_base_t* server,
  xtreme_dist_t& xtreme_dist,
  args_t& args)
{
  args.set_default<int>     ("num_hidden", 4);
  args.set_default<uint64_t>("dim_hidden", 32768);
  args.set_default<uint64_t>("batch_train",    32);
  args.set_default<uint64_t>("batch_validate", 1024);

  int num_hidden = args.get<int>("num_hidden");
  uint64_t dim_hidden = args.get<uint64_t>("dim_hidden");

  xtreme_file_reader_t validate_reader(
    args.get<string>("validate_file"),
    args.get<string>("validate_counts"));
  uint64_t num_test  = validate_reader.get_num_datum();
  uint64_t num_features = validate_reader.get_num_features();
  uint64_t num_labels   = validate_reader.get_num_labels();

  xtreme_file_reader_t train_reader(
    args.get<string>("train_file"),
    args.get<string>("train_counts"));
  uint64_t num_train = train_reader.get_num_datum();
  if(num_features != train_reader.get_num_features()) {
    throw std::runtime_error("miscmatch number of features");
  }
  if(num_labels != train_reader.get_num_labels()) {
    throw std::runtime_error("miscmatch number of labels");
  }

  model_config_t model_config {
    .num_features = num_features,
    .num_labels   = num_labels,
    .dim_hidden   = vector<uint64_t>(num_hidden, dim_hidden)
  };

  args.set_default<bool>("use_momentum", true);
  bool use_momentum = args.get<bool>("use_momentum");

  updater_desc_t updater_desc = use_momentum ? 
    updater_desc_t {
      .dtype = default_dtype(),
      .t = updater_desc_t::momentum_t {}
    }                                        :
    updater_desc_t {
      .dtype = default_dtype(),
      .t = updater_desc_t::vanilla_t { }
    }                                        ;

  args.set_default<float>("eta", 0.5);
  scalar_t _lr(default_dtype(), write_with_ss(args.get<float>("learning_rate")));
  scalar_t _eta(default_dtype(), write_with_ss(args.get<float>("eta")));

  map<string, scalar_t> scalar_vars {
    { "eta", _eta },
    { "learning_rate", _lr },
  };

  int num_config_threads_per_machine = args.get<int>("config_threads");
  autoplace_config_t autoplace_config = autoplace_config_t::make_default01(
    world_size, num_config_threads_per_machine);

  uint64_t batch_train    = args.get<uint64_t>("batch_train");
  uint64_t batch_validate = args.get<uint64_t>("batch_validate");

  float regularize_scale = args.get<float>("regularize_scale");

  xtreme_t info = xtreme_t::make(
    batch_train,
    batch_validate,
    regularize_scale,
    model_config,
    updater_desc,
    autoplace_config);

  // TODO: remove
  // {
  //   std::ofstream f("g.gv");
  //   info.train.graph.print_graphviz(f);
  //   DOUT("printed g.gv");

  //   DOUT("mse id is " << info.train.mse);
  //   DOUT("reg id is " << info.train.regular);
  //   DOUT("los id is " << info.train.loss);
  //   DOUT("info.train.grads " << info.train.grads);
  // }

  {
    args.set_default<double>("alpha", 0.5);
    double alpha = args.get<double>("alpha");

    auto const& counts = train_reader.get_counts();

    server->insert_tensor(
      info.train.sampling,
      info.train.inn_rels.at(info.train.sampling),
      compute_sampling(alpha, counts).copy(default_dtype()));

    server->insert_tensor(
      info.train.mean_logits,
      info.train.inn_rels.at(info.train.mean_logits),
      compute_mean_logits(counts).copy(default_dtype()));
  }

  // initialize the weights
  for(int const& w_id: info.train.ff_weights) {
    xtreme_dist.insert_random(w_id, info.train.inn_rels.at(w_id));
  }
  for(int const& w_id: info.train.norm_weights) {
    server->insert_constant(
      w_id, info.train.inn_rels.at(w_id), scalar_t::one(default_dtype()));
  }

  // fill with random values
  for(auto const& [gid, fill]: info.train.init_fills) {
    if(!fill.is_constant()) {
      throw std::runtime_error("this fill must be constant");
    }
    auto const& c = fill.get_constant();
    server->insert_constant(gid, info.train.inn_rels.at(gid), c.value);
  }

  args.set_default<int>("num_runs", 2);
  int num_runs = args.get<int>("num_runs");

  args.set_default<int>("num_trains_per_run", 2);
  int num_trains_per_run = args.get<int>("num_trains_per_run");

  int iter = 1;
  for(int which_run = 0; which_run != num_runs; ++which_run) {
    for(int which_train = 0; which_train != num_trains_per_run; ++which_train) {
      vector<datum_t> data = train_reader(batch_train);

      // Insert inn_data(features) and out_data(labels) for this batch
      xtreme_dist.insert_features(
        info.train.inn_data,
        default_dtype(), info.train.get_placement(info.train.inn_data),
        data);
      xtreme_dist.insert_labels(
        info.train.out_data,
        default_dtype(), info.train.get_placement(info.train.out_data),
        data);

      server->remap(info.train.inn_rels);

      update_vars(updater_desc, iter++, scalar_vars);
      server->execute(
        info.train.taskgraph,
        info.train.out_rels,
        scalar_vars);

      double mse = server->get_tensor_from_gid(info.train.mse).sum_to_f64();
      DOUT("mse:  " << mse);
      double regu = server->get_tensor_from_gid(info.train.regular).sum_to_f64();
      DOUT("regu: " << regu);
      double loss = server->get_tensor_from_gid(info.train.loss).sum_to_f64();
      DOUT("loss: " << loss);

      for(auto const& gid: info.train.grads) {
        auto tensor = server->get_tensor_from_gid(gid);
        double mn = tensor.min().convert(dtype_t::f64).f64();
        double mx = tensor.max().convert(dtype_t::f64).f64();
        double av = tensor.sum_to_f64() / double(tensor.nelem());
        auto shape = info.train.graph.nodes[gid].op.out_shape();
        DOUT("grad " << gid << ": [" << mn << "," << av << ", " << mx << "]  " << shape);
      }

      if(std::isnan(loss) || std::isinf(loss)) {
        throw std::runtime_error("loss is nan or inf");
      }

      server->remap_gids(info.train.next_iter_remap);

      for(auto const& gid: info.train.ff_weights) {
        auto tensor = server->get_tensor_from_gid(gid);
        double mn = tensor.min().convert(dtype_t::f64).f64();
        double mx = tensor.max().convert(dtype_t::f64).f64();
        double av = tensor.sum_to_f64() / double(tensor.nelem());
        auto shape = info.train.graph.nodes[gid].op.out_shape();
        DOUT("ff " << gid << ": [" << mn << "," << av << "," << mx << "]  " << shape);
      }
    }

    server->remap_gids(info.remap_to_validate());

    vector<datum_t> data = validate_reader(batch_validate);
    validate_reader.to_beginning(); // TODO: how to use validate reader?

    // Insert inn_data(features) and out_data(labels) for this batch
    xtreme_dist.insert_features(
      info.validate.inn_data,
      default_dtype(), info.validate.get_placement(info.train.inn_data),
      data);

    server->remap(info.validate.inn_rels);

    server->execute(
      info.validate.taskgraph,
      info.validate.out_rels);

    auto scores = compute_scores(
      validate_reader.get_counts(),
      server->get_tensor_from_gid(info.validate.predictions),
      data);
    scores.print(std::cout);

    DOUT("");

    server->remap_gids(info.remap_from_validate());
  }
}
