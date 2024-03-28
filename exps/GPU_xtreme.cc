#include "../src/base/args.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/reference.h"
#include <cstdint>

using tensor_t = graph_writer_t::tensor_t;

struct random_inserter_t;

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

  graph_writer_t& writer;

  vector<ff_t> ffs;
  vector<rms_norm_t> norms;
  sigmoid_out_t out_layer;
};

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

void main_rank_zero(
  gpu_mg_server_t* server,
  random_inserter_t& random_inserter,
  args_t& args,
  autoplace_config_t config)
{
  uint64_t batch = args.get<uint64_t>("batch");

  model_config_t model_config; 
  model_config.num_features = args.get<uint64_t>("num_features");
  model_config.num_labels   = args.get<uint64_t>("num_labels");
  model_config.dim_hidden   = args.get<vector<uint64_t>>("dim_hidden");

  graph_writer_t writer;
  int label_id;
  {
    model_t model(writer, model_config);

    tensor_t inn_data = writer.input({batch, model_config.num_features});
    tensor_t predictions = model.forward(inn_data);
    predictions.save_inplace();
    // DOUT("predictions: " << predictions.get_id());
    tensor_t out_data = writer.input({batch, model_config.num_labels});
    label_id = out_data.get_id();

    tensor_t mse = compute_mse(writer, out_data, predictions);
    mse.save_inplace();
    // DOUT("mse: " << mse.get_id());

    vector<tensor_t> weights = vector_concatenate(
      model.get_ff_weights(), model.get_norm_weights());
    vector<tensor_t> grads = writer.backprop(mse, weights);

    scalarop_t grad_update = scalarop_t::combine(
      scalarop_t::make_sub(),
      {
        scalarop_t::make_identity(default_dtype()),
        scalarop_t::make_scale(scalar_t(default_dtype(), write_with_ss(1e-6)))
      }
    );

    for(auto const& [w,g]: vector_zip(weights, grads)) {
      tensor_t new_w = writer.straight_bew(grad_update, w, g);
      new_w.save_inplace();
      // DOUT("new_w: " << new_w.get_id());
    }
  }

  auto const& graph = writer.get_graph();
  // print the graph
  std::ofstream f("extreme_graph.gv");
  graph.print_graphviz(f);
  DOUT("printed extreme_graph.gv");
  // {
  //   map<int, dbuffer_t> inn_data;
  //   for(int gid = 0; gid != graph.nodes.size(); ++gid){
  //     auto const& node = graph.nodes[gid];
  //     if (node.op.is_input()){
  //       uint64_t nelem = product(node.op.out_shape());
  //       dbuffer_t d = make_dbuffer(node.op.out_dtype(), nelem);
  //       d.zeros();
  //       inn_data.insert({gid, d});
  //     }
  //   }
  //   auto out_data = reference_compute_graph(graph, inn_data);
  //   for (auto const& [gid, d]: out_data){
  //     auto nelem = product(graph.nodes[gid].op.out_shape());
  //     DOUT("gid: " << gid << " Value: " << d.sum_to_f64() << " Num elements: " << nelem);
  //   }
  // }
  /////////////////////////

  vector<placement_t> pls = autoplace01(graph, config);
  // for(auto const pl: pls) {
  //   DOUT(pl.partition);
  // }

  // DLINE;
  scalar_t lower(default_dtype(), "-0.0000001");
  scalar_t upper(default_dtype(),  "0.0000001");
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      if (gid == label_id){
        random_inserter(gid, pls.at(gid), 
          scalar_t::zero(default_dtype()), scalar_t::one(default_dtype()));
      } else {
        random_inserter(gid, pls.at(gid), lower, upper);
      }
    }
  }
  // DLINE;
  server->execute_graph(graph, pls);

  // for (int gid: server->get_gids()) {
  //   double value = server->get_tensor_from_gid(gid).sum_to_f64();
  //   DOUT("gid: " << gid << " Value: " << value);
  // }
}
