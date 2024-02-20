#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"

#include "../src/misc/update.h"

#include "../src/autoplace/autoplace.h"

#include "../src/einsummable/gwriter.h"

using tensor_t = graph_writer_t::tensor_t;

struct datum_t;

struct xtreme_dist_t {
  xtreme_dist_t(
    communicator_t& comm,
    server_base_t* s)
    : communicator(comm), server(s), register_cmd(s->get_registered_cmd())
  {}

  void insert_random(int gid, relation_t const& rel);

  void insert_labels(  int gid, relation_t const& rel, vector<datum_t> const& data);

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
};

void main_rank_zero(
  int world_size,
  server_base_t* server,
  xtreme_dist_t& xtreme_dist,
  args_t& args);

int main(int argc, char** argv) {
  int expected_argc = 7;
  if(argc < expected_argc) {
    return 1;
  }

  set_default_dtype(dtype_t::f32);

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int num_threads = parse_with_ss<int>(argv[5]);
  int num_contraction_threads = parse_with_ss<int>(argv[6]);

  std::unique_ptr<server_base_t> server(
    new cpu_tg_server_t(communicator, mem_size, num_threads));

  xtreme_dist_t xtreme_dist(communicator, server.get());

  if(!is_rank_zero) {
    server->register_listen(
      xtreme_dist.insert_random_cmd(),
      [&]{ xtreme_dist.client_insert_random(); });
    server->register_listen(
      xtreme_dist.insert_labels_cmd(),
      [&]{ xtreme_dist.client_insert_labels(); });
    server->register_listen(
      xtreme_dist.insert_features_cmd(),
      [&]{ xtreme_dist.client_insert_features(); });

    server->listen();
    return 0;
  }

  args_t args(argc-(expected_argc-1), argv+(expected_argc-1));

  main_rank_zero(world_size, server.get(), xtreme_dist, args);

  server->shutdown();

  return 0;
}

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
    scalarops.back() = scalarop_t::make_sigmoid();

    for(int i = 0; i != scalarops.size(); ++i) {
      layers.emplace_back(writer, szs[i], szs[i+1], scalarops[i]);
    }
  }

  tensor_t forward(tensor_t data)
  {
    tensor_t x = data;
    for(auto& layer: layers) {
      x = layer.forward(x);
    }
    return x;
  }

  vector<tensor_t> get_weights() const {
    return vector_from_each_member(layers, tensor_t, weight);
  }

  graph_writer_t& writer;

  vector<ff_t> layers;
};

tensor_t mse(graph_writer_t& writer, tensor_t sampling, tensor_t x, tensor_t y)
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
    // (4) weights
    // (5) vector_mapfst(init_fills)   <- "constants"
    graph_t graph;
    int inn_data;
    int out_data;
    int sampling;
    int loss;
    vector<int> weights;
    vector<tuple<int, fill_t>> init_fills;

    map<int, relation_t> inn_rels;
    taskgraph_t taskgraph;
    map<int, relation_t> out_rels;
    vector<tuple<int, int>> next_iter_remap;
  };

  struct validate_t {
    graph_t graph;
    int inn_data;
    vector<int> weights;
    vector<tuple<int, int>> constants_tid_to_vid;
    int predictions;

    map<int, relation_t> inn_rels;
    taskgraph_t taskgraph;
    map<int, relation_t> out_rels;
  };

  vector<tuple<int, int>> remap_to_validate() const {
    vector<tuple<int, int>> ret = validate.constants_tid_to_vid;
    for(int i = 0; i != train.weights.size(); ++i) {
      ret.emplace_back(train.weights[i], validate.weights[i]);
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
    model_config_t const& model_config,
    updater_desc_t const& updater_desc,
    autoplace_config_t const& autoplace_config);

  static
  train_t make_train_info(
    uint64_t batch,
    model_config_t const& model_config,
    updater_desc_t const& updater_desc,
    autoplace_config_t const& autoplace_config);

  static
  validate_t make_validate_info(
    uint64_t batch,
    vector<tuple<int, vector<uint64_t>, dtype_t>> const& constants,
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
  model_config_t const& model_config,
  updater_desc_t const& updater_desc,
  autoplace_config_t const& autoplace_config)
{
  graph_writer_t writer;
  model_t model(writer, model_config);

  tensor_t inn_data = writer.input({batch, model_config.num_features});
  tensor_t predictions = model.forward(inn_data);
  tensor_t out_data = writer.input({batch, model_config.num_labels});

  tensor_t sampling = writer.input(vector<uint64_t>{model_config.num_labels});
  sampling.save_inplace();

  tensor_t loss = mse(writer, sampling, out_data, predictions);
  loss.save_inplace();

  vector<tensor_t> weights = model.get_weights();
  vector<tensor_t> grads = writer.backprop(loss, weights);

  xtreme_t::train_t ret {
    .graph = writer.get_graph(),
    .inn_data = inn_data.get_id(),
    .out_data = out_data.get_id(),
    .sampling = sampling.get_id(),
    .loss = loss.get_id(),
    .weights = vector_from_each_method(weights, int, get_id),
  };

  vector<tuple<int, int>> old_news;
  ret.init_fills = update_weights(
    updater_desc, ret.graph, old_news, ret.weights,
    vector_from_each_method(grads, int, get_id));

  for(auto const& [old_id, new_id]: old_news) {
    ret.graph.nodes[new_id].op.set_save(true);
    ret.next_iter_remap.emplace_back(new_id, old_id);
  }
  // At this point, next_iter_remap includes
  // (1) weights
  // (2) states added by the updater
  ret.next_iter_remap.emplace_back(sampling.get_id(), sampling.get_id());
  // Now also make sure to include sampling so that it won't get deleted

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
  model_config_t const& model_config,
  autoplace_config_t const& autoplace_config)
{
  graph_writer_t writer;
  model_t model(writer, model_config);

  tensor_t inn_data = writer.input({batch, model_config.num_features});
  tensor_t predictions = model.forward(inn_data);
  predictions.save_inplace();

  vector<tensor_t> weights = model.get_weights();
  for(auto& w: weights) {
    w.save_inplace();
  }

  vector<tuple<int, int>> constant_ids;
  for(auto const& [train_id, shape, dtype]: constants) {
    tensor_t t = writer.input(shape, dtype);
    t.save_inplace();
    constant_ids.emplace_back(train_id, t.get_id());
  }

  xtreme_t::validate_t ret {
    .graph = writer.get_graph(),
    .inn_data = inn_data.get_id(),
    .weights = vector_from_each_method(weights, int, get_id),
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
  model_config_t const& model_config,
  updater_desc_t const& updater_desc,
  autoplace_config_t const& autoplace_config)
{
  auto train = make_train_info(batch_train, model_config, updater_desc, autoplace_config);

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
    batch_validate, state_shapes, model_config, autoplace_config);

  return xtreme_t {
    .train = std::move(train),
    .validate = std::move(validate)
  };
}

struct pscores_t {
  double p1;
  double p3;
  double p5;
  double psp1;
  double psp3;
  double psp5;

  std::ostream& print(std::ostream& out) const {
    out << "p1:   " << p1   << std::endl;
    out << "p3:   " << p3   << std::endl;
    out << "p5:   " << p5   << std::endl;
    out << "psp1: " << psp1 << std::endl;
    out << "psp3: " << psp3 << std::endl;
    out << "psp5: " << psp5 << std::endl;

    return out;
  }
};

uint64_t istream_consume_uint(std::istream& inn) {
  string ret;
  while(inn) {
    char c = inn.peek();
    if(isdigit(c)) {
      inn.get();
      ret.push_back(c);
    } else {
      break;
    }
  }

  if(ret.size() == 0) {
    throw std::runtime_error("did not read any numeric");
  }

  return parse_with_ss<uint64_t>(ret);
}

// A simple float is just <num>.<num>
double istream_consume_simple_double(std::istream& inn) {
  string ret;
  while(inn) {
    char c = inn.peek();
    if(isdigit(c)) {
      inn.get();
      ret.push_back(c);
    } else if(c == '.') {
      inn.get();
      ret.push_back(c);
      break;
    }
  }
  if(ret.size() == 0) {
    throw std::runtime_error("did not read any numeric");
  }
  bool d = false;
  while(inn) {
    char c = inn.peek();
    if(isdigit(c)) {
      inn.get();
      ret.push_back(c);
      d = true;
    } else {
      break;
    }
  }
  if(!d) {
    throw std::runtime_error("did not read numeric after decimal");
  }
  return parse_with_ss<double>(ret);
}

vector<int> read_counts_file(string filename) {
  std::ifstream file(filename);
  if(!file) {
    throw std::runtime_error("could not open counts file");
  }

  vector<int> ret;
  while(file) {
    ret.push_back(int(istream_consume_uint(file)));
    if(!file) {
      break;
    }
    char c = file.get();
    if(c != ',') {
      break;
    }
  }

  return ret;
}

tuple<uint64_t, uint64_t, uint64_t>
read_header(std::ifstream& file) {
  tuple<uint64_t, uint64_t, uint64_t> ret;
  auto& [a,b,c] = ret;
  a = istream_consume_uint(file);
  istream_expect(file, " ");
  b = istream_consume_uint(file);
  istream_expect(file, " ");
  c = istream_consume_uint(file);
  istream_expect(file, "\n");

  return ret;
}

struct datum_t {
  vector<uint64_t> labels;
  vector<tuple<uint64_t, double>> scores;
};

std::ostream& operator<<(std::ostream& out, datum_t const& d) {
  out << "d|" << d.labels << "|" << d.scores;
  return out;
}

datum_t read_datum(std::ifstream& file) {
  datum_t ret;

  while(true) {
    ret.labels.push_back(istream_consume_uint(file));

    int which = istream_expect_or(file, { ",", " ", "\n" });
    if(which == 1) {
      break;
    }
    if(which == 2) {
      // Should this even occur?
      return ret;
    }
  }

  while(true) {
    ret.scores.emplace_back();
    auto& [u,d] = ret.scores.back();
    u = istream_consume_uint(file);
    istream_expect(file, ":");
    d = istream_consume_simple_double(file);
    int which = istream_expect_or(file, { " ", "\n" });
    if(which == 1) {
      break;
    }
  }

  return ret;
}

struct xtreme_file_reader_t {
  xtreme_file_reader_t(
    string const& data_filename,
    string const& counts_filename)
    : file(data_filename),
      counts(read_counts_file(counts_filename)),
      iter(0)
  {
    auto [num_datum_, num_features_, num_labels_] = read_header(file);
    num_datum = num_datum_;
    num_features = num_features_;
    num_labels = num_labels_;

    if(counts.size() != num_labels) {
      throw std::runtime_error("num labels != counts.size()");
    }
  }

  datum_t operator()() {
    if(iter == num_datum) {
      to_beginning();
    }

    datum_t ret = read_datum(file);
    iter++;
    return ret;
  }

  vector<datum_t> operator()(uint64_t n) {
    vector<datum_t> ret;
    ret.reserve(n);
    for(int i = 0; i != n; ++i) {
      ret.push_back(this->operator()());
    }
    return ret;
  }

  vector<int> const& get_counts() const { return counts; }

  uint64_t get_num_datum()    const { return num_datum; }
  uint64_t get_num_features() const { return num_features; }
  uint64_t get_num_labels()   const { return num_labels; }

private:
  std::ifstream file;
  vector<int> counts;
  int iter;

  uint64_t num_datum;
  uint64_t num_features;
  uint64_t num_labels;

  void to_beginning() {
    iter = 0;
    file.clear();
    file.seekg(0);
    read_header(file);
  }
};

template <typename T>
vtensor_t<uint64_t> _get_top_choices(
  T const* data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  topn = std::min(topn, ncol);
  vtensor_t<uint64_t> ret({int(nrow), int(topn)});
  uint64_t* ret_vec = ret.get().data();
  for(int i = 0; i != nrow; ++i) {
    T const*  d = data    + i*ncol;
    uint64_t* r = ret_vec + i*topn;
    vector<T const*> tops = select_topn(d, d + ncol, topn);
    for(int j = 0; j != topn; ++j) {
      r[j] = std::distance(d, tops[j]);
    }
  }
  return ret;
}

vtensor_t<uint64_t> get_top_choices(
  dbuffer_t const& data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  if(data.dtype == dtype_t::f16) {
    return _get_top_choices(data.f16(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f32) {
    return _get_top_choices(data.f32(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f64) {
    return _get_top_choices(data.f64(), nrow, ncol, topn);
  }
  throw std::runtime_error("get_top_choices: no dtype support here");
}

pscores_t compute_scores(
  vector<int> const& counts,
  dbuffer_t predictions,
  vector<datum_t> const& data)
{
  // tops should contain for for each row, the top choice, second choice, ...
  vtensor_t<uint64_t> tops = get_top_choices(predictions, data.size(), counts.size(), 5);

  vector<double> probs;
  {
    uint64_t total = 0;
    for(auto const& cnt: counts) {
      total += cnt;
    }
    for(auto const& cnt: counts) {
      probs.push_back(double(cnt) / double(total));
    }
  }

  pscores_t ret {
    .p1   = 0.0,
    .p3   = 0.0,
    .p5   = 0.0,
    .psp1 = 0.0,
    .psp3 = 0.0,
    .psp5 = 0.0
  };
  for(int b = 0; b != data.size(); ++b) {
    datum_t const& d = data[b];
    set<uint64_t> cs(d.labels.begin(), d.labels.end());
    double p = 0.0;
    double pp = 0.0;

    auto increment = [&](uint64_t const& l) {
      if(cs.count(l) > 0) {
        p += 1.0;
        pp += 1.0 / probs[l];
      }
    };

    // 0
    increment(tops(b,0));
    ret.p1   += p;
    ret.psp1 += pp;

    // 1, 2
    increment(tops(b,1));
    increment(tops(b,2));
    ret.p3   += p;
    ret.psp3 += pp;

    // 3, 4
    increment(tops(b,3));
    increment(tops(b,4));
    ret.p5   += p;
    ret.psp5 += pp;
  }

  ret.p1   /= (double(data.size()));
  ret.psp1 /= (double(data.size()));

  ret.p3   /= (double(data.size()) * 3.0);
  ret.psp3 /= (double(data.size()) * 3.0);

  ret.p5   /= (double(data.size()) * 5.0);
  ret.psp5 /= (double(data.size()) * 5.0);

  return ret;
}

// Set the sampling of l to
//   alpha * x + (1-alpha) * y
// Such that
// 1. the sum of sampling equals num_labels.
// 2. When alpha = 1.0, each label has the same weight
// 3. When alpha = 0.0, each label contributes to the sampling the same amount.
//
// So x is proportinal to 1.0
// and for y, we want the expected value of label l to be the same across all l,
// when data is pulled proportional to count[l].
// So E[l] = 1 = (count[l] / total_counts) * unnormalize_y[l].
// Solving for unormalized_y[l],
//   y[l] is proportional to 1 / count[l].
//
// So x[l] = 1.0, where sum_l x[l] = num_labels.
// y[l] = num_labels * (total / count[l]) / sum_l (total / count[l])
//      = num_labels * (total / count[l]) / inv_total
//      = y_term * (total / count[l])
dbuffer_t compute_sampling(
  double alpha,
  vector<int> const& counts)
{
  uint64_t num_labels = counts.size();

  double total = double(vector_sum(counts));
  double inv_total = 0.0;
  for(auto const& cnt: counts) {
    inv_total += total / double(cnt);
  }
  double y_term = double(num_labels) / inv_total;

  dbuffer_t d = make_dbuffer(dtype_t::f64, num_labels);
  d.fill(scalar_t(alpha));
  for(uint64_t l = 0; l != num_labels; ++l) {
    double& val = d.f64()[l];
    val += (1-alpha)*y_term*(total / double(counts[l]));
  }

  DOUT("num_labels is " << num_labels);
  DOUT("sum of d is   " << d.sum_to_f64());

  return d;
}

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
  int num_layers = args.get<int>("num_layers");
  uint64_t dim_hidden = args.get<uint64_t>("dim_hidden");

  // Amazon 3M
  uint64_t num_train = 1717899;
  uint64_t num_test  = 742507;
  uint64_t num_features = 337067;
  uint64_t num_labels   = 2812281;

  xtreme_file_reader_t validate_reader(
    args.get<string>("validate_file"),
    args.get<string>("validate_counts"));
  xtreme_file_reader_t train_reader(
    args.get<string>("train_file"),
    args.get<string>("train_counts"));

  model_config_t model_config {
    .num_features = num_features,
    .num_labels   = num_labels,
    .dim_hidden   = vector<uint64_t>(num_layers, dim_hidden)
  };

  updater_desc_t updater_desc {
    .dtype = default_dtype(),
    .t = updater_desc_t::adamw_t { .min_precision = default_dtype() }
  };

  scalar_t _lr(default_dtype(), write_with_ss(args.get<float>("learning_rate")));
  map<string, scalar_t> scalar_vars {
    { "beta1", scalar_t(default_dtype(), "0.9") },
    { "beta2", scalar_t(default_dtype(), "0.999") },
    { "eta", _lr },
    { "learning_rate", _lr },
  };

  int num_config_threads_per_machine = args.get<int>("config_threads");
  autoplace_config_t autoplace_config = autoplace_config_t::make_default01(
    world_size, num_config_threads_per_machine);

  uint64_t batch_train    = args.get<uint64_t>("batch_train");
  uint64_t batch_validate = args.get<uint64_t>("batch_validate");

  xtreme_t info = xtreme_t::make(
    batch_train,
    batch_validate,
    model_config,
    updater_desc,
    autoplace_config);

  {
    args.set_default<double>("alpha", 0.5);
    double alpha = args.get<double>("alpha");

    auto const& counts = train_reader.get_counts();

    server->insert_tensor(
      info.train.sampling,
      info.train.inn_rels.at(info.train.sampling),
      compute_sampling(alpha, counts).copy(default_dtype()));
  }

  // initialize the weights
  for(int const& w_id: info.train.weights) {
    xtreme_dist.insert_random(w_id, info.train.inn_rels.at(w_id));
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

  args.set_default<int>("num_trainers_per_run", 2);
  int num_trains_per_run = args.get<int>("num_trains_per_run");

  int iter = 1;
  for(int which_run = 0; which_run != num_runs; ++which_run) {
    for(int which_train = 0; which_train != num_trains_per_run; ++which_train) {
      vector<datum_t> data = train_reader(batch_train);

      // Insert inn_data(features) and out_data(labels) for this batch
      xtreme_dist.insert_features(
        info.train.inn_data,
        info.train.inn_rels.at(info.train.inn_data),
        data);
      xtreme_dist.insert_labels(
        info.train.out_data,
        info.train.inn_rels.at(info.train.out_data),
        data);

      server->remap(info.train.inn_rels);

      update_vars(updater_desc, iter++, scalar_vars);
      server->execute(
        info.train.taskgraph,
        info.train.out_rels,
        scalar_vars);

      double loss = server->get_tensor_from_gid(info.train.loss).sum_to_f64();
      DOUT("loss: " << loss);

      server->remap_gids(info.train.next_iter_remap);
    }

    vector<vector<int>> validate_out_data;

    server->remap_gids(info.remap_to_validate());

    vector<datum_t> data = validate_reader(batch_validate);

    // Insert inn_data(features) and out_data(labels) for this batch
    xtreme_dist.insert_features(
      info.validate.inn_data,
      info.validate.inn_rels.at(info.train.inn_data),
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

    server->remap_gids(info.remap_from_validate());
  }
}

void xtreme_dist_t::insert_random(int gid, relation_t const& rel)
{
  communicator.broadcast_string(register_cmd);
  communicator.broadcast_string(insert_random_cmd());
  communicator.broadcast_string(rel.to_wire());

  _insert_random(rel);

  server->insert_gid_without_data(gid, rel);
}

void xtreme_dist_t::insert_labels(
  int gid, relation_t const& rel, vector<datum_t> const& data)
{
  communicator.broadcast_string(register_cmd);
  communicator.broadcast_string(insert_labels_cmd());
  communicator.broadcast_string(rel.to_wire());

  vector<int> cs;
  vector<uint64_t> labels;
  cs.reserve(data.size());
  for(auto const& datum: data) {
    cs.push_back(datum.labels.size());
    vector_concatenate_into(labels, datum.labels);
  }

  std::exclusive_scan(cs.begin(), cs.end(), cs.begin(), 0);

  communicator.broadcast_vector(cs);
  communicator.broadcast_vector(labels);

  _insert_labels(rel, cs, labels);

  server->insert_gid_without_data(gid, rel);
}

void xtreme_dist_t::insert_features(
  int gid, relation_t const& rel, vector<datum_t> const& data)
{
  communicator.broadcast_string(register_cmd);
  communicator.broadcast_string(insert_features_cmd());
  communicator.broadcast_string(rel.to_wire());

  vector<int> cs;
  vector<uint64_t> features;
  vector<double> scores;
  cs.reserve(data.size());
  for(auto const& datum: data) {
    cs.push_back(datum.labels.size());
    vector_concatenate_into(features, vector_mapfst(datum.scores));
    vector_concatenate_into(scores,   vector_mapsnd(datum.scores));
  }

  std::exclusive_scan(cs.begin(), cs.end(), cs.begin(), 0);

  communicator.broadcast_vector(cs);
  communicator.broadcast_vector(features);
  communicator.broadcast_vector(scores);

  _insert_features(rel, cs, features, scores);

  server->insert_gid_without_data(gid, rel);
}

void xtreme_dist_t::client_insert_random() {
  _insert_random(relation_t::from_wire(communicator.recv_string(0)));
}

void xtreme_dist_t::client_insert_labels() {
  _insert_labels(
    relation_t::from_wire(communicator.recv_string(0)),
    communicator.recv_vector<int>(0),
    communicator.recv_vector<uint64_t>(0));
}

void xtreme_dist_t::client_insert_features() {
  _insert_features(
    relation_t::from_wire(communicator.recv_string(0)),
    communicator.recv_vector<int>(0),
    communicator.recv_vector<uint64_t>(0),
    communicator.recv_vector<double>(0));
}

void xtreme_dist_t::_insert_random(relation_t const& rel)
{
  auto const& [dtype, pl, tids_] = rel;

  auto const& tids      = tids_.get();
  auto const& partition = pl.partition;
  auto const& locs      = pl.locations.get();

  vector<uint64_t> block_nelems = partition.all_block_sizes().get();

  if(tids.size() != locs.size() || tids.size() != block_nelems.size()) {
    throw std::runtime_error("should not occur: _insert_random");
  }

  int this_rank = communicator.get_this_rank();

  map<int, tuple<int, buffer_t>> data;
  for(int i = 0; i != locs.size(); ++i) {
    if(server->is_local_location(locs[i])) {
      dbuffer_t d = make_dbuffer(dtype, block_nelems[i]);
      d.random("-0.0001", "0.0001");
      data.insert({tids[i], {locs[i], d.data}});
    }
  }

  server->local_insert_tensors(data);
}

template <typename T>
tuple<typename vector<T>::const_iterator,typename vector<T>::const_iterator>
_exclusive_sum_access(
  vector<int> const& exclusive_sum,
  vector<T> const& items,
  int b)
{
  auto ii = items.begin();
  auto beg = ii + exclusive_sum.at(b);
  if(b + 1 == exclusive_sum.size()) {
    return { beg, items.end() };
  } else {
    return { beg, ii + exclusive_sum[b+1] };
  }
}

void xtreme_dist_t::_insert_labels(
  relation_t const& rel,
  vector<int> const& exclusive_sum,
  vector<uint64_t> const& labels)
{
  auto get_labels = [&](uint64_t b) {
    return _exclusive_sum_access(exclusive_sum, labels, b);
  };

  auto const& [dtype, pl, tids] = rel;
  auto const& partition = pl.partition;
  auto const& locs      = pl.locations;

  auto block_shape = partition.block_shape();
  if(block_shape.size() != 2) {
    throw std::runtime_error("unexpected in insert labels");
  }

  map<int, tuple<int, buffer_t>> data;

  int n_block_row = block_shape[0];
  int n_block_col = block_shape[1];
  for(int block_row = 0; block_row != n_block_row; ++block_row) {
  for(int block_col = 0; block_col != n_block_col; ++block_col) {
    int const& loc = locs(block_row, block_col);
    if(!server->is_local_location(loc)) {
      continue;
    }

    auto [b_beg, b_end] = partition.partdims[0].which_vals(block_row);
    auto [l_beg, l_end] = partition.partdims[1].which_vals(block_col);
    uint64_t local_b_sz = b_end - b_beg;
    uint64_t local_l_sz = l_end - l_beg;

    dbuffer_t d = make_dbuffer(dtype_t::f64, local_b_sz*local_l_sz);
    d.zeros();

    uint64_t local_b = 0;
    for(uint64_t b = b_beg; b != b_end; ++b, ++local_b) {
      double* ret = d.f64() + local_b*local_l_sz;
      auto [beg_ls, end_ls] = get_labels(b);
      for(auto iter = beg_ls; iter != end_ls; ++iter) {
        auto const& label = *iter;
        if(label >= l_beg && label < l_end) {
          uint64_t local_l = label - l_beg;
          ret[local_l] = 1.0;
        }
      }
    }

    int const& tid = tids(block_row, block_col);
    data.insert({tid, {loc, d.copy(dtype).data}});
  }}

  server->local_insert_tensors(data);
}

void xtreme_dist_t::_insert_features(
  relation_t const& rel,
  vector<int>      const& exclusive_sum,
  vector<uint64_t> const& features,
  vector<double>   const& scores)
{
  auto get_features = [&](uint64_t b) {
    return _exclusive_sum_access(exclusive_sum, features, b);
  };
  auto get_scores = [&](uint64_t b) {
    return _exclusive_sum_access(exclusive_sum, scores, b);
  };

  auto const& [dtype, pl, tids] = rel;
  auto const& partition = pl.partition;
  auto const& locs      = pl.locations;

  auto block_shape = partition.block_shape();
  if(block_shape.size() != 2) {
    throw std::runtime_error("unexpected in insert labels");
  }

  map<int, tuple<int, buffer_t>> data;

  int n_block_row = block_shape[0];
  int n_block_col = block_shape[1];
  for(int block_row = 0; block_row != n_block_row; ++block_row) {
  for(int block_col = 0; block_col != n_block_col; ++block_col) {
    int const& loc = locs(block_row, block_col);
    if(!server->is_local_location(loc)) {
      continue;
    }

    auto [b_beg, b_end] = partition.partdims[0].which_vals(block_row);
    auto [f_beg, f_end] = partition.partdims[1].which_vals(block_col);
    uint64_t local_b_sz = b_end - b_beg;
    uint64_t local_f_sz = f_end - f_beg;

    dbuffer_t d = make_dbuffer(dtype_t::f64, local_b_sz*local_f_sz);
    d.zeros();

    uint64_t local_b = 0;
    for(uint64_t b = b_beg; b != b_end; ++b, ++local_b) {
      double* ret = d.f64() + local_b*local_f_sz;
      auto [iter_f, end_f] = get_features(b);
      auto [iter_s, end_s] = get_scores(b);
      for(; iter_f != end_f; ++iter_f, ++iter_s) {
        auto const& feature = *iter_f;
        auto const& score   = *iter_s;
        if(feature >= f_beg && feature < f_end) {
          uint64_t local_f = feature - f_beg;
          ret[local_f] = score;
        }
      }
    }

    int const& tid = tids(block_row, block_col);
    data.insert({tid, {loc, d.copy(dtype).data}});
  }}

  server->local_insert_tensors(data);
}
