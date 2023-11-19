#include "../src/base/setup.h"

#include "../src/base/args.h"

#include "../src/einsummable/graph.h"

#include "../src/engine/communicator.h"
#include "../src/server/cpu/server.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/loadbalanceplace.h"
#include "../src/autoplace/alocate.h"
#include "../src/autoplace/autolinns.h"
#include "../src/autoplace/autolinns2.h"

#include "../llama/modules.h"

#include <fstream>

void usage() {
  std::cout << "Setup usage: addr_zero is_client world_size memsize(GB)\n";
  std::cout << "Plus args for graphs\n";
}

tuple<
  graph_t, 
  optional<vector<placement_t>>>
build_graph(args_t& args);

vector<placement_t> autoplace(
  graph_t const& graph,
  int world_size,
  int num_threads_per,
  int max_branching = 1,
  parts_space_t space = parts_space_t::contraction,
  bool double_workers = false);

void main0(int argc, char** argv) {
  if(argc < 4) {
    usage();
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  int num_threads = std::max(1, int(std::thread::hardware_concurrency()));

  // TODO: how to pick num_channels and num channels per move?
  int num_channels = 32;
  int num_channels_per_move = 1;

  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  args_t args(argc-4, argv+4);

  exec_state_t::priority_t priority_type;
  {
    args.set_default("priority_type", "given");
    string val = args.get<string>("priority_type");
    if(val == "given") {
      priority_type = exec_state_t::priority_t::given;
    } else if(val == "bfs") {
      priority_type = exec_state_t::priority_t::bfs;
    } else if(val == "dfs") {
      priority_type = exec_state_t::priority_t::dfs;
    } else if(val == "random") {
      priority_type = exec_state_t::priority_t::random;
    } else {
      throw std::runtime_error("invalid exec state value");
    }
  }

  if(is_rank_zero) {
    DOUT("world size:                      " << world_size);
    DOUT("memory allocated:                " << (mem_size/GB) << " GB");
    DOUT("number of threads in threadpool: " << num_threads);
    DOUT("number of channels per move:     " << num_channels_per_move);
    DOUT("number of channels               " << num_channels);
    DOUT("dtype:                           " << default_dtype());
  }

  cpu_mg_server_t server(
    communicator, mem_size, num_threads, num_channels_per_move, priority_type);

  if(is_rank_zero) {
    args.set_default("pp", true);
    server.set_parallel_partialize(args.get<bool>("pp"));

    server.set_use_storage(false);

    parts_space_t space;
    {
      args.set_default("space", "contraction");
      string space_ = args.get<string>("space");
      if(space_ == "contraction") {
        space = parts_space_t::contraction;
      } else if(space_ == "all") {
        space = parts_space_t::all;
      } else if(space_ == "all_range") {
        space = parts_space_t::all_range;
      }
    }

    args.set_default("double_workers", false);
    bool double_workers = args.get<bool>("double_workers");

    args.set_default("max_branching", int(1));
    int max_branching = args.get<int>("max_branching");

    args.set_default<int>("nrep", 1);
    int nrep = args.get<int>("nrep");

    vector<tuple<string, string>> run_these {
      {"attention", "7B"},
      {"attention", "13B"},
      {"attention", "65B"},
      {"attention", "30B"},
      {"attention", "65B"},
      {"aff",       "7B"},
      {"aff",       "13B"},
      {"aff",       "30B"},
      {"aff",       "65B"}
    };
    for(auto const& [graph_, model_]: run_these) {
      DLINEOUT(graph_ << " " << model_);
      args.insert_arg("graph", graph_);
      args.insert_arg("model", model_);

      // execute this
      auto [graph, maybe_placements] = build_graph(args);
      vector<placement_t> pls;
      if(!maybe_placements) {
        pls = autoplace(
          graph, world_size, num_threads, max_branching, space, double_workers);
      } else {
        pls = maybe_placements.value();
      }
  
      for(int rep = 0; rep != nrep; ++rep) {
        // initialize input tensors and distribute across the cluster
       for(int gid = 0; gid != graph.nodes.size(); ++gid) {
          auto const& node = graph.nodes[gid];
          if(node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.0001", "0.0001");
            //tensor.ones();
            server.insert_tensor(gid, pls[gid], tensor);
          }
        }
  
        // execute
        server.execute_graph(graph, pls);
  
        //for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        //  auto const& node = graph.nodes[gid];
        //  if(node.op.is_save()) {
        //    DOUT(server.get_tensor_from_gid(gid).sum_to_f64());
        //  }
        //}
      }
    }

    server.shutdown();
  } else {
    server.listen();
  }
}

void main1(int argc, char** argv) {
  args_t args(argc, argv);
  int world_size = args.get<int>("world_size");
  int num_threads = args.get<int>("num_threads_per");

  args.set_default("space", "contraction");
  string space_ = args.get<string>("space");

  parts_space_t space;
  if(space_ == "contraction") {
    space = parts_space_t::contraction;
  } else if(space_ == "all") {
    space = parts_space_t::all;
  } else if(space_ == "all_range") {
    space = parts_space_t::all_range;
  }

  args.set_default("max_branching", int(1));
  int max_branching = args.get<int>("max_branching");

  DLINEOUT("world size      : " << world_size);
  DLINEOUT("num threads per : " << num_threads);
  DLINEOUT("max branching   : " << max_branching);

  auto [graph, maybe_placements] = build_graph(args);
  vector<placement_t> pls;
  if(!maybe_placements) {
    pls = autoplace(
      graph, world_size, num_threads, max_branching, space, false);
  } else {
    pls = maybe_placements.value();
  }
}

void _print_pl_info(
  string msg,
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto [_0, _1, taskgraph] =
    taskgraph_t::make(graph, placements);

  int nlocs = 1;
  for(auto const &pl: placements) {
    for(auto const& l: pl.locations.get()) {
      nlocs = std::max(nlocs, l+1);
    }
  }

  if(msg.size() < 45) {
    msg.resize(45, ' ');
  }

  int num_input_msgs = 0;
  uint64_t num_input_bytes = 0;
  int num_core_msgs = 0;
  uint64_t num_core_bytes = 0;
  set<int> inputs_everywhere = taskgraph.get_input_everywhere_ids();
  for(int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
    auto const& node = taskgraph.nodes[tid];
    if(node.op.is_move()) {
      uint64_t sz = node.op.get_move().size;
      if(inputs_everywhere.count(tid) > 0) {
        num_input_msgs++;
        num_input_bytes += sz;
      } else {
        num_core_msgs++;
        num_core_bytes += sz;
      }
    }
  }

  vector<uint64_t> tensor_move_costs = compute_tensor_move_costs(graph, placements);
  uint64_t input_total = 0;
  uint64_t core_total = 0;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      input_total += tensor_move_costs[gid];
    } else {
      core_total += tensor_move_costs[gid];
    }
  }
  auto to_mb = [](uint64_t n) { return double(n)/1e6; };
  DOUT("(" << msg << ") input "
      << num_input_msgs << "#, " << to_mb(num_input_bytes) << "MB, "
      << to_mb(input_total) << "MB | core "
      << num_core_msgs << "#, " << to_mb(num_core_bytes) << "MB, "
      << to_mb(core_total) << "MB");

  ///////
  //for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //  auto const& pl = placements[gid];

  //  vector<int> cnts(nlocs, 0);
  //  for(int const& loc: pl.locations.get()) {
  //    cnts[loc]++;
  //  }

  //  DOUT(gid << ": " << cnts << "  " << pl.locations.get());
  //}
  //DOUT("");
  //DOUT("");
}

vector<placement_t> autoplace(
  graph_t const& graph,
  int world_size,
  int num_threads_per,
  int max_branching,
  parts_space_t space,
  bool double_workers)
{
  int multiplier = double_workers ? 2 : 1 ;
  gremlin_t* gremlin_parts = new gremlin_t("parts");
  auto parts = autopartition_for_bytes(
    graph,
    multiplier * world_size * num_threads_per,
    max_branching,
    space);
  delete gremlin_parts;

  DOUT(" partition cost " << double(autopartition_for_bytes_cost(graph, parts)) / 1e9);

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f, parts);
    DOUT("printed g.gv");
  }

  //{
  //  uint64_t flops_per_byte_moved = 100;
  //  auto ret = autolocate_bipartite(
  //    graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("bipartite 100", graph, ret);
  //  return ret;
  //}

  //{
  //  uint64_t flops_per_byte_moved = 100;
  //  auto ret = autolocate_agg_at_a_time_from_inns_v2(
  //    graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("v2 100", graph, ret);
  //}

  {
    uint64_t flops_per_byte_moved = 100;
    auto ret = autolocate_agg_at_a_time_from_inns(
      graph, parts, world_size, flops_per_byte_moved);
    _print_pl_info("agg-at-a-time-from-inns 100", graph, ret);
    return ret;
  }
}

graph_t build_matmul(uint64_t ni, uint64_t nj, uint64_t nk) {
  graph_writer_t writer;
  tensor_t lhs = writer.input({ni,nj});
  tensor_t rhs = writer.input({nj,nk});
  tensor_t out = writer.matmul(lhs, rhs).save();
  return writer.get_graph();
}

graph_t make_graph_ff(
  uint64_t batch,
  uint64_t hidden,
  uint64_t dim)
{
  graph_writer_t writer;

  tensor_t x = writer.input({batch, dim});

  tensor_t w1 = writer.input({hidden, dim});
  tensor_t w2 = writer.input({dim, hidden});
  tensor_t w3 = writer.input({hidden, dim});

  tensor_t w1t = w1.transpose(0, 1);
  tensor_t w2t = w2.transpose(0, 1);
  tensor_t w3t = w3.transpose(0, 1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  tensor_t a = writer.matmul(x, w1t);
  //tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

  tensor_t b = writer.matmul(x, w3t) ;

  tensor_t c = writer.mul(a, b);

  tensor_t out = writer.matmul(c, w2t);
  out.save_inplace();

  return writer.get_graph();
}

graph_t make_graph_attention(model_args_t const& args)
{
  graph_writer_t writer;

  set_default_dtype(dtype_t::f32);

  full_shape_t embedding_shape({
    full_dim_t::singleton(args.batch_size),
    full_dim_t::singleton(args.max_seq_len),
    args.full_dim()
  });

  full_shape_t kqv_shape({ args.full_dim(), args.full_dim() });

  tensor_t x = writer.input(embedding_shape);

  tensor_t wq = writer.input(kqv_shape);
  tensor_t wk = writer.input(kqv_shape);
  tensor_t wv = writer.input(kqv_shape);
  tensor_t wo = writer.input(kqv_shape);

  tensor_t xq = writer.matmul(x, wq.transpose(0,1));
  tensor_t xk = writer.matmul(x, wk.transpose(0,1));
  tensor_t xv = writer.matmul(x, wv.transpose(0,1));

  vector<uint64_t> full_xshape = {
    args.batch_size, args.max_seq_len, args.n_heads, args.head_dim()
  };

  xq = xq.view_full(full_xshape);
  xk = xk.view_full(full_xshape);
  xv = xv.view_full(full_xshape);

  tensor_t keys = xk;
  tensor_t values = xv;

  xq = xq.transpose(1, 2);
  keys = keys.transpose(1, 2);
  values = values.transpose(1, 2);

  scalarop_t scale = scalarop_t::make_scale(
    scalar_t(default_dtype(), write_with_ss(
      1.0 / (std::sqrt(double(1.0) * args.head_dim()))
    ))
  );

  tensor_t scores;
  scores = writer.matmul(xq, keys.transpose(2, 3));
  scores = writer.ew(scale, scores);

  scores = writer.softmax(scores);

  tensor_t output;
  output = writer.matmul(scores, values);
  output = output.transpose(1, 2);

  output = output.view(embedding_shape);

  output = writer.matmul(output, wo.transpose(0,1)).save();

  return writer.get_graph();
}

graph_t make_graph_attention_feedforward(model_args_t const& args)
{
  graph_writer_t writer;

  set_default_dtype(dtype_t::f32);

  full_shape_t embedding_shape({
    full_dim_t::singleton(args.batch_size),
    full_dim_t::singleton(args.max_seq_len),
    args.full_dim()
  });

  full_shape_t kqv_shape({ args.full_dim(), args.full_dim() });

  uint64_t hidden_dim = 4 * args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    args.multiple_of * ( (hidden_dim + args.multiple_of - 1) / args.multiple_of );

  full_shape_t to_hidden(
    { full_dim_t::singleton(hidden_dim), args.full_dim() }
  );
  full_shape_t to_dim(
    { args.full_dim(), full_dim_t::singleton(hidden_dim) }
  );

  tensor_t x = writer.input(embedding_shape);
  {
    tensor_t h = x;

    tensor_t wq = writer.input(kqv_shape);
    tensor_t wk = writer.input(kqv_shape);
    tensor_t wv = writer.input(kqv_shape);
    tensor_t wo = writer.input(kqv_shape);

    tensor_t xq = writer.matmul(x, wq.transpose(0,1));
    tensor_t xk = writer.matmul(x, wk.transpose(0,1));
    tensor_t xv = writer.matmul(x, wv.transpose(0,1));

    vector<uint64_t> full_xshape = {
      args.batch_size, args.max_seq_len, args.n_heads, args.head_dim()
    };

    xq = xq.view_full(full_xshape);
    xk = xk.view_full(full_xshape);
    xv = xv.view_full(full_xshape);

    tensor_t keys = xk;
    tensor_t values = xv;

    xq = xq.transpose(1, 2);
    keys = keys.transpose(1, 2);
    values = values.transpose(1, 2);

    scalarop_t scale = scalarop_t::make_scale(
      scalar_t(default_dtype(), write_with_ss(
        1.0 / (std::sqrt(double(1.0) * args.head_dim()))
      ))
    );

    tensor_t scores;
    scores = writer.matmul(xq, keys.transpose(2, 3));
    scores = writer.ew(scale, scores);

    scores = writer.softmax(scores);

    tensor_t output;
    output = writer.matmul(scores, values);
    output = output.transpose(1, 2);

    output = output.view(embedding_shape);

    output = writer.matmul(output, wo.transpose(0,1)).save();

    x = writer.add(output,h);
  }

  {
    tensor_t h = x;

    tensor_t w1 = writer.input(to_hidden);
    tensor_t w2 = writer.input(to_dim);
    tensor_t w3 = writer.input(to_hidden);

    tensor_t w1t = w1.transpose(0,1);
    tensor_t w2t = w2.transpose(0,1);
    tensor_t w3t = w3.transpose(0,1);
  
    scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

    tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

    tensor_t b = writer.matmul(x, w3t) ;

    tensor_t c = writer.mul(a, b);

    tensor_t output = writer.matmul(c, w2t);

    x = writer.add(output,h);
  }

  x = x.save();

  return writer.get_graph();
}

uint64_t compute_hidden(
  uint64_t dim,
  uint64_t multiple_of)
{
  uint64_t ret = 4 * dim;
  ret = uint64_t( (2.0 * ret) / 3.0 );
  ret = multiple_of * ( (ret + multiple_of - 1) / multiple_of );
  return ret;
}

partition_t make_split_partition(
  vector<uint64_t> const& shape,
  vector<int> const& splits)
{
  if(shape.size() != splits.size()) {
    throw std::runtime_error("invalid inputs: make split");
  }

  vector<partdim_t> pds;
  for(int i = 0; i != shape.size(); ++i) {
    pds.push_back(partdim_t::split(shape[i], splits[i]));
  }
  return partition_t(pds);
}

tuple<graph_t, vector<placement_t>>
make_einsummable_graph(
  string str,
  vector<int> const& splits,
  vector<uint64_t> const& dims)
{
  auto [inns, out_rank] = einsummable_t::parse_str(str);
  einsummable_t e(dims, inns, out_rank, scalarop_t::make_mul(), castable_t::add);
  graph_constructor_t g;
  vector<int> inn_gids;
  for(int i = 0; i != inns.size(); ++i) {
    vector<uint64_t> d = e.get_input_from_join(dims,   i);
    vector<int>      s = e.get_input_from_join(splits, i);
    partition_t p = make_split_partition(d, s);
    inn_gids.push_back(g.insert_input(p));
  }

  int join = g.insert_einsummable(
    make_split_partition(dims, splits),
    e,
    inn_gids);

  partition_t out_part = make_split_partition(
    vector<uint64_t>(dims.begin(), dims.begin() + out_rank),
    vector<int>(splits.begin(), splits.begin() + out_rank));

  int out = g.insert_formation(out_part, join);

  return {g.graph, g.get_placements()};
}

model_args_t make_model_args(args_t& args) {
  args.set_default("batch", uint64_t(1));
  args.set_default("model", "7B");
  string model = args.get<string>("model");
  DLINEOUT("model: " << model);
  uint64_t batch = args.get<uint64_t>("batch");
  model_args_t margs;
  if(model == "7B") {
    margs = model_args_t::llama_7B(batch);
  } else if(model == "13B") {
    margs = model_args_t::llama_13B(batch);
  } else if(model == "30B") {
    margs = model_args_t::llama_30B(batch);
  } else if(model == "65B") {
    margs = model_args_t::llama_65B(batch);
  } else {
    throw std::runtime_error("arg: model incorrect");
  }
  return margs;
}

tuple<graph_t, optional<vector<placement_t>>> 
build_graph(args_t& args)
{
  args.set_default("graph", "matmul");

  DLINEOUT("graph: " << args.get<string>("graph"));
  if(args.get<string>("graph") == "matmul") {
    args.set_default("ni", uint64_t(10000));
    args.set_default("nj", uint64_t(10000));
    args.set_default("nk", uint64_t(10000));
    return {
      build_matmul(
        args.get<uint64_t>("ni"),
        args.get<uint64_t>("nj"),
        args.get<uint64_t>("nk")),
      std::nullopt
    };
  } else if(args.get<string>("graph") == "ff") {
    args.set_default<uint64_t>("batch",       1   );
    args.set_default<uint64_t>("seqlen",      2048);
    args.set_default<uint64_t>("dim",         4096);
    args.set_default<uint64_t>("multiple_of", 256 );

    uint64_t batch       = args.get<uint64_t>("batch");
    uint64_t seqlen      = args.get<uint64_t>("seqlen");
    uint64_t dim         = args.get<uint64_t>("dim");
    uint64_t multiple_of = args.get<uint64_t>("multiple_of");

    uint64_t hidden = compute_hidden(dim, multiple_of);
    return {make_graph_ff(batch * seqlen, hidden, dim), std::nullopt};
  } else if(args.get<string>("graph") == "attention") {
    model_args_t margs = make_model_args(args);
    return {make_graph_attention(margs), std::nullopt};
  } else if(args.get<string>("graph") == "aff") {
    model_args_t margs = make_model_args(args);
    return {make_graph_attention_feedforward(margs), std::nullopt};
  } else if(args.get<string>("graph") == "bmm1") {
    auto [graph, ps] = make_einsummable_graph(
      "aebf,cdef->abcd",
      vector<int>{1,1,8,1,4,1},
      vector<uint64_t>{1,512,64,128,64,128});
    return {graph, optional<vector<placement_t>>{ps}};
  } else if(args.get<string>("graph") == "contraction1") {
    auto [graph, ps] = make_einsummable_graph(
      "abef,cdef->abcd",
      vector<int>{1,1,8,1,1,4},
      vector<uint64_t>{1,512,64,128,64,128});
    return {graph, optional<vector<placement_t>>{ps}};
  } else if(args.get<string>("graph") == "bmm2") {
    auto [graph, ps] = make_einsummable_graph(
      "aebf,cdef->abcd",
      vector<int>{1,1,8,1,1,4},
      vector<uint64_t>{1,512,32,128,32,128});
    return {graph, optional<vector<placement_t>>{ps}};
  } else if(args.get<string>("graph") == "contraction2") {
    auto [graph, ps] = make_einsummable_graph(
      "abef,cdef->abcd",
      vector<int>{1,1,8,1,4,1},
      vector<uint64_t>{1,512,32,128,32,128});
    return {graph, optional<vector<placement_t>>{ps}};
  } else {
    throw std::runtime_error("invalid graph: " + args.get<string>("graph"));
  }
}

///////////////////////////////////////////////////////

#include <mkl_cblas.h>
#include <mkl.h>

#include "../src/engine/cpu/kernel_executor.h"

template <typename T> 
vector<T*> _fill_out_(
  uint64_t const& nb,
  bool const& batched,
  T* ptr,
  uint64_t sz)
{
  if(batched) {
    vector<T*> ret;
    ret.reserve(nb);
    for(uint64_t i = 0; i != nb; ++i) {
      ret.push_back(ptr + i*sz);
    }
    return ret;
  } else {
    return vector<T*>(nb, ptr);
  }
}

void bmm(
  uint64_t const& nb,
  bool const& batched_out,
  bool const& batched_lhs,
  bool const& batched_rhs,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* _out,
  void const* _lhs,
  void const* _rhs)
{
  if(nb == 1) {
    throw std::runtime_error("bwoah");
  }

  DLINEOUT("nb" << nb << " bout_lhs_rhs" << batched_out << batched_lhs << batched_rhs << " ni,nj,nk|" << ni << "," << nj << "," << nk << " tl,tr" << trans_lhs << trans_rhs);

  MKL_INT group_count;
  vector<MKL_INT> group_sizes;
  vector<float> beta_array;
  if(batched_out) {
    group_count = 1;
    group_sizes.push_back(nb);
    beta_array.push_back(0.0);
  } else {
    group_count = 2;
    group_sizes.push_back(1);
    group_sizes.push_back(nb-1);
    beta_array.push_back(0.0);
    beta_array.push_back(1.0);
  }
  DLINEOUT(group_count << " " << group_sizes << " | " << beta_array);

  vector<CBLAS_TRANSPOSE> trans_lhs_(group_count, trans_lhs ? CblasTrans : CblasNoTrans);
  vector<CBLAS_TRANSPOSE> trans_rhs_(group_count, trans_rhs ? CblasTrans : CblasNoTrans);
  DLINEOUT(trans_lhs_.size() << " " << trans_rhs_.size());

  vector<MKL_INT> nis(group_count, ni);
  vector<MKL_INT> nks(group_count, nk);
  vector<MKL_INT> njs(group_count, nj);

  DLINEOUT(nis << " " << nks << " " << njs);

  vector<float> alpha_array(group_count, 1.0);
  DLINEOUT(alpha_array << " " << beta_array);

  float const* lhs_ = reinterpret_cast<float const*>(_lhs);
  float const* rhs_ = reinterpret_cast<float const*>(_rhs);
  float      * out_ = reinterpret_cast<float      *>(_out);

  vector<float const*> lhs = _fill_out_(nb, batched_lhs, lhs_, ni*nj);
  vector<float const*> rhs = _fill_out_(nb, batched_rhs, rhs_, nj*nk);
  vector<float      *> out = _fill_out_(nb, batched_out, out_, ni*nk);
  DLINEOUT(lhs.size() << " " << rhs.size() << " " << out.size());

  cblas_sgemm_batch(
    CblasRowMajor,
    trans_lhs_.data(), 
    trans_rhs_.data(),
    nis.data(), nks.data(), njs.data(),       // 4,5,6
    alpha_array.data(),                       // 7
    lhs.data(),              
    trans_lhs ? nis.data() : njs.data(),      // 9
    rhs.data(),
    trans_rhs ? njs.data() : nks.data(),
    beta_array.data(),
    out.data(),
    nks.data(),
    group_count,
    group_sizes.data());

  //void cblas_sgemm_batch(
  //  const CBLAS_LAYOUT Layout,            // 1
  //  const CBLAS_TRANSPOSE* transa_array, 
  //  const CBLAS_TRANSPOSE* transb_array, 
  //  const MKL_INT* m_array,               // 4
  //  const MKL_INT* n_array, 
  //  const MKL_INT* k_array, 
  //  const float* alpha_array,             // 7
  //  const float **a_array, 
  //  const MKL_INT* lda_array,             // 9
  //  const float **b_array,                // 10
  //  const MKL_INT* ldb_array, 
  //  const float* beta_array, 
  //  float **c_array, 
  //  const MKL_INT* ldc_array,             // 14
  //  const MKL_INT group_count, 
  //  const MKL_INT* group_size);           // 16
}  

void main2(int argc, char** argv) {
  args_t args(argc, argv);
  args.set_default("nb", uint64_t(2));
  args.set_default("ni", uint64_t(10));
  args.set_default("nj", uint64_t(10));
  args.set_default("nk", uint64_t(10));
  args.set_default("bout", true);
  args.set_default("blhs", true);
  args.set_default("brhs", true);

  uint64_t nb = args.get<uint64_t>("nb"); 
  uint64_t ni = args.get<uint64_t>("ni"); 
  uint64_t nj = args.get<uint64_t>("nj"); 
  uint64_t nk = args.get<uint64_t>("nk"); 

  bool bout = args.get<bool>("bout");
  bool blhs = args.get<bool>("blhs");
  bool brhs = args.get<bool>("brhs");

  dbuffer_t L  = make_dbuffer(dtype_t::f32, (blhs ? nb : 1)*ni*nj);
  dbuffer_t R  = make_dbuffer(dtype_t::f32, (brhs ? nb : 1)*nj*nk);
  dbuffer_t O  = make_dbuffer(dtype_t::f32, (bout ? nb : 1)*ni*nk);
  dbuffer_t O1 = make_dbuffer(dtype_t::f32, (bout ? nb : 1)*ni*nk);

  L.ones();
  R.ones();

  for(int i = 0; i != 1; ++i) {
    bmm(nb, bout,blhs,brhs, ni,nj,nk, false, false, O1.raw(), L.raw(), R.raw()); 
    DLINEOUT("SUCCESS ON BMM");
  }

  batch_matrix_multiply(
    dtype_t::f32, 
    nb, bout,blhs,brhs, ni,nj,nk, false, false, O.raw(), L.raw(), R.raw()); 

  DOUT(nb << " | " << ni << "," << nj << "," << nk);
  DOUT(O.min() << "          " << O.max());
}

void main3() {
  uint64_t nb = 16; 
  uint64_t ni = 512; 
  uint64_t nj = 128; 
  uint64_t nk = 512; 
  
  cpu_kernel_executor_t executor;  
  auto [inns, out_rank] = einsummable_t::parse_str("bij,bjk->bik");
  vector<uint64_t> join_shape = einsummable_t::construct_join_shape(
    inns,
    { {nb,ni,nj}, {nb,nj,nk} }).value();
  einsummable_t e(
    join_shape, inns, out_rank, 
    scalarop_t::make_mul(dtype_t::f32), castable_t::add);

  uint64_t worksize = executor.build(e).value();
  DLINEOUT("worksize is " << worksize);
  vector<uint8_t> workspace(worksize);

  dbuffer_t L = make_dbuffer(dtype_t::f32, nb*ni*nj);
  dbuffer_t R = make_dbuffer(dtype_t::f32, nb*nj*nk);
  dbuffer_t O = make_dbuffer(dtype_t::f32, nb*ni*nk);

  L.ones();
  R.ones();

  executor(e, O.raw(), {L.raw(), R.raw()}, 
    tuple<void*, uint64_t>{reinterpret_cast<void*>(workspace.data()), worksize});
}

int main(int argc, char** argv) {
  main2(argc, argv);
  main3();
}
