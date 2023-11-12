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

graph_t build_graph(args_t& args);

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
  int num_channels = 8;
  int num_channels_per_move = 2;

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

    // execute this
    graph_t graph = build_graph(args);
    vector<placement_t> pls = autoplace(
      graph, world_size, num_threads, max_branching, space, double_workers);

    args.set_default<int>("nrep", 1);
    int nrep = args.get<int>("nrep");
    for(int rep = 0; rep != nrep; ++rep) {
      if(rep == nrep-1) {
        get_cpu_kernel_timetracker().clear();
      }

      // initialize input tensors and distribute across the cluster
      for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if(node.op.is_input()) {
          auto const& input = node.op.get_input();
          dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
          tensor.random("-0.001", "0.001");
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

    server.shutdown();

    get_cpu_kernel_timetracker().print_totals(std::cout);
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

  graph_t graph = build_graph(args);
  vector<placement_t> pls = autoplace(graph, world_size, num_threads, max_branching, space);
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

uint64_t compute_hidden(
  uint64_t dim,
  uint64_t multiple_of)
{
  uint64_t ret = 4 * dim;
  ret = uint64_t( (2.0 * ret) / 3.0 );
  ret = multiple_of * ( (ret + multiple_of - 1) / multiple_of );
  return ret;
}

graph_t build_graph(args_t& args)
{
  args.set_default("graph", "matmul");

  if(args.get<string>("graph") == "matmul") {
    args.set_default("ni", uint64_t(10000));
    args.set_default("nj", uint64_t(10000));
    args.set_default("nk", uint64_t(10000));
    return build_matmul(
      args.get<uint64_t>("ni"),
      args.get<uint64_t>("nj"),
      args.get<uint64_t>("nk"));
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
    return make_graph_ff(batch * seqlen, hidden, dim) ;
  } else if(args.get<string>("graph") == "attention") {
    args.set_default("batch", uint64_t(1));
    args.set_default("model", "7B");
    string model = args.get<string>("model");
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
    return make_graph_attention(margs);
  } else {
    throw std::runtime_error("invalid graph");
  }
}

int main(int argc, char** argv) {
  main0(argc, argv);
}
