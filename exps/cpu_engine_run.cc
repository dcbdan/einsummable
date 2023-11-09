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
  parts_space_t space = parts_space_t::contraction,
  bool double_workers = false);

void main_(int argc, char** argv) {
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
  int num_channels_per_move = 4;

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

    // execute this
    graph_t graph = build_graph(args);
    vector<placement_t> pls = autoplace(
      graph, world_size, num_threads, space, double_workers);

    args.set_default<int>("nrep", 1);
    int nrep = args.get<int>("nrep");
    for(int rep = 0; rep != nrep; ++rep) {
      // initialize input tensors and distribute across the cluster
      for(int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if(node.op.is_input()) {
          auto const& input = node.op.get_input();
          dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
          tensor.random("-0.01", "0.01");
          //tensor.ones();
          server.insert_tensor(gid, pls[gid], tensor);
        }
      }

      // execute
      server.execute_graph(graph, pls);
    }

    server.shutdown();
  } else {
    server.listen();
  }
}

int main(int argc, char** argv) {
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

  DLINEOUT("world size      : " << world_size);
  DLINEOUT("num threads per : " << num_threads);
  graph_t graph = build_graph(args);
  vector<placement_t> pls = autoplace(graph, world_size, num_threads, space);
}

void _print_pl_info(
  string msg,
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto [_0, _1, taskgraph] =
    taskgraph_t::make(graph, placements);

  if(msg.size() < 45) {
    msg.resize(45, ' ');
  }

  int num_msgs = 0;
  uint64_t num_bytes = 0;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_move()) {
      num_msgs++;
      num_bytes += node.op.get_move().size;
    }
  }
  vector<uint64_t> tensor_move_costs = compute_tensor_move_costs(graph, placements);
  uint64_t tensor_move_bytes_total = vector_sum(tensor_move_costs);
  DOUT("(" << msg << ") taskgraph with " << num_msgs << " moves, "
    << double(num_bytes)/1e6 << " MB bytes moved");
    //<< double(tensor_move_bytes_total)/1e6 << " MB from tensor move");
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& pl = placements[gid];
    uint64_t const& tensor_move_cost = tensor_move_costs[gid];
    //DOUT(gid << ": " << double(tensor_move_cost)/1e6
    //  );
    //  << " " << pl.locations.get());
  }
}

vector<placement_t> autoplace(
  graph_t const& graph,
  int world_size,
  int num_threads_per,
  parts_space_t space,
  bool double_workers)
{
  int multiplier = double_workers ? 2 : 1 ;
  auto parts = autopartition_for_bytes(
    graph,
    multiplier * world_size * num_threads_per,
    space);

  DOUT(" partition cost " << double(autopartition_for_bytes_cost(graph, parts)) / 1e9);

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f, parts);
    DOUT("printed g.gv");
  }

  {
    uint64_t flops_per_byte_moved = 100;
    auto ret = autolocate_bipartite(
      graph, parts, world_size, flops_per_byte_moved);
    _print_pl_info("bipartite 100", graph, ret);
  }
  {
    auto ret = load_balanced_placement(graph, parts, world_size, false);
    _print_pl_info("from inputs", graph, ret);
  }

  {
    auto ret = load_balanced_placement_from_outs(graph, parts, world_size, false);
    _print_pl_info("from outputs", graph, ret);
  }

  {
    auto ret = autolocate(graph, parts, world_size);
    _print_pl_info("a-locate", graph, ret);
  }

  //DOUT("");

  {
    uint64_t flops_per_byte_moved = 100;
    auto ret = autolocate_agg_at_a_time_from_inns(
      graph, parts, world_size, flops_per_byte_moved);
    _print_pl_info("agg-at-a-time-from-inns 100", graph, ret);
  }

  //{
  //  uint64_t flops_per_byte_moved = 1000;
  //  auto ret = autolocate_agg_at_a_time_from_inns(
  //    graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time-from-inns 1000", graph, ret);
  //}

  //{
  //  uint64_t flops_per_byte_moved = 10000;
  //  auto ret = autolocate_agg_at_a_time_from_inns(
  //    graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time-from-inns 10000", graph, ret);
  //}

  //{
  //  uint64_t flops_per_byte_moved = 100000;
  //  auto ret = autolocate_agg_at_a_time_from_inns(
  //    graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time-from-inns 100000", graph, ret);
  //}

  //DOUT("");

  //{
  //  uint64_t flops_per_byte_moved = 100;
  //  auto ret = autolocate_agg_at_a_time(graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time 100", graph, ret);
  //}

  //{
  //  uint64_t flops_per_byte_moved = 1000;
  //  auto ret = autolocate_agg_at_a_time(graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time 1000", graph, ret);
  //}

  //{
  //  uint64_t flops_per_byte_moved = 10000;
  //  auto ret = autolocate_agg_at_a_time(graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time 10000", graph, ret);
  //}

  //{
  //  uint64_t flops_per_byte_moved = 100000;
  //  auto ret = autolocate_agg_at_a_time(graph, parts, world_size, flops_per_byte_moved);
  //  _print_pl_info("agg-at-a-time 100000", graph, ret);
  //}

  auto ret = load_balanced_placement_from_outs(graph, parts, world_size, false);
  return ret;

  //auto ret1 = load_balanced_placement(graph, parts, world_size, false);
  //auto cost1 = vector_sum(compute_tensor_move_costs(graph, placements));

  //auto ret2 = load_balanced_placement_from_outs(graph, parts, world_size, false);
  //auto cost2 = vector_sum(compute_tensor_move_costs(graph, placements));

  //if(cost1 < cost2) {
  //  return ret1;
  //} else {
  //  return ret2;
  //}
}

using tensor_t     = graph_writer_t::tensor_t;
using full_dim_t   = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

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
  } else {
    throw std::runtime_error("invalid graph");
  }
}
