#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/engine/communicator.h"

#include "../src/server/cpu/server.h"

void usage() {
  std::cout << "Setup usage: addr_zero is_client world_size memsize\n";
  std::cout << "Extra usage for server: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

tuple<graph_t, vector<placement_t>> build_graph_pls(
  int world_size, int argc, char** argv);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f64);

  if(argc < 4) {
    usage();
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);

  cpu_mg_server_t server(communicator, mem_size);

  if(is_rank_zero) {
    // execute this
    auto [graph, pls] = build_graph_pls(world_size, argc - 4, argv + 4);

    // initialize input tensors and distribute across the cluster
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      auto const& node = graph.nodes[gid];
      if(node.op.is_input()) {
        auto const& input = node.op.get_input();
        dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
        //tensor.random("-0.01", "0.01");
        tensor.ones();
        server.insert_tensor(gid, pls[gid], tensor);
      }
    }

    // execute
    server.execute_graph(graph, pls);

    // get the outputs to here
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      auto const& node = graph.nodes[gid];
      if(node.op.is_save()) {
        dbuffer_t tensor = server.get_tensor_from_gid(gid);
        DOUT("gid sum is: " << tensor.sum());
      }
    }

    server.shutdown();
  } else {
    server.listen();
  }
}

tuple<graph_t, vector<placement_t>> build_graph_pls(
  int world_size, int argc, char** argv)
{
  if(argc != 13) {
    usage();
    throw std::runtime_error("incorrect number of args");
  }

  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  try {
    ni = parse_with_ss<uint64_t>(argv[1]);
    nj = parse_with_ss<uint64_t>(argv[2]);
    nk = parse_with_ss<uint64_t>(argv[3]);

    li = parse_with_ss<int>(argv[4]);
    lj = parse_with_ss<int>(argv[5]);

    rj = parse_with_ss<int>(argv[6]);
    rk = parse_with_ss<int>(argv[7]);

    ji = parse_with_ss<int>(argv[8]);
    jj = parse_with_ss<int>(argv[9]);
    jk = parse_with_ss<int>(argv[10]);

    oi = parse_with_ss<int>(argv[11]);
    ok = parse_with_ss<int>(argv[12]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    throw std::runtime_error("incorrect number of args");
  }

  graph_constructor_t g;
  dtype_t dtype = default_dtype();

  int lhs = g.insert_input(partition_t({
    partdim_t::split(ni, li),
    partdim_t::split(nj, lj) }));
  int rhs = g.insert_input(partition_t({
    partdim_t::split(nj, rj),
    partdim_t::split(nk, rk) }));

  int join = g.insert_einsummable(
    partition_t({
      partdim_t::split(ni, ji),
      partdim_t::split(nk, jk),
      partdim_t::split(nj, jj)
    }),
    einsummable_t::from_matmul(ni, nj, nk),
    {lhs, rhs});

  int out = g.insert_formation(
    partition_t({
      partdim_t::split(ni, oi),
      partdim_t::split(nk, ok)
    }),
    join);

  auto pls = g.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  // randomly assign the locations
  if(world_size > 1) {
    for(auto& pl: pls) {
      for(auto& loc: pl.locations.get()) {
        loc = runif(world_size);
      }
    }
  }

  return {g.graph, pls};
}

