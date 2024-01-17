#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include <fstream>

void usage() {
  std::cout << "Usage: nlocs ni nj nk li lj rj rk ji jj jk oi ok\n";
}

int main(int argc, char** argv) {
  bool with_taskgraph = true;
  bool with_memgraph  = true;

  if(argc != 14) {
    usage();
    return 1;
  }

  int nlocs;
  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  try {
    nlocs = parse_with_ss<int> (argv[1]);

    ni = parse_with_ss<uint64_t>(argv[2]);
    nj = parse_with_ss<uint64_t>(argv[3]);
    nk = parse_with_ss<uint64_t>(argv[4]);

    li = parse_with_ss<int>(argv[5]);
    lj = parse_with_ss<int>(argv[6]);

    rj = parse_with_ss<int>(argv[7]);
    rk = parse_with_ss<int>(argv[8]);

    ji = parse_with_ss<int>(argv[9]);
    jj = parse_with_ss<int>(argv[10]);
    jk = parse_with_ss<int>(argv[11]);

    oi = parse_with_ss<int>(argv[12]);
    ok = parse_with_ss<int>(argv[13]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
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

  int rhs2 = g.insert_input({7, 20});

  int join2 = g.insert_einsummable(
    einsummable_t::from_matmul(10, 7, 20),
    {out, rhs2}
  );

  for (int i = 0; i < g.graph.nodes.size(); i++) {
    auto const& node = g.graph.nodes[i];
    if (node.op.is_einsummable()) {
      auto const& einsum = node.op.get_einsummable();
      std::cout << "=====================================================" << std::endl;
      std::cout << "These are input nodes" << std::endl << "\t";
      for (int j = 0; j < node.inns.size(); j++) {
        std::cout << node.inns[j] << " ";
      }
      std::cout << std::endl;
      std::cout << "This is join shape: " << std::endl << "\t";
      for (auto const& shape : einsum.join_shape) {
        std::cout << shape << " ";
      }
      std::cout << std::endl;
      std::cout << "=====================================================" << std::endl;
    }
    if (node.op.is_formation()) {
      auto const& formation = node.op.get_formation();
      std::cout << "=====================================================" << std::endl;
      std::cout << "This is a formation node" << std::endl;
      std::cout << "This is the shape: " << std::endl << "\t";
      for (auto const& shape : formation.shape) {
        std::cout << shape << " ";
      }
      std::cout << std::endl;
      std::cout << "=====================================================" << std::endl;
    }
  }

  graph_t const& graph = g.graph;

  taskgraph_t taskgraph;

  auto pls = g.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  // just set every block in every location to something
  // random
  set_seed(0);
  for(auto& placement: pls) {
    for(auto& loc: placement.locations.get()) {
      loc = runif(nlocs);
    }
  }

  auto [_0, _1, _taskgraph] = taskgraph_t::make(graph, pls);
  taskgraph = _taskgraph;

  auto [_2, _3, memgraph] = memgraph_t::make(taskgraph);
  // ^ note that memsizes and allocat_settings not being provided

  {
    std::ofstream f("gprint.gv");
    graph.print_graphviz(f);
    DOUT("wrote g.gv");
  }

  {
    std::ofstream f("tgprint.gv");
    taskgraph.print_graphviz(f);
    DOUT("wrote tg.gv");
  }

  {
    std::ofstream f("mgprint.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }
}

