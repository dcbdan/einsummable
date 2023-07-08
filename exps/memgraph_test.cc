#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"

#include <fstream>

void usage() {
  DOUT("pi pj pk di dj dk np");
}

void main_mm(int argc, char** argv) {
  if(argc != 8) {
    usage();
    return;
  }

  int pi, pj, pk;
  uint64_t di, dj, dk;
  int np;
  try {
    pi = parse_with_ss<int>(     argv[1]);
    pj = parse_with_ss<int>(     argv[2]);
    pk = parse_with_ss<int>(     argv[3]);
    di = parse_with_ss<uint64_t>(argv[4]);
    dj = parse_with_ss<uint64_t>(argv[5]);
    dk = parse_with_ss<uint64_t>(argv[6]);
    np = parse_with_ss<int>(     argv[7]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return;
  }

  auto g = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
  {
    std::cout << "tg.gv" << std::endl;
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
  }

  // it could be the case that not all locs are actually used,
  // for example 1 1 2 100 100 100 88
  // Here only 2 locs will really be used, not all 88...
  np = taskgraph.num_locs();

  tuple<
    map<int, mem_t>, // input -> mem
    map<int, mem_t>, // save -> mem
    memgraph_t>
    _info1 = memgraph_t::make_without_evict(
      taskgraph,
      {},
      { allocator_strat_t::lowest_dependency, 1 } );
  auto const& [_2, _3, m1] = _info1;

  {
    std::cout << "m1.gv" << std::endl;
    std::ofstream f("m1.gv");
    m1.print_graphviz(f);
  }

  memgraph_t m2 = memgraph_t::from_wire(m1.to_wire());

  {
    std::cout << "m2.gv" << std::endl;
    std::ofstream f("m2.gv");
    m2.print_graphviz(f);
  }
}

void main01() {
  graph_constructor_t g;
  int inn = g.insert_input({20});
  int aaa = g.insert_formation(
    partition_t({ partdim_t::repeat(2, 10) }),
    inn);
  int bbb = g.insert_formation(
    partition_t({ partdim_t::repeat(1, 20) }),
    aaa);
  int ccc = g.insert_formation(
    partition_t({ partdim_t::repeat(2, 10) }),
    bbb);


  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  auto np = taskgraph.num_locs();

  auto [_2, _3, memgraph] = memgraph_t::make_without_evict(
    taskgraph, {},
    { allocator_strat_t::first, 1 }
  );

  std::cout << "Printing to reblock_mg.gv" << std::endl;
  std::ofstream f("reblock_mg.gv");
  memgraph.print_graphviz(f);
}

int main(int argc, char** argv) {
  main_mm(argc, argv);
  main01();
}
