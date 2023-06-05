#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute.h"

#include <fstream>
#include <memory>

void usage() {
  DOUT("pi pj pk di dj dk np");
}

int main(int argc, char** argv) {
  if(argc != 8) {
    usage();
    return 1;
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
    return 1;
  }

  auto g = three_dimensional_matrix_multiplication(
    pi,pj,pk, di,dj,dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
  // it could be the case that not all locs are actually used,
  // for example 1 1 2 100 100 100 88
  // Here only 2 locs will really be used, not all 88...
  np = taskgraph.num_locs();

  // have everyone share the same cache
  vector<int> compute_loc_to_cache(np, 0);

  {
    tuple<
      map<int, mem_t>, // input -> mem
      map<int, mem_t>, // save -> mem
      memgraph_t>
      _info1 = memgraph_t::make_without_evict(
        taskgraph,
        compute_loc_to_cache,
        {},
        allocator_strat_t::lowest_dependency);
    auto const& [_2, _3, memgraph] = _info1;

    // std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
    // std::ofstream f("mm3d_mem_lowest_dep.gv");
    // memgraph.print_graphviz(f);
    uint64_t buffer_size = 0;
    execute(memgraph);
  }

  

//   {
//     tuple<
//       map<int, mem_t>, // input -> mem
//       map<int, mem_t>, // save -> mem
//       memgraph_t>
//       _info1 = memgraph_t::make_without_evict(
//         taskgraph,
//         compute_loc_to_cache,
//         {},
//         allocator_strat_t::first);
//     auto const& [_2, _3, memgraph] = _info1;

//     std::cout << "Printing to mm3d_mem_first.gv" << std::endl;
//     std::ofstream f("mm3d_mem_first.gv");
//     memgraph.print_graphviz(f);
//   }
}
