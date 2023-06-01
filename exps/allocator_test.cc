#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"

#include <fstream>

void main01() {
std::cout << "trying to test new allocator_t" << std::endl;

  /*Take an example that we have a device with 100 bytes memory*/
  allocator_t allocator = allocator_t(100);

  allocator.print();
  DOUT("");

  auto [o0, _0] = allocator.allocate(6);
  auto [o1, _1] = allocator.allocate(4);
  auto [o2, _2] = allocator.allocate(2);
  auto [o3, _3] = allocator.allocate(7);
  DOUT("_0" << _0);
  DOUT("_1" << _1);
  DOUT("_2" << _2);
  DOUT("_3" << _3);
  allocator.free(o0,0);
  allocator.free(o1,0);
  allocator.free(o2,0);
  allocator.free(o3,0);
  allocator.print();
  auto [o5, _4] = allocator.allocate(10);
  DOUT("_4 " << _4);
  allocator.print();
}

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

    std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
    std::ofstream f("mm3d_mem_lowest_dep.gv");
    memgraph.print_graphviz(f);
  }

  {
    tuple<
      map<int, mem_t>, // input -> mem
      map<int, mem_t>, // save -> mem
      memgraph_t>
      _info1 = memgraph_t::make_without_evict(
        taskgraph,
        compute_loc_to_cache,
        {},
        allocator_strat_t::first);
    auto const& [_2, _3, memgraph] = _info1;

    std::cout << "Printing to mm3d_mem_first.gv" << std::endl;
    std::ofstream f("mm3d_mem_first.gv");
    memgraph.print_graphviz(f);
  }
}
