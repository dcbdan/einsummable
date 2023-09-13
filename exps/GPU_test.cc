#include "../src/einsummable/memgraph.h"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute_multi_gpu.h"

#include "GPU_correctness.cc"

#include <cstdint>
#include <fstream>
#include <memory>

void mem_check(memgraph_t const &m) {
  for (int idx = 0; idx != m.nodes.size(); ++idx) {
    auto const &node = m.nodes[idx];
    for (auto input_idx : node.inns) {
      // Our Node x has Node y as its input
      if (m.nodes[input_idx].outs.find(idx) == m.nodes[input_idx].outs.end()) {
        // But Node y doesn't have Node x as its output
        std::printf("Error: Node %d has node % d as its input but node %d "
                    "doesn't have node %d as its output\n",
                    idx, input_idx, input_idx, idx);
        exit(1);
      }
    }
    for (auto output_idx : node.outs) {
      // Our Node x has Node y as its output
      if (m.nodes[output_idx].inns.find(idx) ==
          m.nodes[output_idx].inns.end()) {
        // But Node y doesn't have Node x as its input
        std::printf("Error: Node %d has node % d as its output but node %d "
                    "doesn't have node %d as its input\n",
                    idx, output_idx, output_idx, idx);
        exit(1);
      }
    }
  }
}

struct random_placement_t {
  placement_t operator()(vector<uint64_t> const &total_shape) {
    vector<partdim_t> partdims;
    for (uint64_t const &n : total_shape) {
      auto const &[beg_, end_] = part_size_rng;
      int p;
      if (end_ > n) {
        p = 1;
      } else {
        p = runif(beg_, end_);
      }
      partdims.push_back(partdim_t::split(n, p));
    }
    partition_t part(partdims);

    return placement_t::random(part, nloc);
  }

  tuple<int, int> part_size_rng;
  int nloc;
};

void usage() { DOUT("pi pj pk di dj dk np"); }

// testing 3d matmul on a single GPU
//int main_matmul(int argc, char **argv) {
//  if (argc != 8) {
//    usage();
//    return 1;
//  }
//
//  int pi, pj, pk;
//  uint64_t di, dj, dk;
//  int np;
//  try {
//    pi = parse_with_ss<int>(argv[1]);
//    pj = parse_with_ss<int>(argv[2]);
//    pk = parse_with_ss<int>(argv[3]);
//    di = parse_with_ss<uint64_t>(argv[4]);
//    dj = parse_with_ss<uint64_t>(argv[5]);
//    dk = parse_with_ss<uint64_t>(argv[6]);
//    np = parse_with_ss<int>(argv[7]);
//  } catch (...) {
//    std::cout << "Parse error." << std::endl << std::endl;
//    usage();
//    return 1;
//  }
//
//  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);
//
//  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
//  // it could be the case that not all locs are actually used,
//  // for example 1 1 2 100 100 100 88
//  // Here only 2 locs will really be used, not all 88...
//  np = taskgraph.num_locs();
//
//  // have everyone share the same cache
//  vector<int> compute_loc_to_cache(np, 0);
//
//  size_t allocator_size = 4lu * 1024lu * 1024lu * 1024lu;
//
//  {
//    tuple<map<int, mem_t>, // input -> mem
//          map<int, mem_t>, // save -> mem
//          memgraph_t>
//        _info1 = memgraph_t::make_without_evict(
//            taskgraph, {allocator_size},
//            {allocator_strat_t::lowest_dependency, 4});
//    auto const &[_2, _3, memgraph] = _info1;
//
//    std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
//    std::ofstream f("mm3d_mem_lowest_dep.gv");
//    memgraph.print_graphviz(f);
//    // execute(memgraph, gpu_allocate_memory(memgraph.mem_sizes()[0]));
//    // check_correctness(memgraph, false);
//
//    // check if the sizes of the memgraph is lower than what we have given
//    if (memgraph.mem_sizes()[0] > allocator_size) {
//      std::cout << "Error: the size of the memgraph is larger than the size "
//                   "given to the allocator"
//                << std::endl;
//      exit(1);
//    }
//
//    check_bounds(memgraph, memgraph.mem_sizes()[0]);
//    execute_test(memgraph);
//  }
//
//  return 0;
//}

// testing 3d matmul on a multiple GPUs
int main_matmul_multi_gpu(int argc, char **argv) {

  if (argc != 8) {
    usage();
    return 1;
  }

  int pi, pj, pk;
  uint64_t di, dj, dk;
  int np;
  try {
    pi = parse_with_ss<int>(argv[1]);
    pj = parse_with_ss<int>(argv[2]);
    pk = parse_with_ss<int>(argv[3]);
    di = parse_with_ss<uint64_t>(argv[4]);
    dj = parse_with_ss<uint64_t>(argv[5]);
    dk = parse_with_ss<uint64_t>(argv[6]);
    np = parse_with_ss<int>(argv[7]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
  // it could be the case that not all locs are actually used,
  // for example 1 1 2 100 100 100 88
  // Here only 2 locs will really be used, not all 88...
  np = taskgraph.num_locs();

  // have everyone share the same cache
  vector<int> compute_loc_to_cache(np, 0);

  size_t allocator_size = 6lu * 1024lu * 1024lu * 1024lu;

  vector<uint64_t> mem_sizes;

  for (int i = 0; i < np; ++i){
    mem_sizes.push_back(allocator_size);
  }

  {
    tuple<map<int, mem_t>, // input -> mem
          map<int, mem_t>, // save -> mem
          memgraph_t>
        _info1 = memgraph_t::make_without_evict(
            taskgraph, mem_sizes,
            {allocator_strat_t::lowest_dependency, 4});
    auto const &[_2, _3, memgraph] = _info1;

    std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
    std::ofstream f("mm3d_mem_lowest_dep.gv");
    memgraph.print_graphviz(f);

    // Do some checks before we execute
    for (int i = 0; i < np; ++i){
      // check if the sizes of the memgraph is lower than what we have given
      if (memgraph.mem_sizes()[i] > allocator_size) {
        std::cout << "Error: the size of the memgraph is larger than the size "
                    "given to the allocator"
                  << std::endl;
        exit(1);
      }

      // print the memgraph sizes on all gpus
      std::cout << "memgraph size on gpu " << i << ": " << memgraph.mem_sizes()[i] << std::endl;

      check_bounds(memgraph, memgraph.mem_sizes()[i]);
    }
    
    execute_multi_gpu_test(memgraph);
  }

  return 0;
}

// testing on how contraction works
int main_contraction() {
  //contractionTest(5, 5, 10);
  return 0;
}

// testing the allocator gives alignment and create no error
int main_alignment() {
  //alignmentTest(5, 7, 14);
  return 0;
}

// testing if the GPU can run a layer of deep ff network
// Note: cannot check correctness because the CPU reference is very slow
//int main_ff() {
//  using id_t = graph_writer_t::tensor_t;
//
//  graph_writer_t writer;
//
//  uint64_t nb = 100;
//  uint64_t nw = 101;
//
//  id_t x = writer.input({nb, nw});
//
//  uint64_t nw_prev = nw;
//  int nlayer = 4;
//  for (int i = 0; i != nlayer; ++i) {
//    uint64_t nw_next = runif(20, 121);
//    id_t w = writer.input({nw_prev, nw_next});
//    x = writer.matmul(x, w);
//    nw_prev = nw_next;
//  }
//
//  auto result = x.save();
//
//  graph_t const &graph = writer.get_graph();
//
//  random_placement_t random_placement{.part_size_rng = {2, 4}, .nloc = 1};
//
//  vector<placement_t> placements;
//  placements.reserve(graph.nodes.size());
//  for (int gid = 0; gid != graph.nodes.size(); ++gid) {
//    auto const &node = graph.nodes[gid];
//    placements.push_back(random_placement(node.op.shape()));
//  }
//
//  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);
//
//  int np = taskgraph.num_locs();
//  vector<uint64_t> compute_loc_to_cache(np, 0);
//
//  std::cout << "Generating memgraph now" << std::endl;
//
//  auto [_2, _3, memgraph] =
//      memgraph_t::make_without_evict(taskgraph, {10000000});
//
//  // print the number of nodes in the graph
//  std::cout << "Number of nodes in the graph: " << memgraph.nodes.size()
//            << std::endl;
//  // print the input and output of every node
//  // for(int i = 0; i < memgraph.nodes.size(); ++i) {
//  //   std::cout << "Node " << i << " has input: ";
//  //   for(auto in: memgraph.nodes[i].inns) {
//  //     std::cout << in << " ";
//  //   }
//  //   std::cout << "and output: ";
//  //   for(auto out: memgraph.nodes[i].outs) {
//  //     std::cout << out << " ";
//  //   }
//  //   std::cout << std::endl;
//  // }
//
//  std::ofstream f("deepff.gv");
//  memgraph.print_graphviz(f);
//  mem_check(memgraph);
//  std::cout << "Starting execution" << std::endl;
//
//  execute(memgraph, gpu_allocate_memory(memgraph.mem_sizes()[0], 0));
//  // check_correctness(memgraph, false);
//  return 0;
//}

int main(int argc, char **argv) {
  // main_ff();
  // main_matmul(argc, argv);
  main_matmul_multi_gpu(argc, argv);
  // contractionTest2();
  return 0;
}
