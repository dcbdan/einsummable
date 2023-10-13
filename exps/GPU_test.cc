#include "../src/einsummable/memgraph.h"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute_multi_gpu.h"

#include "GPU_correctness.cc"

#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"
#include "../src/engine/gpu/workspace.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

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

void execute_compile_from_taskgraph(taskgraph_t const& taskgraph) {
  // it could be the case that not all locs are actually used,
  // for example 1 1 2 100 100 100 88
  // Here only 2 locs will really be used, not all 88...
  int np = taskgraph.num_locs();

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
}

memgraph_t taskgraph_to_memgraph(taskgraph_t const& taskgraph) {
  // it could be the case that not all locs are actually used,
  // for example 1 1 2 100 100 100 88
  // Here only 2 locs will really be used, not all 88...
  int np = taskgraph.num_locs();

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
    
    return memgraph;
  }
}

// testing 3d matmul on a multiple GPUs
void main_matmul_multi_gpu(int argc, char **argv) {
  if (argc != 8) {
    usage();
    return;
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
    return;
  }

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  execute_compile_from_taskgraph(taskgraph);
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

template <typename T>
void slow_mm_(
  uint64_t ni, uint64_t nj, uint64_t nk,
  T* out, T const* lhs, T const* rhs)
{
  std::fill(out, out + ni*nk, 0);
  for(uint64_t i = 0; i != ni; ++i) {
  for(uint64_t j = 0; j != nj; ++j) {
  for(uint64_t k = 0; k != nk; ++k) {
    out[i*nk + k]   += 
      lhs[i*nj + j]  * 
      rhs[j*nk + k]  ;
  }}}
}

void slow_mm(dtype_t const& dd, 
  uint64_t ni, uint64_t nj, uint64_t nk,
  void* out, void const* lhs, void const* rhs)
{
  if(dd == dtype_t::f16) {
    slow_mm_(ni,nj,nk,
      static_cast<float16_t*>(out),
      static_cast<float16_t const*>(lhs),
      static_cast<float16_t const*>(rhs));
  } else if(dd == dtype_t::f32) {
    slow_mm_(ni,nj,nk,
      static_cast<float*>(out),
      static_cast<float const*>(lhs),
      static_cast<float const*>(rhs));
  } else if(dd == dtype_t::f64) {
    slow_mm_(ni,nj,nk,
      static_cast<double*>(out),
      static_cast<double const*>(lhs),
      static_cast<double const*>(rhs));
  } else {
    throw std::runtime_error("not implemented");
  }
}

void mm_test2() {
  dtype_t dtype = dtype_t::f32;

  void* a;
  void* b;
  void* c;
  void* w;

  kernel_manager_t km;

  uint64_t ni = 10000;

  handle_cuda_error(cudaMalloc(&a, ni*ni*dtype_size(dtype)));
  handle_cuda_error(cudaMalloc(&b, ni*ni*dtype_size(dtype)));
  handle_cuda_error(cudaMalloc(&c, ni*ni*dtype_size(dtype)));

  // A  B   C
  //ij,jk->ik
  einsummable_t e = einsummable_t::from_matmul(ni, ni, ni, dtype);

  uint64_t wsz = km.build(e).value().value();
  handle_cuda_error(cudaMalloc(&w, wsz));

  cudaStream_t stream;
  handle_cuda_error(cudaStreamCreate(&stream));
  DLINE;

  for(int i = 0; i != 10; ++i) {
    km(e, stream, c, {a,b}, tuple<void*, uint64_t>{w, wsz});
  }

  handle_cuda_error(cudaDeviceSynchronize());
}

void mm_test() {
  dtype_t dtype = dtype_t::f32;

  void* a;
  void* b;
  void* c;
  void* w;

  vector<tuple<uint64_t, uint64_t, uint64_t>> szs = {
    {1024, 1024, 1024},
    {2048, 1024, 1024},
    {1024, 2048, 1024},
    {1024, 1024, 2048},
    {2048,2048,2048},
    {4096,4096,4096},
    {16384,16384,16384},
    {20048, 20048, 20048},
    {20048, 20048, 20048},
    {20048, 20048, 20048},
    {20048, 20048, 20048},
    {20048, 20048, 20048}
  };

  bool test = false;

  kernel_manager_t km;

  for(auto const& [ni,nj,nk]: szs) {
    DOUT(ni << ", " << nj << ", " << nk);

    handle_cuda_error(cudaMalloc(&a, ni*nj*dtype_size(dtype)));
    handle_cuda_error(cudaMalloc(&b, nj*nk*dtype_size(dtype)));
    handle_cuda_error(cudaMalloc(&c, ni*nk*dtype_size(dtype)));
    DLINE;

    dbuffer_t aa     = make_dbuffer(dtype, test ? ni*nj : 1);
    dbuffer_t bb     = make_dbuffer(dtype, test ? nj*nk : 1);
    dbuffer_t cc_cpu = make_dbuffer(dtype, test ? ni*nk : 1);
    dbuffer_t cc_gpu = make_dbuffer(dtype, test ? ni*nk : 1);
    DLINE;

    if(test) {
      aa.random("-0.000001","0.000001");
      bb.random("-0.000001","0.000001");
      DLINE;
      handle_cuda_error(cudaMemcpy(a, aa.raw(), aa.size(), cudaMemcpyDefault));
      DLINE;
      handle_cuda_error(cudaMemcpy(b, bb.raw(), bb.size(), cudaMemcpyDefault));
      DLINE;
    }

    // A  B   C
    //ij,jk->ik
    einsummable_t e = einsummable_t::from_matmul(ni, nj, nk, dtype);

    //reference_einsummable_inplace(e, cc_cpu, {aa,bb});
    if(test) {
      slow_mm(dtype, ni, nj, nk, cc_cpu.raw(), aa.raw(), bb.raw());
      DLINE;
    }

    DLINE;
 
    DLINE;
    uint64_t wsz = km.build(e).value().value();
    DOUT("wsz is " << wsz);
    handle_cuda_error(cudaMalloc(&w, wsz));
    DLINE;

    cudaStream_t stream;
    handle_cuda_error(cudaStreamCreate(&stream));
    DLINE;

    km(e, stream, c, {a,b}, tuple<void*, uint64_t>{w, wsz});
    
    DLINE;
    handle_cuda_error(cudaStreamDestroy(stream));

    DLINE;
    handle_cuda_error(cudaDeviceSynchronize());

    if(test) {
      handle_cuda_error(cudaMemcpy(cc_gpu.raw(), c, cc_gpu.size(), cudaMemcpyDefault));
      if(!is_close(cc_cpu, cc_gpu)) {
        throw std::runtime_error("is not close!");
      }
    }

    handle_cuda_error(cudaFree(a));
    handle_cuda_error(cudaFree(b));
    handle_cuda_error(cudaFree(c));
    handle_cuda_error(cudaFree(w));
  }
}

void mm_test3() {
  dtype_t dtype = dtype_t::f32;

  vector<void*> a;
  vector<void*> b;
  vector<void*> c;
  vector<void*> w;

  kernel_manager_t km;

  uint64_t ni = 10000;

  // A  B   C
  //ij,jk->ik
  einsummable_t e = einsummable_t::from_matmul(ni, ni, ni, dtype);

  uint64_t wsz = km.build(e).value().value();

  int nrep = 10;
  for(int i = 0; i != nrep; ++i) {
    a.emplace_back();
    b.emplace_back();
    c.emplace_back();
    w.emplace_back();
    handle_cuda_error(cudaMalloc(&a.back(), ni*ni*dtype_size(dtype)));
    handle_cuda_error(cudaMalloc(&b.back(), ni*ni*dtype_size(dtype)));
    handle_cuda_error(cudaMalloc(&c.back(), ni*ni*dtype_size(dtype)));
    handle_cuda_error(cudaMalloc(&w.back(), wsz));
  }

  cudaStream_t stream;
  handle_cuda_error(cudaStreamCreate(&stream));
  DLINE;

  for(int i = 0; i != nrep; ++i) {
    km(e, stream, c[i], {a[i],b[i]}, tuple<void*, uint64_t>{w[i], wsz});
  }

  handle_cuda_error(cudaDeviceSynchronize());

  for(int i = 0; i != nrep; ++i) {
    handle_cuda_error(cudaFree(a[i]));
    handle_cuda_error(cudaFree(b[i]));
    handle_cuda_error(cudaFree(c[i]));
    handle_cuda_error(cudaFree(w[i]));
  }
}

void dcb01() {
  uint64_t sz = 10000;
  graph_t graph;
  //{
  //  graph_writer_t gwriter;
  //  auto l = gwriter.input({sz,sz});
  //  auto r = gwriter.input({sz,sz});
  //  for(int i = 0; i != 5; ++i) {
  //    auto o = gwriter.matmul(l,r);
  //    o.save_inplace();
  //  }
  //  graph = gwriter.get_graph();
  //}
  {
    graph_writer_t gwriter;

    auto z = gwriter.input({sz,sz});
    for(int i = 0; i != 3; ++i) {
      auto w = gwriter.input({sz,sz});
      if(i % 2 == 0) {
        z = gwriter.matmul(z, w);
      } else {
        z = gwriter.matmul(w, z);
      }
    }
    z.save_inplace();

    graph = gwriter.get_graph();
  }

  auto const& [_0, _1, taskgraph] = 
    taskgraph_t::make(graph, graph.make_singleton_placement());
  
  execute_compile_from_taskgraph(taskgraph);
}

void execute_memgraph_gpu(
  memgraph_t const& memgraph,
  std::vector<void*> buffers)
{
  kernel_manager_t km;

  exec_graph_t graph =
    exec_graph_t::make_gpu_exec_graph(memgraph, 0, km);

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new gpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(buffers))
    }
  ));

  exec_state_t state(graph, resource_manager);

  state.event_loop();
}

void engine_1(int argc, char** argv){
  if (argc != 8) {
    usage();
    return;
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
    return;
  }

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  memgraph_t memgraph = taskgraph_to_memgraph(taskgraph);

  auto num_gpu = memgraph.mem_sizes().size();
  // allocate ptrs for gpu
  std::vector<void*> gpu_ptrs;
  auto mem_sizes = memgraph.mem_sizes();
  for (int i = 0; i < num_gpu; ++i){
    gpu_ptrs.push_back(gpu_allocate_memory(mem_sizes[i], i));
  }

  DOUT("executing...");
  execute_memgraph_gpu(memgraph, gpu_ptrs);
  DOUT("executed.");
}

int main(int argc, char **argv) {
//  // main_ff();
//  // main_matmul(argc, argv);
//  main_matmul_multi_gpu(argc, argv);
//  // contractionTest2();
//  return 0;

  //mm_test3();
  //mm_test2();
  //mm_test();
  // dcb01();
  engine_1(argc, argv);
}
