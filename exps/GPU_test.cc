#include "GPU_correctness.cc"
#include "gpu_kernel_manager.h"
#include "utility.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/types.h>
#include "../../src/engine/exec_graph.h"

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



memgraph_t taskgraph_to_memgraph(taskgraph_t const& taskgraph) {

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
      // std::cout << "memgraph size on gpu " << i << ": " << memgraph.mem_sizes()[i] << std::endl;

      check_bounds(memgraph, memgraph.mem_sizes()[i]);
    }
    
    return memgraph;
  }
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

void engine_1(int argc, char** argv){
  if (argc != 6) {
    DOUT("pi pj pk matrix_dimension np");
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
    dj = parse_with_ss<uint64_t>(argv[4]);
    dk = parse_with_ss<uint64_t>(argv[4]);
    np = parse_with_ss<int>(argv[5]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return;
  }

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());

  memgraph_t memgraph = taskgraph_to_memgraph(taskgraph);

  int num_gpus_per_node = 4;
  translate_execute(memgraph, true, num_gpus_per_node);
}

void server_1 (int argc, char** argv){
  if (argc != 4){
    DOUT("Square matrix multiplication");
    DOUT("1) world_size 2) matrix_dim 3) num_partition");
    return;
  }
  int world_size, partition;
  uint64_t matrix_dim;
  try {
    world_size = parse_with_ss<int>(argv[1]);
    matrix_dim = parse_with_ss<uint64_t>(argv[2]);
    partition = parse_with_ss<int>(argv[3]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("1) world_size 2) matrix_dim 3) num_partition");
    return;
  }

  server_execute_mm(world_size, matrix_dim, partition);
}

// do (A*B) * (C*D) * (E*F)... on the server
void server_multiple_mm (int argc, char** argv){
  if (argc != 3){
    DOUT("Square matrix multiplication");
    DOUT("1) matrix_dim 2) num gpus");
    return;
  }
  int world_size = 1;
  int num_gpus;
  uint64_t matrix_dim;
  try {
    matrix_dim = parse_with_ss<int>(argv[1]);
    num_gpus = parse_with_ss<uint64_t>(argv[2]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("1) world_size 2) matrix_dim");
    return;
  }

  server_execute_multiple_mm(world_size, matrix_dim, num_gpus);
}

void server_mm_partition(int argc, char** argv){
  if (argc != 4){
    DOUT("Square matrix multiplication");
    DOUT("1) matrix_dim 2) num gpus 3)partition");
    return;
  }
  int world_size = 1;
  int num_gpus;
  uint64_t matrix_dim;
  int partition;
  try {
    matrix_dim = parse_with_ss<int>(argv[1]);
    num_gpus = parse_with_ss<uint64_t>(argv[2]);
    partition = parse_with_ss<int>(argv[3]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("1) matrix_dim 2) num gpus 3)partition");
    return;
  }

  server_execute_mm_partition(matrix_dim, num_gpus, partition);
}


// do 3d matmul on the server
void server_3d_matmul (int argc, char** argv){
  if (argc != 8) {
    DOUT("pi pj pk di dj dk np");
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

  // time the execution
  auto start = std::chrono::high_resolution_clock::now();

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);
  
  auto graph = g.graph;

  auto pls = g.get_placements();

  {
    std::ofstream f("3d_matmul.gv");
    vector<partition_t> part;
    for (auto const& p : pls) {
      part.push_back(p.partition);
    }

    graph.print_graphviz(f, part);
    DOUT("printed 3d_matmul.gv");
  }

  int world_size = 1;

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 4lu * 1024lu * 1024lu * 1024lu;
  // uint64_t mem_size = 0.001 * 1024lu * 1024lu * 1024lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  gpu_mg_server_t server(c, buffer_sizes);
  server.set_split_off_inputs(false);

  // initialize input tensors and distribute across the cluster
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& input = node.op.get_input();
      dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
      tensor.random("-0.01", "0.01");
      // tensor.ones();
      // DOUT(tensor);
      server.insert_tensor(gid, pls[gid], tensor);
    }
  }
  // Time the random initialization
  auto data_init_time = std::chrono::high_resolution_clock::now();
  auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_init_time-start);
  DOUT("Random initialization time: " << init_duration.count() / 1000000.0 << " seconds");
  // DOUT("Printing graphviz...")
  // std::ofstream f("g_multiply.gv");
  // graph.print_graphviz(f);

  server.execute_graph(graph, pls);

  auto execution_time = std::chrono::high_resolution_clock::now();
  auto execution_duration = std::chrono::duration_cast<std::chrono::microseconds>(execution_time-data_init_time);
  DOUT("Server execution time: " << execution_duration.count() / 1000000.0 << " seconds");

  //// get the outputs to here
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_save()) {
      dbuffer_t tensor = server.get_tensor_from_gid(gid);
      // DOUT(tensor);
      //DOUT("gid sum is: " << tensor.sum());
    }
  }

  // print the execution time in seconds
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
  DOUT("Total server time: " << duration.count() / 1000000.0 << " seconds");

  server.shutdown();

}

// do softmax on the server
void server_softmax (int argc, char** argv){

  int np;
  try {
    np = parse_with_ss<int>(argv[1]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("np matrix_dimension partition");
    return;
  }

  np = 2;

  graph_writer_t g;

  auto input = g.input({1000, 1000});
  auto out = g.softmax_v3_scale(scalar_t(float(0.1)), input);
  
  auto graph = g.get_graph();
  // auto pls = autoplace(graph, np);

  // manually create the partitions
  vector<partition_t> partitions;
  partition_t my_part({partdim_t::split(1000, 1), partdim_t::split(1000, 2)});
  partition_t my_part2({partdim_t::split(1000, 1)});
  partitions.emplace_back(my_part);
  partitions.emplace_back(my_part);
  partitions.emplace_back(my_part2);
  partitions.emplace_back(my_part);
  partitions.emplace_back(my_part2);
  partitions.emplace_back(my_part);

  vector<placement_t> pls;
  for (auto part: partitions){
    pls.emplace_back(part);
    vector<int>& locs = pls.back().locations.get();
    for (int i = 0; i < locs.size(); ++i){
      locs[i] = i;
    }
  }

  {
    std::ofstream f("softmax.gv");
    // vector<partition_t> part;
    // for (auto const& p : pls) {
    //   part.push_back(p.partition);
    // }

    graph.print_graphviz(f, partitions);
    DOUT("printed softmax.gv");
  }

  int world_size = 1;

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 4lu * 1024lu * 1024lu * 1024lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  gpu_mg_server_t server(c, buffer_sizes);
  server.set_split_off_inputs(false);

  // initialize input tensors and distribute across the cluster
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& input = node.op.get_input();
      dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
      tensor.random("-0.01", "0.01");
      // tensor.ones();
      // DOUT(tensor);
      server.insert_tensor(gid, pls[gid], tensor);
    }
  }

  server.execute_graph(graph, pls);

  //// get the outputs to here
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_save()) {
      dbuffer_t tensor = server.get_tensor_from_gid(gid);
      // DOUT(tensor);
      //DOUT("gid sum is: " << tensor.sum());
    }
  }

  server.shutdown();

}

// running feed forward neural network on the server
void server_ffnn(int argc, char** argv){

  // time the execution
  auto start = std::chrono::high_resolution_clock::now();

  // DEFINE PARAMETERS HERE
  uint64_t batch_size = 256;
  uint64_t H_1 = 1 << 10;
  uint64_t H_2 = 1 << 14;
  uint64_t output_class = 1 << 14;
  uint64_t input_dim = 1 << 19;

  uint64_t H_test = 100;
  uint64_t output_test = 10;
  uint64_t input_dim_test = 10;

  vector<uint64_t> dims = {input_dim, H_2, output_class};
  int np;
  int partition;

  if (argc != 3) {
    DOUT("np partition");
    return;
  }

  try {
    np = parse_with_ss<int>(argv[1]);
    partition = parse_with_ss<int>(argv[2]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("np partition");
    return;
  }

  // auto graph = generate_ffnn(batch_size, dims);
  auto graph = ffnn_specific();
  auto pls = autoplace(graph, np, partition);
  int world_size = 1;

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 16lu * 1000lu * 1000lu * 1000lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  gpu_mg_server_t server(c, buffer_sizes);
  server.set_split_off_inputs(true);

  // initialize input tensors and distribute across the cluster
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& input = node.op.get_input();
      dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
      tensor.random("-0.01", "0.01");
      // tensor.ones();
      // DOUT(tensor);
      server.insert_tensor(gid, pls[gid], tensor);
    }
  }
  
  // Time the random initialization
  auto data_init_time = std::chrono::high_resolution_clock::now();
  auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(data_init_time-start);
  DOUT("Random initialization time: " << init_duration.count() / 1000000.0 << " seconds");
  // DOUT("Printing graphviz...")
  // std::ofstream f("g_multiply.gv");
  // graph.print_graphviz(f);

  server.execute_graph(graph, pls);

  auto execution_time = std::chrono::high_resolution_clock::now();
  auto execution_duration = std::chrono::duration_cast<std::chrono::microseconds>(execution_time-data_init_time);
  DOUT("Server execution time: " << execution_duration.count() / 1000000.0 << " seconds");

  //// get the outputs to here
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_save()) {
      dbuffer_t tensor = server.get_tensor_from_gid(gid);
      // DOUT(tensor);
      //DOUT("gid sum is: " << tensor.sum());
    }
  }

  // print the execution time in seconds
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
  DOUT("Total server time: " << duration.count() / 1000000.0 << " seconds");

  server.shutdown();
}

void mm_graph(int argc, char** argv){
  graph_writer_t g;

  int np;
  uint64_t matrix_dim;
  int partition;

  if (argc != 4) {
    DOUT("np matrix_dimension partition");
    return;
  }

  try {
    np = parse_with_ss<int>(argv[1]);
    matrix_dim = parse_with_ss<int>(argv[2]);
    partition = parse_with_ss<int>(argv[3]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("np matrix_dimension partition");
    return;
  }
  auto A = g.input({matrix_dim, matrix_dim});
  auto B = g.input({matrix_dim, matrix_dim});
  auto E = g.matmul(A,B);
  auto C = g.input({matrix_dim, matrix_dim});
  auto D = g.input({matrix_dim, matrix_dim});
  auto F = g.matmul(C,D);
  auto G = g.matmul(E,F).save();

  graph_t graph = g.get_graph();
  auto pls = autoplace(graph, np);

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);
  uint64_t mem_size = 2lu * 1000lu * 1000lu * 1000lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  bool use_storage = true;
  bool split_off_inputs = false;

  auto [_2, _3, maybe_init_memgraph, core_memgraph] = memgraph_t::make_(
    taskgraph, {}, buffer_sizes, {}, allocator_settings_t::gpu_alignment_settings(),
    use_storage, split_off_inputs);

  std::cout << "mm_graph.gv" << std::endl;
  std::ofstream f("mm_graph.gv");
  core_memgraph.print_graphviz(f);
  
}

void mm_part_graph(int argc, char** argv){

  int np;
  uint64_t matrix_dim;
  int partition;

  if (argc != 4) {
    DOUT("np matrix_dimension partition");
    return;
  }

  try {
    np = parse_with_ss<int>(argv[1]);
    matrix_dim = parse_with_ss<int>(argv[2]);
    partition = parse_with_ss<int>(argv[3]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("np matrix_dimension partition");
    return;
  }

  auto [graph, part] = build_matmul_even_splits(matrix_dim, partition);
  auto pls = alocate01(graph, part, np, 100);

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

  taskgraph_stats(taskgraph);

  uint64_t mem_size = 2lu * 1000lu * 1000lu * 1000lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  bool use_storage = true;
  bool split_off_inputs = true;

  auto [_2, _3, maybe_init_memgraph, core_memgraph] = memgraph_t::make_(
    taskgraph, {}, buffer_sizes, {}, allocator_settings_t::gpu_alignment_settings(),
    use_storage, split_off_inputs);

  memgraph_mem_stats(core_memgraph);

  std::cout << "mm_part_graph.gv" << std::endl;
  std::ofstream f("mm_part_graph.gv");
  core_memgraph.print_graphviz(f);
}


void ffnn_graph(int argc, char** argv){
  int np;
  int partition;

  if (argc != 3) {
    DOUT("np partition");
    return;
  }

  try {
    np = parse_with_ss<int>(argv[1]);
    partition = parse_with_ss<int>(argv[2]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("np partition");
    return;
  }

  // auto graph = generate_ffnn(batch_size, dims);
  auto graph = ffnn_specific();
  auto pls = autoplace(graph, np, partition);
  int world_size = 1;

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

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

  auto to_mb = [](uint64_t n) { return double(n)/1e6; };
  DOUT("input "
      << num_input_msgs << "#, " << to_mb(num_input_bytes) << "MB, "
      << num_core_msgs << "#, " << to_mb(num_core_bytes) << "MB");

  uint64_t mem_size = 16lu * 1000lu * 1000lu * 1000lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < np; ++i){
    buffer_sizes.push_back(mem_size);
  }

  bool use_storage = true;
  bool split_off_inputs = true;

  auto [_2, _3, maybe_init_memgraph, core_memgraph] = memgraph_t::make_(
    taskgraph, {}, buffer_sizes, {}, allocator_settings_t::gpu_alignment_settings(),
    use_storage, split_off_inputs);

  std::cout << "ffnn_graph.gv" << std::endl;
  std::ofstream f("ffnn_graph.gv");
  core_memgraph.print_graphviz(f);

  int num_evict_nodes = 0;
  uint64_t num_evict_bytes = 0;
  int num_load_nodes = 0;
  uint64_t num_load_bytes = 0;
  for (int tid = 0; tid < core_memgraph.nodes.size(); ++tid){
    auto const& node = core_memgraph.nodes[tid];
    if (node.op.is_evict()){
      num_evict_nodes++;
      num_evict_bytes += node.op.get_evict().src.size;
    }
    else if (node.op.is_load()){
      num_load_nodes++;
      num_load_bytes += node.op.get_load().dst.size;
    }
  }
  DOUT("evict " << num_evict_nodes << "#, " << to_mb(num_evict_bytes) << "MB");
  DOUT("load " << num_load_nodes << "#, " << to_mb(num_load_bytes) << "MB");
}

void mm_test2() {
  dtype_t dtype = dtype_t::f32;

  void* a;
  void* b;
  void* c;
  void* w;

  kernel_manager_t km;

  uint64_t ni = 10;

  handle_cuda_error(cudaMalloc(&a, ni*ni*dtype_size(dtype)));
  handle_cuda_error(cudaMalloc(&b, ni*ni*dtype_size(dtype)));
  handle_cuda_error(cudaMalloc(&c, ni*ni*dtype_size(dtype)));

  dbuffer_t output = make_dbuffer(dtype_t::f32, ni*ni);
  output.random("-1.0", "1.0");

  if (cudaMemcpy(c, output.data->data, output.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy output");
  }

  std::cout << "c before: " << std::endl;
  printFloatGPU(c, ni*ni);

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
  std::cout << "c after: " << std::endl;
  printFloatGPU(c, ni*ni);
}

// void cublaMatmulCheck(){
//   auto iter = 10000;
//   for (int i = 0; i < iter; ++i){
//     cudaSetDevice(0);
//     // generate random ints of ni, nj, nk in range (1, 100)
//     int ni = runif(1, 100);
//     int nj = runif(1, 100);
//     int nk = runif(1, 100);
//     bool trans_l = runif(1);
//     bool trans_r = runif(1);
//     bool swap = runif(1);
//     dbuffer_t input_1 = make_dbuffer(dtype_t::f32, ni*nj);
//     input_1.ones();
//     dbuffer_t input_2 = make_dbuffer(dtype_t::f32, nj*nk);
//     input_2.ones();

//     dbuffer_t output = make_dbuffer(dtype_t::f32, ni*nk);

//     kernel_manager_t::matmul_t matmul;
//     matmul.dtype = dtype_t::f32;
//     matmul.ni = ni;
//     matmul.nj = nj;
//     matmul.nk = nk;
//     matmul.trans_l = trans_l;
//     matmul.trans_r = trans_r;
//     matmul.swap = swap;

//     void* gpu_input_1 = gpu_allocate_memory(input_1.size(), 0);
//     void* gpu_input_2 = gpu_allocate_memory(input_2.size(), 0);
//     cudaMemcpy(gpu_input_1, input_1.raw(), input_1.size(), cudaMemcpyHostToDevice);
//     cudaMemcpy(gpu_input_2, input_2.raw(), input_2.size(), cudaMemcpyHostToDevice);

//     void* gpu_output = gpu_allocate_memory(output.size(), 0);

//     kernel_manager_t km(0);
//     auto stream = cuda_create_stream();
//     km.execute_matmul(matmul, stream, gpu_output, gpu_input_1, gpu_input_2);
//     handle_cuda_error(cudaStreamSynchronize(stream));
//     cudaMemcpy(output.raw(), gpu_output, output.size(), cudaMemcpyDeviceToHost);

//     DOUT(output.max());
//     DOUT(output.min());
//     // throw an error if min and max are not the same
//     if (output.max() != output.min()){
//       throw std::runtime_error("cublaMatmulCheck failed");
//     }
//   }
// }

// void reluCheck(){
//   uint64_t size = 10000;
//   auto buffer_size = size * sizeof(float);
//   graph_writer_t writer;
//   tensor_t x = writer.input({size, size});
//   auto 
// }

scalarop_t scalarop_km(){
  dtype_t dtype = dtype_t::f32;
  // build the scalarop:
  //  a,a->a | +[*[constant{f32|0.999},hole|f32@0],*[constant{f32|0.000999987},power{2}[hole|f32@1]]]
  scalar_t coeff1 = scalar_t(dtype, "0.999");
  scalar_t coeff2 = scalar_t(dtype, "0.000999987");
  scalarop_t constant1 = scalarop_t::make_constant(coeff1);
  scalarop_t constant2 = scalarop_t::make_constant(coeff2);
  scalarop_t power = scalarop_t::make_power(2);
  scalarop_t mul = scalarop_t::make_mul(dtype);
  scalarop_t add = scalarop_t::make_add(dtype);
  scalarop_t arg = scalarop_t::make_arg(0, dtype);
  scalarop_t ret1 = scalarop_t::combine(mul, {constant1, arg});
  scalarop_t ret2 = scalarop_t::combine(mul, {constant2, scalarop_t::combine(power, {arg})});
  scalarop_t ret = scalarop_t::combine(add, {ret1, ret2});
  // auto [str, bytes] = ret.to_cpp_bytes();
  // DOUT(str);
  // // cast bytes to a float and print it
  // float* f = (float*)bytes.data();
  // DOUT(*f);
  // // cast bytes + 4 to a float and print it
  // f = (float*)(bytes.data() + 4);
  // DOUT(*f);

  // build the scalarop: +[*[constant{f32|0.999},hole|f32@0],*[constant{f32|0.000999987}, *[hole|f32@1, [hole|f32@1]]
  scalarop_t ret2_new = scalarop_t::combine(mul, {constant2, scalarop_t::replace_arguments(mul, {arg, arg})});
  scalarop_t ret_new = scalarop_t::combine(add, {ret1, ret2_new});

  DOUT(ret_new);

  return ret;
}

scalarop_t scalarop_km2(){
  // build es[512,64]+ ab,ab,a->a | *[hole|f32@0,*[hole|f32@1,*[constant{f32|-1},power{-2}[hole|f32@2]]]]
  dtype_t dtype = dtype_t::f32;
  scalarop_t one = scalarop_t::make_constant(scalar_t::one(dtype));
  scalarop_t arg0 = scalarop_t::make_arg(0, dtype);
  scalarop_t arg1 = scalarop_t::make_arg(1, dtype);
  scalarop_t arg2 = scalarop_t::make_arg(2, dtype);
  scalarop_t mul = scalarop_t::make_mul(dtype);
  scalarop_t neg = scalarop_t::make_neg(dtype);
  scalarop_t power = scalarop_t::make_power(-2);
  scalarop_t ret = scalarop_t::replace_arguments(power, {arg2});
  ret = scalarop_t::replace_arguments(neg, {ret});
  ret = scalarop_t::replace_arguments(mul, {arg1, ret});
  ret = scalarop_t::replace_arguments(mul, {arg0, ret});
  DOUT(ret);
  auto [str, bytes] = ret.to_cpp_bytes();
  DOUT(str);
  auto power_num = *((double*)(bytes.data() + 4));
  DOUT("Power is " << power_num);
  if (power_num == -2){
    DOUT("Power is -2");
  }

  // x**(-2) = 1/(x**2) = 1/(x*x)
  scalarop_t div = scalarop_t::make_div(dtype);
  scalarop_t new_ret = scalarop_t::replace_arguments(div, {one, scalarop_t::replace_arguments(mul, {arg2, arg2})});
  DOUT("new_ret 1): " << new_ret);
  new_ret = scalarop_t::replace_arguments(neg, {new_ret});
  DOUT("new_ret 2): " << new_ret);
  new_ret = scalarop_t::replace_arguments(mul, {arg1, new_ret});
  DOUT("new ret 3): " << new_ret);
  new_ret = scalarop_t::replace_arguments(mul, {arg0, new_ret});
  DOUT("new ret is " << new_ret);
  return ret;
}

// ite_==[hole|f32@0,hole|f32@1,constant{f32|1},constant{f32|0}]
scalarop_t scalarop_km3(){
  scalarop_t arg0 = scalarop_t::make_arg(0, dtype_t::f32);
  scalarop_t arg1 = scalarop_t::make_arg(1, dtype_t::f32);
  scalarop_t is_equal = scalarop_t::make_is_equal(dtype_t::f32);
  scalarop_t ret = scalarop_t::replace_arguments(is_equal, {arg0, arg1});
  DOUT(ret);
  DOUT(ret.to_cppstr());

  return ret;
}

// ab,ab,a->a | *[hole|f32@0,*[hole|f32@1,*[constant{f32|-1},power{-2}[hole|f32@2]]]]
scalarop_t scalarop_km4(){
  scalarop_t arg0 = scalarop_t::make_arg(0, dtype_t::f32);
  scalarop_t arg1 = scalarop_t::make_arg(1, dtype_t::f32);
  scalarop_t arg2 = scalarop_t::make_arg(2, dtype_t::f32);
  scalarop_t mul = scalarop_t::make_mul(dtype_t::f32);
  scalarop_t neg = scalarop_t::make_neg(dtype_t::f32);
  scalarop_t power = scalarop_t::make_power(-2);
  scalarop_t ret = scalarop_t::replace_arguments(power, {arg2});
  ret = scalarop_t::replace_arguments(neg, {ret});
  ret = scalarop_t::replace_arguments(mul, {arg1, ret});
  ret = scalarop_t::replace_arguments(mul, {arg0, ret});
  DOUT(ret);
  DOUT(ret.to_cppstr());
  return ret;
}

int main(int argc, char **argv) {
  // server_1(argc, argv);
  // server_3d_matmul(argc, argv);
  // server_multiple_mm(argc, argv);
  // engine_1(argc, argv);
  // cublaMatmulCheck();
  // server_ffnn(argc, argv);
  // ffnn_graph(argc, argv);
  // mm_graph(argc, argv);
  // mm_part_graph(argc, argv);
  // server_mm_partition(argc, argv);
  // lowerTri_test();
  // constant_test(); 
  // ew_test();
  // scalarop_km4();
  server_softmax(argc, argv);
}
