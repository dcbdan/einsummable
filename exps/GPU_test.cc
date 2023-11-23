#include "GPU_correctness.cc"
#include "gpu_kernel_manager.h"
#include "utility.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>

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
    DOUT("1) world_size 2) matrix_dim");
    return;
  }
  int world_size, partition;
  uint64_t matrix_dim;
  try {
    world_size = parse_with_ss<int>(argv[1]);
    matrix_dim = parse_with_ss<uint64_t>(argv[2]);
  } catch (...) {
    std::cout << "Parse error." << std::endl << std::endl;
    DOUT("1) world_size 2) matrix_dim");
    return;
  }

  server_execute_multiple_mm(world_size, matrix_dim, partition);
}


// do 3d matmul on the server
void server_3d_mamtmul (int argc, char** argv){
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

  // time the execution
  auto start = std::chrono::high_resolution_clock::now();

  auto g = three_dimensional_matrix_multiplication(pi, pj, pk, di, dj, dk, np);
  auto graph = g.graph;
  auto pls = g.get_placements();
  int world_size = 1;

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 6lu * 1024lu * 1024lu * 1024lu;
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

// running feed forward neural network on the server
void server_ffnn(){

  // time the execution
  auto start = std::chrono::high_resolution_clock::now();

  // DEFINE PARAMETERS HERE
  // int batch_size = 100;
  // int num_layers = 3;
  // vector<uint64_t> dims = {784, 100, 10};
  int np = 1;
  int batch_size = 100;
  int num_layers = 2;
  vector<uint64_t> dims = {784, 10};

  auto graph = generate_ffnn(batch_size, dims);
  auto pls = autoplace(graph, np);
  int world_size = 1;

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 6lu * 1024lu * 1024lu * 1024lu;
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

int main(int argc, char **argv) {
  // server_1(argc, argv);
  // server_3d_mamtmul(argc, argv);
  // server_multiple_mm(argc, argv);
  // engine_1(argc, argv);
  // cublaMatmulCheck();
  server_ffnn();
}
