#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"
#include "../src/engine/gpu/workspace.h"
#include "../src/server/gpu/server.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/engine/communicator.h"

#include "../src/server/base.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include "../src/base/setup.h"
#include "../src/einsummable/reference.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sys/types.h>
#include <tuple>

// print the information of the memgraph
void print_memgraph(memgraph_t memgraph){
  // print the input and output of every node
  for (int i = 0; i < memgraph.nodes.size(); ++i) {
    auto node = memgraph.nodes[i];
    // print device location
    std::cout << "Device: " << node.op.get_loc() << " ";
    std::cout << "Node " << i << " has input: ";
    for (auto in : node.inns) {
      std::cout << in << " ";
    }
    std::cout << "and output: ";
    for (auto out : node.outs) {
      std::cout << out << " ";
    }
    std::cout << "Node type: ";
    node.op.print_type();
    if (node.op.is_touch()){
      // print the group id
      std::cout << " Group id: " << node.op.get_apply().group;
      auto mem_touch = node.op.get_apply().mems[0];
      // print the touch size
      std::cout << " Touch size: " << mem_touch.size;
    }
    if (node.op.is_move()){
      // print src and dst device
      std::cout << " Src device: " << node.op.get_move().get_src_loc();
      std::cout << " Dst device: " << node.op.get_move().get_dst_loc();
    }
    std::cout << std::endl;
  }
}

// check if the offset in mems is greater than the bound
// throw an error if it is
// also check if the offset + size is greater than the bound
void mem_t_check(std::vector<mem_t> mems, uint64_t bound) {
  for (auto mem : mems) {
    if (mem.offset > bound) {
      // print the offset and bound
      std::cout << "Offset: " << mem.offset << " Bound: " << bound << std::endl;
      throw std::runtime_error("Error: offset is greater than the bound.");
    }
    if (mem.offset + mem.size > bound) {
      // print the offset and bound
      std::cout << "Offset: " << mem.offset << " Bound: " << bound << std::endl;
      throw std::runtime_error("Error: offset + size is greater than the bound.");
    }
  }  
}

// check if all nodes in the memgraph are within the memory bound
void check_bounds(memgraph_t memgraph, uint64_t bound){
  // print the bound 
  for (auto node: memgraph.nodes){
    if (node.op.is_inputmem()){
      auto op = node.op.get_inputmem();
      if (op.offset + op.size > bound){
        std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: INPUTMEM");
      }
    }
    else if (node.op.is_apply()){
      auto op = node.op.get_apply();
      mem_t_check(op.mems, bound);
    }
    else if (node.op.is_move()){
      auto op = node.op.get_move();
      if (std::get<1>(op.src) + op.size > bound){
        std::cout << "Offset: " << std::get<1>(op.src) << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: MOVE");
      }
      if (std::get<1>(op.dst) + op.size > bound){
        std::cout << "Offset: " << std::get<1>(op.dst) << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: MOVE");
      }
    }
    else if (node.op.is_evict()){
      memloc_t src = node.op.get_evict().src;
      if (src.offset + src.size > bound){
        std::cout << "Offset: " << src.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: EVICT");
      }
    }
    else if (node.op.is_load()){
      memloc_t dst = node.op.get_load().dst;
      if (dst.offset + dst.size > bound){
        std::cout << "Offset: " << dst.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: LOAD");
      }
    }
    else if (node.op.is_partialize()){
      auto op = node.op.get_partialize();
      if (op.offset + op.size > bound){
        std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: PARTIALIZE");
      }
    }
    else if (node.op.is_alloc()){
      auto op = node.op.get_alloc();
      if (op.offset + op.size > bound){
        std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: ALLOC");
      }
    }
    else if (node.op.is_del()){
      auto op = node.op.get_del();
      if (op.offset + op.size > bound){
        std::cout << "Offset: " << op.offset << " Bound: " << bound << std::endl;
        throw std::runtime_error("Memory bound exceeded: DEL");
      }
    }
  }
}

void translate_execute(memgraph_t memgraph, bool debug, int num_gpus_per_node){
  if (debug){
    print_memgraph(memgraph);
  }

  DOUT("Translate and execute memgraph");

  auto num_gpu = memgraph.mem_sizes().size();
  // allocate ptrs for gpu
  std::vector<void*> gpu_ptrs;
  auto mem_sizes = memgraph.mem_sizes();
  for (int i = 0; i < num_gpu; ++i){
    gpu_ptrs.push_back(gpu_allocate_memory(mem_sizes[i], i));
  }

  kernel_manager_t km;

  exec_graph_t graph =
    exec_graph_t::make_gpu_exec_graph(memgraph, 0, km, num_gpus_per_node, gpu_ptrs[0]);

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new gpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(gpu_ptrs))
    }
  ));

  exec_state_t state(graph, resource_manager);

  DOUT("executing...");
  state.event_loop();
  DOUT("executed.");
}

tuple<graph_t, vector<placement_t>> build_graph_pls(
  int world_size, uint64_t matrix_dim, int partition)
{
  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  ni = nj = nk = matrix_dim;
  li = lj = partition;
  rj = rk = partition;
  ji = jj = jk = partition;
  oi = ok = partition;

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


void server_execute(int world_size, uint64_t matrix_dim, int partition){

  communicator_t c("0.0.0.0", true, world_size);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;
  uint64_t mem_size = 6lu * 1024lu * 1024lu * 1024lu;
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < world_size; ++i){
    buffer_sizes.push_back(mem_size);
  }

  gpu_mg_server_t server(c, buffer_sizes);

  auto [graph, pls] = build_graph_pls(world_size, matrix_dim, partition);

  // initialize input tensors and distribute across the cluster
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& input = node.op.get_input();
      dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
      // tensor.random("-0.01", "0.01");
      tensor.ones();
      DOUT(tensor);
      server.insert_tensor(gid, pls[gid], tensor);
    }
  }
  // DOUT("Printing graphviz...")
  // std::ofstream f("g_multiply.gv");
  // graph.print_graphviz(f);

  server.execute_graph(graph, pls);

  //// get the outputs to here
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_save()) {
      dbuffer_t tensor = server.get_tensor_from_gid(gid);
      DOUT(tensor);
      //DOUT("gid sum is: " << tensor.sum());
    }
  }

  server.shutdown();
}

void contractionTest(int di, int dj, int dk) {
  auto num_elems = di * dj + dj * dk + di * dk;
  auto buffer_size = num_elems * sizeof(float);
  // create the einsummable
  auto einsummable = einsummable_t::from_matmul(di, dj, dk);
  einsummable = einsummable.merge_adjacent_dims();
  // create two input dbuffers
  dbuffer_t input1 = make_dbuffer(dtype_t::f32, di * dj);
  dbuffer_t input2 = make_dbuffer(dtype_t::f32, dj * dk);
  // input1.random("-1.0", "1.0");
  // input2.random("-1.0", "1.0");
  input1.ones();
  input2.ones();
  dbuffer_t output = make_dbuffer(dtype_t::f32, di * dk);
  output.random("-1.0", "1.0");
  // print cpu output
  std::cout << "CPU output: " << std::endl;
  printFloatCPU(reinterpret_cast<const float*>(output.data->data), di * dk);

  auto km = kernel_manager_t();
  auto maybe_built = km.build(einsummable);
  if (!maybe_built) {
    throw std::runtime_error("Failed to build einsummable");
  }
  auto built = maybe_built.value();
  if (!built.known()){
    throw std::runtime_error("Workspace is unknown");
  }
  auto workspace_size = km.workspace_size(einsummable).value();

  auto device = 0;

  cudaSetDevice(device);

  dbuffer_t cpu_out = reference_einsummable(einsummable, {input1, input2});

  auto gpu_input1 = gpu_allocate_memory(input1.data->size, device);
  auto gpu_input2 = gpu_allocate_memory(input2.data->size, device);
  auto gpu_output = gpu_allocate_memory(cpu_out.data->size, device);
  

  cuda_stream_t stream;
  if (cudaMemcpy(gpu_input1, input1.data->data, input1.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 1");
  }
  if (cudaMemcpy(gpu_input2, input2.data->data, input2.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 2");
  }
  if (cudaMemcpy(gpu_output, output.data->data, output.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy output");
  }

  if (workspace_size > 0){
    auto workspace_ptr = gpu_allocate_memory(workspace_size, device);
    km(einsummable, stream.stream, gpu_output, {gpu_input1, gpu_input2}, 
     std::make_tuple(workspace_ptr, workspace_size));
  }
  else{
    km(einsummable, stream.stream, gpu_output, {gpu_input1, gpu_input2});
  }
  cudaDeviceSynchronize();

  std::cout << "GPU input 1: " << std::endl;
  printFloatGPU(reinterpret_cast<const float*>(gpu_input1), di * dj);
  std::cout << "GPU input 2: " << std::endl;
  printFloatGPU(reinterpret_cast<const float*>(gpu_input2), dj * dk);
  std::cout << "GPU output: " << std::endl;
  printFloatGPU(reinterpret_cast<const float*>(gpu_output), di * dk);


  // dbuffer_t gpu_out = make_dbuffer(
  //     dtype_t::f32, std::floor(cpu_out.data->size / sizeof(float)));
  // if (cudaMemcpy(gpu_out.data->data, gpu_output, cpu_out.data->size,
  //                cudaMemcpyDeviceToHost) != cudaSuccess) {
  //   throw std::runtime_error("cudaMemcpy");
  // }

  // // compare the results
  // auto result = is_close(cpu_out, gpu_out);
  // // print messages based on the result
  // if (result) {
  //   std::cout << "Contraction test passed" << std::endl;
  // } else {
  //   std::cout << "Contraction test failed" << std::endl;
  // }

  // if (!result) {
  //   std::cout << "Expected result: " << std::endl;
  //   printFloatCPU(reinterpret_cast<const float *>(cpu_out.data->data),
  //                 std::floor(cpu_out.data->size / sizeof(float)));
  //   std::cout << "Actual result: " << std::endl;
  //   printFloatCPU(reinterpret_cast<const float *>(gpu_out.data->data),
  //                 std::floor(gpu_out.data->size / sizeof(float)));
  // }
}

