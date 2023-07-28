#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute.h"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sys/types.h>

// print the information of the memgraph
void print_memgraph(memgraph_t memgraph){
  // print the input and output of every node
  for (int i = 0; i < memgraph.nodes.size(); ++i) {
    std::cout << "Node " << i << " has input: ";
    for (auto in : memgraph.nodes[i].inns) {
      std::cout << in << " ";
    }
    std::cout << "and output: ";
    for (auto out : memgraph.nodes[i].outs) {
      std::cout << out << " ";
    }
    std::cout << "Node type: ";
    memgraph.nodes[i].op.print_type();
    if (memgraph.nodes[i].op.is_touch()){
      // print the group id
      std::cout << " Group id: " << memgraph.nodes[i].op.get_apply().group;
    }
    std::cout << std::endl;
  }
}

// check if the offset in mems is greater than the bound
// throw an error if it is
// also check if the offset + size is greater than the bound
void mem_t_check(std::vector<mem_t> mems, int bound) {
  for (auto mem : mems) {
    if (mem.offset > bound) {
      throw std::runtime_error("Error: offset is greater than the bound.");
    }
    if (mem.offset + mem.size > bound) {
      throw std::runtime_error("Error: offset + size is greater than the bound.");
    }
  }  
}

// check if all nodes in the memgraph are within the memory bound
void check_bounds(memgraph_t memgraph, size_t bound){
  for (auto node: memgraph.nodes){
    if (node.op.is_inputmem()){
      auto op = node.op.get_inputmem();
      if (op.offset + op.size > bound){
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
        throw std::runtime_error("Memory bound exceeded: MOVE");
      }
      if (std::get<1>(op.dst) + op.size > bound){
        throw std::runtime_error("Memory bound exceeded: MOVE");
      }
    }
    else if (node.op.is_evict()){
      memloc_t src = node.op.get_evict().src;
      if (src.offset + src.size > bound){
        throw std::runtime_error("Memory bound exceeded: EVICT");
      }
    }
    else if (node.op.is_load()){
      memloc_t dst = node.op.get_load().dst;
      if (dst.offset + dst.size > bound){
        throw std::runtime_error("Memory bound exceeded: LOAD");
      }
    }
    else if (node.op.is_partialize()){
      auto op = node.op.get_partialize();
      if (op.offset + op.size > bound){
        throw std::runtime_error("Memory bound exceeded: PARTIALIZE");
      }
    }
    else if (node.op.is_alloc()){
      auto op = node.op.get_alloc();
      if (op.offset + op.size > bound){
        throw std::runtime_error("Memory bound exceeded: ALLOC");
      }
    }
    else if (node.op.is_del()){
      auto op = node.op.get_del();
      if (op.offset + op.size > bound){
        throw std::runtime_error("Memory bound exceeded: DEL");
      }
    }
  }
}

void execute_test(memgraph_t memgraph) {

  // print a message
  std::cout << "Checking correctness" << std::endl;
  // create a buffer
  // auto num_elems =  memgraph.mem_sizes()[0] / sizeof(float);
  // dbuffer_t d = make_dbuffer(dtype_t::f32, num_elems);
  // d.random("-1.0", "1.0");
  // // d.fill(scalar_t(float(17.63)));
  // buffer_t b = d.data;
  // auto cpu_ptr = b->data;
  // auto size = b->size;

  // print the number of nodes in the graph
  std::cout << "Number of nodes in the graph: " << memgraph.nodes.size()
            << std::endl;
  bool debug = true;
  if (debug) {
    print_memgraph(memgraph);
  }

  // allocate a buffer on GPU
  auto gpu_ptr = gpu_allocate_memory(memgraph.mem_sizes()[0]);
  // print the memgraph size
  std::cout << "Memgraph size: " << memgraph.mem_sizes()[0] << std::endl;
  // copy data from CPU to GPU
  // if(cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice) !=
  // cudaSuccess) {
  //     throw std::runtime_error("cudaMemcpy");
  // }
  // execute the memgraph on the GPU ptr
  execute(memgraph, gpu_ptr);
  std::cout << "GPU execution has finished" << std::endl;
  // bring the data back
  // dbuffer_t out = make_dbuffer(dtype_t::f32, num_elems);
}

// NOTE: Since the correctness test fills the entire buffer with random values
// without any consideration on the alignment The test only works with alignment
// = 1
void check_correctness(memgraph_t memgraph, bool debug = false) {
  if (debug) {
    // print the number of nodes in the graph
    std::cout << "Number of nodes in the graph: " << memgraph.nodes.size()
              << std::endl;
    // print the input and output of every node
    for (int i = 0; i < memgraph.nodes.size(); ++i) {
      std::cout << "Node " << i << " has input: ";
      for (auto in : memgraph.nodes[i].inns) {
        std::cout << in << " ";
      }
      std::cout << "and output: ";
      for (auto out : memgraph.nodes[i].outs) {
        std::cout << out << " ";
      }
      std::cout << "Node type: ";
      memgraph.nodes[i].op.print_type();
      std::cout << std::endl;
    }
  }

  // print a message
  std::cout << "Checking correctness" << std::endl;
  // create a buffer
  auto num_elems = memgraph.mem_sizes()[0] / sizeof(float);
  dbuffer_t d = make_dbuffer(dtype_t::f32, num_elems);
  d.random("-1.0", "1.0");
  // d.fill(scalar_t(float(17.63)));
  buffer_t b = d.data;
  auto cpu_ptr = b->data;
  auto size = b->size;
  if (debug) {
    std::cout << "Original buffer: " << std::endl;
    printFloatCPU(reinterpret_cast<const float *>(cpu_ptr), num_elems);
  }
  // allocate a buffer on GPU
  auto gpu_ptr = gpu_allocate_memory(memgraph.mem_sizes()[0]);
  // copy data from CPU to GPU
  if (cudaMemcpy(gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy");
  }
  // execute the memgraph on the GPU ptr
  execute(memgraph, gpu_ptr);
  std::cout << "GPU execution has finished" << std::endl;
  // bring the data back
  dbuffer_t out = make_dbuffer(dtype_t::f32, num_elems);
  if (cudaMemcpy(out.data->data, gpu_ptr, size, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMemcpy");
  }
  std::cout << "Copying from GPU to CPU has finished" << std::endl;

  // run the same computation on CPU
  vector<buffer_t> buffers;
  buffers.push_back(b);
  reference_compute_memgraph(memgraph, buffers);
  std::cout << "CPU reference has finished" << std::endl;

  // compare the results
  auto result = is_close(d, out);
  // print messages based on the result
  if (result) {
    std::cout << "Correctness test passed" << std::endl;
  } else {
    std::cout << "Correctness test failed" << std::endl;
  }

  if (debug && !result) {
    std::cout << "Expected result: " << std::endl;
    printFloatCPU(reinterpret_cast<const float *>(cpu_ptr), num_elems);
    std::cout << "Actual result: " << std::endl;
    printFloatCPU(reinterpret_cast<const float *>(out.data->data), num_elems);
  }
}

// ij,jk->ik
/*
void contractionTest(int di, int dj, int dk) {
  auto num_elems = di * dj + dj * dk + di * dk;
  auto buffer_size = num_elems * sizeof(float);
  // create the einsummable
  auto einsummable = einsummable_t::from_matmul(di, dj, dk);
  einsummable = einsummable.merge_adjacent_dims();
  // create two input dbuffers
  dbuffer_t input1 = make_dbuffer(dtype_t::f32, di * dj);
  dbuffer_t input2 = make_dbuffer(dtype_t::f32, dj * dk);
  input1.random("-1.0", "1.0");
  input2.random("-1.0", "1.0");

  auto gpu_ptr = gpu_allocate_memory(buffer_size);
  if (cudaMemcpy(gpu_ptr, input1.data->data, input1.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 1");
  }
  if (cudaMemcpy(gpu_ptr + di * dj, input2.data->data, input2.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 2");
  }

  // print inputs
  // std::cout << "Input 1: " << std::endl;
  // printFloatCPU(reinterpret_cast<const float*>(input1.data->data), di * dj);
  // std::cout << "Input 2: " << std::endl;
  // printFloatCPU(reinterpret_cast<const float*>(input2.data->data), dj * dk);

  dbuffer_t cpu_out = reference_einsummable(einsummable, {input1, input2});

  // print GPU layout
  // std::cout << "GPU layout before execution: " << std::endl;
  // printFloatGPU(reinterpret_cast<const float*>(gpu_ptr), num_elems);

  auto gpu_input1 = gpu_allocate_memory(input1.data->size);
  auto gpu_input2 = gpu_allocate_memory(input2.data->size);
  auto gpu_output = gpu_allocate_memory(cpu_out.data->size);

  // copy data from CPU to GPU
  if (cudaMemcpy(gpu_input1, input1.data->data, input1.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 1");
  }
  if (cudaMemcpy(gpu_input2, input2.data->data, input2.data->size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy input 2");
  }

  cutensorHandle_t *handle;
  HANDLE_ERROR(cutensorCreate(&handle));
  cutensorContractionDescriptor_t desc;
  build_contraction(&desc, handle, einsummable);
  cudaStream_t stream = cuda_create_stream();
  execute_contraction(stream, handle, &desc, gpu_output, gpu_input1,
                      gpu_input2);

  // print GPU inputs and output
  // std::cout << "GPU input 1: " << std::endl;
  // printFloatGPU(reinterpret_cast<const float*>(gpu_input1), di * dj);
  // std::cout << "GPU input 2: " << std::endl;
  // printFloatGPU(reinterpret_cast<const float*>(gpu_input2), dj * dk);
  // std::cout << "GPU output: " << std::endl;
  // printFloatGPU(reinterpret_cast<const float*>(gpu_output), di * dk);

  dbuffer_t gpu_out = make_dbuffer(
      dtype_t::f32, std::floor(cpu_out.data->size / sizeof(float)));
  if (cudaMemcpy(gpu_out.data->data, gpu_output, cpu_out.data->size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy");
  }

  // compare the results
  auto result = is_close(cpu_out, gpu_out);
  // print messages based on the result
  if (result) {
    std::cout << "Contraction test passed" << std::endl;
  } else {
    std::cout << "Contraction test failed" << std::endl;
  }

  if (!result) {
    std::cout << "Expected result: " << std::endl;
    printFloatCPU(reinterpret_cast<const float *>(cpu_out.data->data),
                  std::floor(cpu_out.data->size / sizeof(float)));
    std::cout << "Actual result: " << std::endl;
    printFloatCPU(reinterpret_cast<const float *>(gpu_out.data->data),
                  std::floor(gpu_out.data->size / sizeof(float)));
  }
}

void alignmentTest(int di, int dj, int dk) {
  int num_elems = di * dj + dj * dk + di * dk;
  int total_size = num_elems * sizeof(float) * 200;
  auto my_allocator =
      allocator_t(total_size, allocator_settings_t::gpu_alignment_settings());

  // randomly initialize the inputs
  dbuffer_t input1 = make_dbuffer(dtype_t::f32, di * dj);
  dbuffer_t input2 = make_dbuffer(dtype_t::f32, dj * dk);
  dbuffer_t output = make_dbuffer(dtype_t::f32, di * dk);
  input1.random("-1.0", "1.0");
  input2.random("-1.0", "1.0");

  auto input_mem1 = my_allocator.allocate(input1.data->size);
  auto offset_input1 = std::get<0>(input_mem1);
  auto input_mem2 = my_allocator.allocate(input2.data->size);
  auto offset_input2 = std::get<0>(input_mem2);
  auto output_mem = my_allocator.allocate(output.data->size);
  // print the offsets
  std::cout << "Input 1 offset: " << std::get<0>(input_mem1) << std::endl;
  // print the offsets
  std::cout << "Input 2 offset: " << std::get<0>(input_mem2) << std::endl;
  // print the offsets
  std::cout << "Output offset: " << std::get<0>(output_mem) << std::endl;

  // cuda malloc
  auto gpu_ptr = gpu_allocate_memory(total_size);

  auto einsummable = einsummable_t::from_matmul(di, dj, dk);
  einsummable = einsummable.merge_adjacent_dims();
  cutensorHandle_t *handle;
  HANDLE_ERROR(cutensorCreate(&handle));
  cutensorContractionDescriptor_t desc;
  build_contraction(&desc, handle, einsummable);
  cudaStream_t stream = cuda_create_stream();
  execute_contraction(stream, handle, &desc,
                      offset_increment(gpu_ptr, std::get<0>(output_mem)),
                      offset_increment(gpu_ptr, std::get<0>(output_mem)),
                      offset_increment(gpu_ptr, std::get<0>(output_mem)));
}

void contractionTest2() {
  size_t my_size = 64000000;
  mem_t input1 = {.offset = 1984000000, .size = my_size};
  mem_t input2 = {.offset = 1152000000, .size = my_size};
  mem_t output = {.offset = 2304000000, .size = my_size};

  auto gpu_ptr = gpu_allocate_memory(2560000000);

  auto einsummable = einsummable_t::from_matmul(4000, 4000, 4000);
  cutensorHandle_t *handle;
  HANDLE_ERROR(cutensorCreate(&handle));
  cutensorContractionDescriptor_t desc;
  build_contraction(&desc, handle, einsummable);
  cudaStream_t stream = cuda_create_stream();
  execute_contraction(stream, handle, &desc,
                      offset_increment(gpu_ptr, output.offset),
                      offset_increment(gpu_ptr, input1.offset),
                      offset_increment(gpu_ptr, input2.offset));
}
*/
