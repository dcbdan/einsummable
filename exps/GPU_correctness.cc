#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"
#include "../src/engine/gpu/workspace.h"
#include "../src/server/gpu/server.h"

#include "../src/server/base.h"

#include <cstdint>
#include <cutensor.h>
#include <cuda_runtime.h>

#include "../src/base/setup.h"

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

  auto num_gpu = memgraph.mem_sizes().size();
  // allocate ptrs for gpu
  std::vector<void*> gpu_ptrs;
  auto mem_sizes = memgraph.mem_sizes();
  for (int i = 0; i < num_gpu; ++i){
    gpu_ptrs.push_back(gpu_allocate_memory(mem_sizes[i], i));
  }

  kernel_manager_t km;

  DOUT("Making exec graph...");
  exec_graph_t graph =
    exec_graph_t::make_gpu_exec_graph(memgraph, 0, km, num_gpus_per_node);
  DOUT("Finished making exec graph...");

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


void server_execute(memgraph_t memgraph, bool debug, vector<uint64_t>buffer_sizes){
  // auto num_gpu = memgraph.mem_sizes().size();
  if (debug){
    print_memgraph(memgraph);
  }

  communicator_t c("0.0.0.0", true, 1);
  gpu_mg_server_t server(c, buffer_sizes);

  // create a map for local insert tensors
  map<int, tuple<int, buffer_t>> data;

  for(int gid = 0; gid != memgraph.nodes.size(); ++gid) {
    auto const& node = memgraph.nodes[gid];
    if(node.op.is_inputmem()) {
      auto const& input_mem = node.op.get_inputmem();
      // how do we know d_type of the input?
      dbuffer_t tensor = make_dbuffer();
      // tensor.random("-0.01", "0.01");
      tensor.ones();
      data[gid] = std::make_tuple(node.op.get_loc(), tensor);
    }
  }

  server.local_insert_tensors(data);

  server.execute_memgraph(memgraph, false);

  //// get the outputs to here
  // for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //   auto const& node = graph.nodes[gid];
  //   if(node.op.is_save()) {
  //     dbuffer_t tensor = server.get_tensor_from_gid(gid);
  //     //DOUT(tensor);
  //     //DOUT("gid sum is: " << tensor.sum());
  //   }
  // }


  server.shutdown();


}