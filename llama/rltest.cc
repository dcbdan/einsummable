#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"
#include "../src/engine/gpu/workspace.h"
#include "../src/server/gpu/server.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/engine/communicator.h"
#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"
#include "../src/einsummable/gwriter.h"

#include "../src/server/base.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cutensor.h>
#include <cuda_runtime.h>

#include "../src/base/setup.h"
#include "gpu_kernel_manager.h"
#include "utility.h"
#include "../src/einsummable/reference.h"

#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <tuple>
#include <vector>

#include "../src/base/args.h"
#include "../src/base/copyregion.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/autoplace/apart.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/autoplace/alocate.h"
#include "../src/base/placement.h"
#include <fstream>
#include <sys/types.h>  
#include <stdio.h>   
#include <stdlib.h>   
#include <string.h>   
#include <unistd.h>   
#include <sys/wait.h>

// #include "../exps/GPU_correctness.cc"
#include "gpu_kernel_manager.h"
#include "utility.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <sys/types.h>
#include "../../src/engine/exec_graph.h"

#define STDIN_FILENO    0       /* Standard input.  */ 
#define STDOUT_FILENO   1       /* Standard output.  */ 
#define STDERR_FILENO   2       /* Standard error output.  */ 
#define MAXLINE 4096 

graph_t generate_ffnn(uint64_t batch, vector<uint64_t> dims){
  graph_writer_t writer;

  using tensor_t = graph_writer_t::tensor_t;

  tensor_t x = writer.input({batch, dims[0]});

  tensor_t out = x;

  for (auto dim = 1; dim < dims.size(); ++dim){
    out = writer.matmul(out, writer.input({dims[dim-1], dims[dim]}));
    printf("out shape: %lu %lu\n", dims[dim-1], dims[dim]);
    if (dim != dims.size() - 1){
      out = writer.ew(scalarop_t::make_relu(), out);
      printf("Relu\n");
    }
    else{
      printf("Softmax\n");
      out = writer.softmax_v1(out);
    }
  }
  out.save_inplace();

  auto graph = writer.get_graph();

  // print the graph
  DOUT("Printing graphviz for ffnn...");
  std::ofstream f("ffnn.gv");
  graph.print_graphviz(f);

  return graph;
}

using tensor_t = graph_writer_t::tensor_t;
  
vector<placement_t> pre_assign_loc(vector<partition_t> const& parts, vector<int> input_list, vector<int> placement)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  int i = 0;
  int rl_idx = 0;
  std::cout << "Assignments: {";
  for(auto const& part: parts) {
    ret.emplace_back(part);
    for(int& loc: ret.back().locations.get()) {
      auto it = std::find(input_list.begin(), input_list.end(), i);
      if (it != input_list.end()){
        loc = 0;
      }
      else{
        loc = placement[rl_idx];
        rl_idx += 1;
      }
      i+=1;
      std::cout << loc << ",";
    }
  }
  std::cout << "}" << std::endl;
//   std::cout << "Total number of nodes: " << i << std::endl;
  return ret;
}

buffer_t make_out(vector<uint64_t> const& shape) {
  dbuffer_t dbuffer = make_dbuffer(dtype_t::f32, product(shape));
  return dbuffer.data;
}

buffer_t make_data(vector<uint64_t> const& shape) {
  buffer_t ret = make_out(shape);
  dbuffer_t(dtype_t::f32, ret).random("-0.00001", "0.00001");
  return ret;
}


void execute_einsummable(int num_gpus, uint64_t mem_size, string addr_zero, bool is_rank_zero, int world_size, std::vector<int> rl_placements, int episode, graph_t graph, vector<partition_t> parts, vector<int> input_list){

  // create a communicator
  communicator_t c(addr_zero, is_rank_zero, world_size);

  // create buffer sizes
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i){
      buffer_sizes.push_back(mem_size);
  }

  // create server

    auto storage_size = 1 * 1000lu * 1000lu * 1000lu;
    bool use_cudagraph = false;

  gpu_mg_server_t server = storage_size > 0                                  ?
    gpu_mg_server_t(c, use_cudagraph, buffer_sizes, storage_size) :
    gpu_mg_server_t(c, use_cudagraph, buffer_sizes)               ;

  server.set_split_off_inputs(true);

  // deploy rl placements
  std::vector<placement_t> placements = pre_assign_loc(parts, input_list, rl_placements);
  
  // initialize input tensors and distribute across the cluster
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& input = node.op.get_input();
      dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
      tensor.random("-0.01", "0.01");
      server.insert_tensor(gid, placements[gid], tensor);
    }
  }
  std::cout << "start execute graph.\n" << std::endl;

  vector<uint64_t> priority(0, 100000);
  // // execute graph with my placements
  server.execute_graph(graph, placements, priority);
  server.shutdown();
}

std::vector<int> stringToIntVector(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            result.push_back(std::stoi(item));
        } catch (const std::invalid_argument& e) {
            // Handle the case where stoi could not convert item to an integer
            std::cerr << "Invalid argument: " << item << std::endl;
        } catch (const std::out_of_range& e) {
            // Handle the case where the converted integer is out of int's range
            std::cerr << "Out of range: " << item << std::endl;
        }
    }
    return result;
}

int main_rank_zero(int num_gpus, uint64_t mem_size, string addr_zero, bool is_rank_zero, int world_size, graph_t graph, vector<partition_t> parts, int num_episodes, int episode_id, float* running_time)
{
  std::vector<int> rl_placements = {0, 1, 2, 3, 3, 2, 1, 3, 2, 0, 1, 0, 0, 3, 1, 2, 0, 1, 2, 3, 3, 2, 1, 3, 2, 0, 1, 0, 0, 3, 1, 2, 0, 1, 2, 3, 3, 2, 1, 3, 2, 0, 1, 0, 0, 3, 1, 2, 0, 1, 2, 3, 3, 2, 1, 3, 2, 0, 1, 0, 0, 3, 1, 2, 1, 2, 0, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3};

  std::vector<int> input_list = {0, 1, 2, 3, 4, 5, 6, 7, 56, 57, 58, 59};
  // start execute einsummable
  std::cout << "start executing einsummable.\n" << std::endl;
  execute_einsummable(num_gpus, mem_size, addr_zero, is_rank_zero, world_size, rl_placements, episode_id, graph, parts, input_list);
  std::cout << "finish executing einsummable.\n" << std::endl;

  // read running time from text file
  std::ifstream inFile("running_time.txt");
  inFile >> *running_time;
  // std::cout << "Assignment: " << rl_placements << std::endl;
  std::cout << "Running time: " << *running_time << " miliseconds " << std::endl;

  return 0;
}

int main(int argc, char** argv) {

  // set default type for all tensors
  set_default_dtype(dtype_t::f16);

  // specify parameters
  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;
  int num_gpus = 4;

  // specify storage
  uint64_t mem_size = 1lu * 1000lu * 1000lu * 1000lu;
  uint64_t batch_size = pow(2,15);
  // H_1 and H_2 are different hidden dimensions of the Hidden layer (1 hidden layer only)
  uint64_t H_1 = 1 << 5;
  uint64_t H_2 = 1 << 16;
  uint64_t output_class = 1 << 5;
  uint64_t input_dim = 1 << 5;

  uint64_t H_test = 100;
  uint64_t output_test = 10;
  uint64_t input_dim_test = 10;
  int num_partition = 16;

  // specify RL parameters
  int num_episodes = 15; // Total number of episodes
  float running_time = 0.0;

  vector<uint64_t> dims = {input_dim, H_2, output_class}; 
  graph_t const& graph = generate_ffnn(batch_size, dims); 

  //partition the graph for location assignment
  vector<partition_t> parts = apart01(graph,num_partition);

  for(int r = 0; r < num_episodes; r++){
    main_rank_zero(num_gpus, mem_size, addr_zero, is_rank_zero, world_size, graph, parts, num_episodes, r, &running_time);
  }
    
  return 0;
}