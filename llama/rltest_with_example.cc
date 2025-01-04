#include "../llama/modules.h"
#include "../src/base/args.h"
#include "../src/base/copyregion.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/autoplace/apart.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/autoplace/alocate.h"
#include "../src/base/placement.h"
#include "../src/engine/communicator.h"
#include "../src/engine/resource_manager.h"
#include "../src/server/gpu/server.h"

#include <fstream>
#include <sys/types.h>  
#include <stdio.h>   
#include <stdlib.h>   
#include <string.h>   
#include <unistd.h>   
#include <sys/wait.h>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <filesystem>


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

using tensor_t = graph_writer_t::tensor_t;

graph_t full_graph(int argc, char** argv) {
  args_t pargs(argc, argv);

  model_args_t margs = model_args_t::llama(1, 1);

  pargs.set_default<int>("max_n_layers", 1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    DLINEOUT("n_layers " << n_layers);
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("batch_size", 1);
  margs.batch_size = pargs.get<uint64_t>("batch_size");

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  tensor_t predictions = model.forward(embeddings);
  predictions.save_inplace();

  graph_t const& graph = writer.get_graph();
  return graph;
}

graph_t one_block(int argc, char** argv) {
  args_t pargs(argc, argv);

  model_args_t margs = model_args_t::llama(1, 1);

  pargs.set_default<uint64_t>("batch_size", 1);
  margs.batch_size = pargs.get<uint64_t>("batch_size");

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  graph_writer_t writer;
  transformer_block_t block(&writer, 0, margs, 0, std::nullopt);

  tensor_t x = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  tensor_t freqs_cis = writer.input(
    { margs.max_seq_len, uint64_div(margs.head_dim(), 2) },
    dtype_t::c64);

  tensor_t y = block.forward(x, freqs_cis, std::nullopt);
  y.save_inplace();

  graph_t const& graph = writer.get_graph();
  return graph;
}





  
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


vector<placement_t> create_same_loc_placements(vector<partition_t> const& parts, int num_locs)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    // std::cout <<  "part: " << part  <<  std::endl;
    // for(int i = 0; i <  part.num_parts(); i++){
    //   std::cout << part.total_shape()  <<  std::endl;
    // }
    
    ret.emplace_back(part);
  }
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


int main(int argc, char** argv) {
  
  // set default type for all tensors
  set_default_dtype(dtype_t::f16);

  // specify RL parameters
  int total_num_episodes = 3000;
  float running_time = 0.0;
  
  // specify einsumable parameters
  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;
  int num_gpus = 4;

  // specify storage
  uint64_t mem_size = 14lu * 1000lu * 1000lu * 1000lu;
  int num_partition = 4;

  // create graph
  // graph_t const& graph = one_block(argc, argv); 
  graph_t const& graph = full_graph(argc, argv);
  vector<partition_t> parts = apart01(graph,num_partition);

  // collect_feat_for_rl(graph, parts);

  // create a communicator
  communicator_t c(addr_zero, is_rank_zero, world_size);
  

  // create server
  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i){
      buffer_sizes.push_back(mem_size);
  }
  auto storage_size = 1 * 1000lu * 1000lu * 1000lu;
  bool use_cudagraph = false;

  gpu_mg_server_t server = storage_size > 0                                  ?
    gpu_mg_server_t(c, use_cudagraph, buffer_sizes, storage_size) :
    gpu_mg_server_t(c, use_cudagraph, buffer_sizes)               ;

  server.set_split_off_inputs(true);

  for(int i = 0; i < total_num_episodes; i++){
    
    std::vector<int> rl_placements = {0, 0, 0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0,
        1, 2, 3, 0, 1, 2, 3, 3, 1, 2, 1, 0, 2, 2, 1, 0, 2, 3, 1, 0, 3, 1, 3, 1,
        2, 3, 0, 3, 3, 2, 1, 0, 2, 2, 1, 0, 2, 1, 3, 1, 0, 1, 3, 1, 0, 1, 3, 1,
        0, 1, 3, 1, 3, 2, 3, 2, 0, 2, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3,
        2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3,
        2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3,
        1, 3, 0, 3, 0, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 0, 0, 3, 1, 0, 0, 3, 1, 3, 1, 3,
        1, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 0, 3, 0, 3,
        0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<uint64_t> rl_priority = {5,   9,   6,   3,   4,  10,   7,   8,  11,  12,  13,  14,  16,  15,
         17,  18,  19,  20,  21,  22,  23,  24,  25,  29,  26,  30,  27,  28,
         33,  34,  31,  41,  43,  32,  42,  44,  35,  36,  46,  45,  37,  38,
         54,  58,  53,  57,  39,  55,  59,  74,  76,  40,  73,  75,  56,  60,
         50,  49,  47,  48,   1,   0,  51,  52,   2,  64,  68,  72,  63,  67,
         71,  77,  79,  81,  84,  78,  62,  66,  70,  61,  65,  69,  83,  88,
         87,  91,  92,  95,  99,  96, 100, 103, 104,  80,  82,  85,  86,  90,
         89,  94,  93,  98, 102,  97, 101, 106, 105, 110, 114, 118, 122, 108,
        112, 116, 109, 113, 117, 120, 121, 126, 130, 134, 138, 142, 146, 107,
        111, 115, 119, 124, 128, 123, 127, 132, 125, 129, 133, 136, 137, 131,
        135, 140, 145, 139, 143, 141, 144, 148, 150, 152, 154, 156, 158, 160,
        162, 164, 166, 168, 172, 184, 171, 183, 188, 187, 175, 176, 180, 179,
        191, 195, 192, 196, 147, 149, 151, 153, 155, 198, 200, 202, 204, 206,
        157, 159, 161, 163, 165, 167, 169, 170, 208, 210, 212, 182, 214, 216,
        181, 173, 174, 185, 186, 178, 177, 189, 193, 190, 194, 218, 219, 220,
        223, 221, 222, 224, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215,
        217, 226, 228, 227, 225};
    std::vector<int> input_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

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
    std::cout << "rl prio size: " << rl_priority.size() << std::endl;
    server.execute_graph(graph, placements, rl_priority); // Your function here
    // inFile >> running_time;
    // std::cout << "Running time: " << running_time << " miliseconds " << std::endl;
  }
  server.shutdown();
    
  return 0;
}