#pragma once
#include "../../base/setup.h"
#include "../../einsummable/memgraph.h"
#include "../../einsummable/reference.h"
#include "gpu_communicator.h"
#include "utility.h"
#include "cutensor.h"
#include "kernels.h"
#include "gpu_kernel_manager.h"
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <unordered_map>

using memgraph_t = memgraph_t;

// we need to define HANDLE_ERROR properly since it's not included in the header
// file defined:
// (https://docs.nvidia.com/cuda/cutensor/getting_started.html#determine-algorithm-and-workspace)
#define HANDLE_ERROR(x)                                                        \
{                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
}

struct multi_gpu_execute_state_t {
  memgraph_t const &memgraph;
  // maintain a queue of tasks that are pending to be executed
  std::queue<int> pending_queue;
  // maintain a queue of tasks that are finished
  std::queue<int> finished_queue;
  //kernel manager for this specific multi_gpu_execute_state_t
  kernel_manager_t km;

  // if we are not modifying the original memgraph, we can use a map
  // to store the dependency count of each node
  // key: node index, value: dependency count of that node
  // whenever a node finishes execution, we decrement the dependency count of
  // all its output if the dependency count of a node reaches 0, we add it to
  // the pending_queue
  std::map<int, int> dependency_count;

  // how many nodes in the memgraph are remaining to execute
  // on the same machine, there should be one counter counting all nodes executing
  // on that machine
  int num_nodes_remaining;

  // keep track of the group id of the touches that are executing
  // if we have a new touch and its group id is in the list,
  // we need to wait for the previous touch to finish
  std::vector<std::set<int>> group_id_executing;
  std::vector<std::set<int>> all_group_ids;
  // keep maps from group id to the a queue of nodes waiting 
  // for the previous touch with the same id to finish
  std::vector<std::map<int, std::queue<int>>> groupID_to_nodeIDX;

  std::vector<cutensorHandle_t*> handles;
  // create a pool of streams for each device
  std::vector<std::queue<cudaStream_t>> stream_pool;
  std::vector<std::queue<int>> node_idx_waiting_for_stream;
  std::vector<std::queue<cudaStream_t>> finished_streams;

  // pointer pointing to the start of the GPU memory
  std::vector<void*> memory_base_ptrs;

  std::unordered_map<einsummable_t, build_result_t> einsum_build_results;

  gpu_comm_t gpu_comm;

  // synchronization variables used by the callback function
  std::mutex m;
  std::condition_variable cv;

  multi_gpu_execute_state_t(memgraph_t const& input_memgraph, std::vector<void*> mem_ptrs);
  // given a memgraph, traverse the graph and get all
  // the dependencies of a node represented by node.inns()
  // return a map from the node to the number of its dependencies
  std::map<int, int> get_dependencies();

  // check if the memgraph traversal is complete
  // (are the dependency count of all nodes going to 0?)
  bool is_complete();

  // update memgraph node when it is finished; modify the dependency counter of
  // the node return a vector of ready nodes if there are any
  vector<int> node_update(int node_idx);

  // add a vector of nodes to the pending queue
  void add_to_pending_queue(std::vector<int> &nodes);

  std::vector<void const *> get_input_mem_ptrs(std::vector<mem_t> mem,
                                              void *memory_base_ptr);

  void printContractionInfo(int node_idx, int num_elems);
  void checkContractionOffset(int node_idx);

  // execute the memgraph
  void run();
};

void execute_multi_gpu(memgraph_t const &memgraph, std::vector<void*> mem_ptrs);