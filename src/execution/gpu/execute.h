#pragma once
#include "../../base/setup.h"
#include "../../einsummable/memgraph.h"
#include "../../einsummable/reference.h"
#include "cutensor.h"
#include "kernels.h"
#include "gpu_kernel_manager.h"
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

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

void test();

// return a memory pointer that is allocated on the gpu
// uses cudaMalloc
void *gpu_allocate_memory(size_t size);

// debugging; set all the values in the memory to a specific value
void init_value(float *ptr, int count, float value);

// all debugging print functions

void printFloatCPU(const float *cpu_ptr, int count);
void printFloatGPU(const float *gpu_ptr, int count);
void *offset_increment(const void *ptr, int offset);
float *float_increment(float *ptr, int offset);

cudaStream_t cuda_create_stream();

struct gpu_execute_state_t {
  memgraph_t const &memgraph;
  // maintain a queue of tasks that are pending to be executed
  std::queue<int> pending_queue;
  // maintain a queue of tasks that are finished
  std::queue<int> finished_queue;
  //kernel manager for this specific gpu_execute_state_t
  kernel_manager_t km;

  // if we are not modifying the original memgraph, we can use a map
  // to store the dependency count of each node
  // key: node index, value: dependency count of that node
  // whenever a node finishes execution, we decrement the dependency count of
  // all its output if the dependency count of a node reaches 0, we add it to
  // the pending_queue
  std::map<int, int> dependency_count;

  // how many nodes in the memgraph are remaining to execute
  // each entry in the map is the number of nodes remaining for this device
  // for now there's only one device, so we only need one entry
  int num_nodes_remaining;

  // keep track of the group id of the touches that are executing
  // if we have a new touch and its group id is in the list,
  // we need to wait for the previous touch to finish
  std::set<int> group_id_executing;
  std::set<int> all_group_ids;
  // keep a map from group id to the a queue of nodes waiting
  // for the previous touch with the same id to finish
  std::map<int, std::queue<int>> groupID_to_nodeIDX;

  cutensorHandle_t *handle;
  // create a pool of streams, every time we need a stream we pop one from the
  // pool
  std::queue<cudaStream_t> stream_pool;
  std::queue<int> node_idx_waiting_for_stream;
  std::queue<cudaStream_t> finished_streams;

  // pointer pointing to the start of the GPU memory
  void *memory_base_ptr;

  std::unordered_map<einsummable_t, cutensorContractionDescriptor_t>
      einsum_to_contraction;

  std::unordered_map<einsummable_t,build_result_t> 
      einsum_build_results;

  // synchronization variables used by the callback function
  std::mutex m;
  std::condition_variable cv;

  // a map from the node of the memgraph to the cutensor plan
  // we only have plans for operation contraction, so we only need to store the
  // plans for those nodes remember to throw an error if the node is not a
  // contraction but trying to access the map map<int,
  // cutensorContractionPlan_t> cutensor_plans;

  gpu_execute_state_t(memgraph_t const &input_memgraph, void *mem_ptr);
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

  void printContractionInfo(int node_idx, int num_elems);
  void checkContractionOffset(int node_idx);

  // execute the memgraph
  void run();
};

void execute(memgraph_t const &memgraph, void *memory_base_ptr);