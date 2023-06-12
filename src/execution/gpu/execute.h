#pragma once
#include "../../base/setup.h"
#include "../../einsummable/memgraph.h"
#include "../../einsummable/reference.h" // buffer_t
#include "cutensor.h"
#include "dummy_kernels.h"
#include "kernels.h"

#include <cstddef>
#include <map>
#include <thread>
#include <cuda_runtime.h>
#include <variant>
#include <mutex>
#include <condition_variable>
#include <vector>

using memgraph_t = memgraph_t;

// we need to define HANDLE_ERROR properly since it's not included in the header file 
// defined: (https://docs.nvidia.com/cuda/cutensor/getting_started.html#determine-algorithm-and-workspace)
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

// given a memgraph, traverse the graph and get all the dependencies of a node represented by node.inns()
// return a map from the node to the number of its dependencies
std::map<int, int> get_dependencies(const memgraph_t &memgraph);

// check if the memgraph traversal is complete
// (are the dependency count of all nodes going to 0?)
bool is_complete(std::map<int, int> &dependency_count);

// return a memory pointer that is allocated on the gpu
// uses cudaMalloc
float* gpu_allocate_memory(size_t size);

struct gpu_execute_state_t 
{
    const memgraph_t &memgraph;
    // buffer_t gpu_memory;
    // maintain a queue of tasks that are pending to be executed
    std::queue<int> pending_queue;

    // if we are not modifying the original memgraph, we can use a map to store the dependency count of each node
    // key: node index, value: dependency count of that node
    // whenever a node finishes execution, we decrement the dependency count of all its output
    // if the dependency count of a node reaches 0, we add it to the pending_queue
    std::map<int, int> dependency_count;

    // how many nodes in the memgraph are remaining to execute
    // each entry in the map is the number of nodes remaining for this device
    // for now there's only one device, so we only need one entry
    std::map<int, int> num_nodes_remaining;

    std::vector<int> group_id_executing;

    std::map<int, std::vector<int>> groupID_to_nodeIDX;

    cutensorHandle_t* handle;

    float* memory_base_ptr;

    std::unordered_map<einsummable_t, cutensorContractionDescriptor_t> einsum_to_contraction;

    // synchronization variables used by the callback function
    std::mutex m;
    std::condition_variable cv;
    
    // a map from the node of the memgraph to the cutensor plan 
    // we only have plans for operation contraction, so we only need to store the plans for those nodes
    // remember to throw an error if the node is not a contraction but trying to access the map
    // map<int, cutensorContractionPlan_t> cutensor_plans;

    gpu_execute_state_t(const memgraph_t &input_memgraph): memgraph(input_memgraph) {

        // create a cutensor handle
        HANDLE_ERROR( cutensorCreate(&handle) );

        // get the size of the memory needed from the memgraph
        // TODO: hardcoded the 0 since we only have 1 gpu for now
        auto mem_size = memgraph.mem_sizes()[0];
        // allocate memory for the gpu
        memory_base_ptr = gpu_allocate_memory(mem_size);

        dependency_count = get_dependencies(memgraph);

        // in the beginning num_nodes_remaining is the number of nodes in the memgraph
        num_nodes_remaining[0] = memgraph.nodes.size();

        // check all elements from the memgraph and add the nodes with no dependencies to the apply_queue
        for (int i = 0; i < memgraph.nodes.size(); i++)
        {
            if (memgraph.nodes[i].inns.size() == 0)
            {
                pending_queue.push(i);
            }
            // check if the node is a contraction
            if (memgraph.nodes[i].op.is_einsummable())
            {
                // get the einsummable object
                auto my_einsummable = memgraph.nodes[i].op.get_einsummable();
                // check is this a contraction
                if (my_einsummable.is_contraction())
                {
                    // merge the adjacent dims
                    einsummable_t my_einsum_merged = my_einsummable.merge_adjacent_dims();
                    // check if the contraction is already in the map
                    if (einsum_to_contraction.find(my_einsum_merged) == einsum_to_contraction.end())
                    {
                        // create a cutensor descriptor
                        cutensorContractionDescriptor_t desc;
                        // when building the contraction we already merge the adjacent dims so we don't need to do it here
                        build_contraction(&desc, handle, my_einsum_merged);
                        // add the contraction to the map
                        einsum_to_contraction[my_einsum_merged] = desc;
                    }
                }
            }
        }
         // print the size of pending_queue in the beginning
        std::cout << "Beginning pending_queue size: " << pending_queue.size() << std::endl;
    }
    void run();
    
};

void execute(const memgraph_t &memgraph);
