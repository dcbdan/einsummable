#include "execute.h"
#include "kernels.h"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <optional>
#include <queue>
#include <sys/types.h>
#include <thread>
#include <vector>
#include <stdio.h>


using std::thread;
using std::queue;

cudaStream_t cuda_create_stream() {
  cudaStream_t ret;
  if(cudaStreamCreate(&ret) != cudaSuccess) {
    throw std::runtime_error("cuda_create_stream");
  }
  return ret;
}

// increment the pointer by the byte offset
// ONLY USE IF THE UNIT OF OFFSET IS BYTE
float* offset_increment(float* ptr, int offset) {
  return (float*)((char*)ptr + offset);
}

// USE THIS IF THE UNIT OF OFFSET IS FLOAT
// float* offset_increment(float* ptr, int offset) {
//     return ptr + offset;
// }

// prints float starting from ptr with count number of elements
void printFloatCPU(const float* cpu_ptr, int count) {
  for (int i = 0; i < count; ++i) {
    printf("%.2f ", cpu_ptr[i]);
  }
  printf("\n");
}

void printFloatGPU(const float* gpu_ptr, int count) {
  float* cpu_ptr = (float*)malloc(count * sizeof(float));
  cudaMemcpy(cpu_ptr, gpu_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  printFloatCPU(cpu_ptr, count);
  free(cpu_ptr);
}

void printContractionInfo(int node_idx, memgraph_t const& memgraph, float* memory_base_ptr){
    auto node = memgraph.nodes[node_idx];
    auto memory_vector = node.op.get_apply().mems;
    // print offsets
    std::cout << "Offset 1: " << memory_vector[1].offset << std::endl;
    std::cout << "Offset 2: " << memory_vector[2].offset << std::endl;
    // print inputs
    std::cout << "Input 1: ";
    printFloatGPU(offset_increment(memory_base_ptr , memory_vector[1].offset), 100);
    std::cout << "Input 2: ";
    printFloatGPU(offset_increment(memory_base_ptr , memory_vector[2].offset), 100);
    std::cout << "Output: ";
    printFloatGPU(offset_increment(memory_base_ptr , memory_vector[0].offset), 100);
}

void init_value(float* ptr, int count, float value) {
  // malloc memory on cpu and cudamemcpy to gpu
  float* tmp = (float*)malloc(count * sizeof(float));
  float* check = (float*)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    tmp[i] = value;
  }
  cudaMemcpy(ptr, tmp, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(tmp, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(check, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
}

// update memgraph node when it is finished; modify the dependency counter of the node
// return a vector of ready nodes if there are any
vector<int> node_update(std::map<int, int>& dependency_count, const memgraph_t& memgraph, 
    int node_idx, std::map<int, int>& num_nodes_remaining, std::set<int>& group_id_executing, 
    std::map<int, std::queue<int>>& groupID_to_nodeIDX) {
    // TODO: hard coded index 0 since we only have 1 device
    // print a update message
    // printf("Node %d finished execution\n", node_idx);
    num_nodes_remaining[0] -= 1;
    vector<int> ready_nodes;
    auto node = memgraph.nodes[node_idx];
    for (auto out: node.outs) {
        dependency_count[out] -= 1;
        // print the node that got decremented
        // printf("Node %d has dependencies decreased\n", out);
        if (dependency_count[out] == 0) {
            ready_nodes.push_back(out);
            // std::cout << "Adding node " << out << " to ready nodes" << std::endl;
        }
    }
    // if this node is a touch, we find if there are any other nodes 
    // in the same group that are waiting for this touch to finish
    if (node.op.is_touch()) {
        // at this point we know that the it's not the first time we see this touch's group id
        auto group_id = node.op.get_apply().group;
        // remove the group id from the executing list since we are done
        // std::cout << "Removing group id from executing list" << std::endl;
        group_id_executing.erase(group_id);
        // find if there are any other nodes in the same group that are waiting for this touch to finish
        if(groupID_to_nodeIDX[group_id].size() != 0) {
            // get one of them and add it to the ready nodes
            auto touch_node_idx = groupID_to_nodeIDX[group_id].front();
            groupID_to_nodeIDX[group_id].pop();
            // std::cout << "Adding touch node " << touch_node_idx << " to ready nodes" << std::endl;
            ready_nodes.push_back(touch_node_idx);
        }
    }
    return ready_nodes;
}

// add a vector of nodes to the queue
void add_to_queue(std::queue<int>& queue, std::vector<int>& nodes) {
    for (auto node: nodes) {
        queue.push(node);
    }
    // print how many elements are in the queue
    // std::cout << "Queue size: " << queue.size() << std::endl;
}

// get the dependency count of each node
std::map<int, int> get_dependencies(const memgraph_t& memgraph) {
    std::map<int, int> dependency_count;
    for (int i = 0; i < memgraph.nodes.size(); i++) {
        dependency_count[i] = memgraph.nodes[i].inns.size();
    }
    return dependency_count;
}

// check if all the nodes finished executing
bool is_complete(std::map<int, int>& dependency_count) {
    auto idx = 0;
    for (auto it = dependency_count.begin(); it != dependency_count.end(); it++) {
        if (it->second != 0) {
            return false;
        }
        idx++;
    }
    return true;
}

// calling cuda malloc to allocate memory for a given size
float* gpu_allocate_memory(size_t size) {
  void* ret;
  if(cudaMalloc(&ret, size) != cudaSuccess) {
    throw std::runtime_error("cuda_malloc");
  }
  return (float*)ret;
}

// helper function to get the input memory pointers from a vector of mem_t
// input memory pointers are mem[1: n]
std::vector<float const*> get_input_mem_ptrs(std::vector<mem_t> mem, float *memory_base_ptr) {
  std::vector<float const*> ret;
    for (int i = 1; i < mem.size(); i++) {
        ret.push_back(offset_increment(memory_base_ptr, mem[i].offset));
    }
  return ret;
}

// get a callback data struct that keeps track of the current node that finished execution
// Has all the data structures required to update things
struct callback_data_t {
  std::mutex* m_ptr;
  std::condition_variable* cv_ptr;
  std::map<int, int>* dependency_count;
  memgraph_t const* memgraph; 
  std::queue<int>* pending_queue;
  int node_idx;
  std::map<int, int>* num_nodes_remaining;
  std::set<int>* group_id_executing;
  std::map<int, std::queue<int>>* groupID_to_nodeIDX;
  float* mem_ptr;

  void operator()() {
    std::mutex& m = *m_ptr;
    auto& cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      auto node = (*memgraph).nodes[node_idx];
    //   if (node.op.is_contraction()){
    //     printContractionInfo(node_idx, *memgraph, mem_ptr);
    //   }
      // update the queue since this node is finished
      auto new_nodes = node_update(*dependency_count, *memgraph, node_idx, 
      *num_nodes_remaining, *group_id_executing, *groupID_to_nodeIDX);
      add_to_queue(*pending_queue, new_nodes);      
    }
    cv.notify_all();
  }
};

void execute(const memgraph_t& memgraph, float* memory_base_ptr) {
    // create a gpu_execute_state_t
    gpu_execute_state_t gpu_execute_state(memgraph, memory_base_ptr);
    gpu_execute_state.run();
}

// function definition of gpu_execute_state_t.run()
void gpu_execute_state_t::run() {

    while (true) {
        // if the num_nodes_remaining is 0, then we are done
        if (is_complete(num_nodes_remaining)) {
            // if the dependency count for all node is 0, then we are done
            // else throw an error
            if (!is_complete(dependency_count)) {
                throw std::runtime_error
                    ("Error: All nodes finished execution but the dependency count doesn't match.");
            }
            else if (pending_queue.size() != 0) {
                throw std::runtime_error
                    ("Error: All nodes finished execution but there are still nodes in the queue.");
            }
            else{
                std::cout << "All nodes finished execution." << std::endl;
                break;
            }
            throw std::runtime_error("Error: Undefined end reached.");
        }
        // locking the mutex until the queue has new things to execute
        {
            std::unique_lock lk(m);
            cv.wait(lk, [&]{
                return pending_queue.size() > 0;
            });
        }
        // execute things that are in the apply_queue until the queue is empty
        while (pending_queue.size() != 0) {
            // print out the pending queue
            // get the first element in the queue
            auto node_idx = pending_queue.front();
            auto node = memgraph.nodes[node_idx];
            // remove the first element from the queue
            pending_queue.pop();
            // execute the node
            if (node.op.is_input() || node.op.is_del() || node.op.is_partialize()) {
                std::unique_lock lk(m);
                // do nothing but update the memgraph execution since that node is finished
                auto new_nodes = node_update(dependency_count, memgraph, node_idx, num_nodes_remaining, 
                    group_id_executing, groupID_to_nodeIDX);
                add_to_queue(pending_queue, new_nodes);
                lk.unlock();
            }
            else if (node.op.is_apply()) {
                // create a cuda stream since for apply we need to execute that on a cuda stream always
                // TODO: may need to keep a pool of streams
                cudaStream_t stream = cuda_create_stream();
                // get the memory offsets
                auto memory_vector = node.op.get_apply().mems;
                // CASE: TOUCH
                if (node.op.is_touch()) {
                    // std::cout << "Got a touch node" << std::endl;
                    auto touch = node.op.get_touch();
                    auto group_id = node.op.get_apply().group;
                    // if we have found this group id in the list, we can't execute until the previous one is done
                    if (group_id_executing.count(group_id) != 0) {
                        // std::cout << "Found a touch node " << node_idx << " with group id " << group_id 
                        //     << " that is already executing." << std::endl;
                        // we can't execute this node since some other node is executing with the same group id
                        // add this node to the map 
                        groupID_to_nodeIDX[group_id].push(node_idx);
                        // skipping the callback since this node didn't execute
                        continue;
                    }
                    else{
                        // else we are free to run this
                        // see this is the first time seeing this group id
                        if (all_group_ids.count(group_id) == 0){
                            // set the castable to nullopt
                            touch.castable = std::nullopt;
                        }
                        else if (group_id < 0){
                            // set the castable to nullopt
                            touch.castable = std::nullopt;
                        }
                        else{
                            if (touch.castable == std::nullopt){
                                throw std::runtime_error("Error: Castable is not set for a touch node.");
                            }
                        }
                        // add this group id to the executing set
                        group_id_executing.insert(group_id);
                        all_group_ids.insert(group_id);
                        auto touch_kernel = build_touch(touch);
                        touch_kernel(stream, offset_increment(memory_base_ptr, memory_vector[0].offset), 
                        offset_increment(memory_base_ptr, memory_vector[1].offset));
                    }
                }
                else {
                    auto my_einsummable = node.op.get_einsummable();
                    // CASE: CONTRACTION
                    if (my_einsummable.is_contraction()) {
                        // merge the adjacent dims
                        // std::cout << "Got a contraction node" << std::endl;
                        // printContractionInfo(node_idx, memgraph, memory_base_ptr);
                        einsummable_t my_einsum_merged = my_einsummable.merge_adjacent_dims();
                        // print an error if we didn't find my_einsum_merged in the map
                        auto einsum_iter = einsum_to_contraction.find(my_einsum_merged);
                        if (einsum_iter == einsum_to_contraction.end()) {
                            throw std::runtime_error
                                ("Error: contraction descriptor not found in the map of contraction plans.");
                        }
                        
                        auto contraction_descriptor = einsum_iter->second;
                        execute_contraction(stream, handle,& contraction_descriptor, 
                            offset_increment(memory_base_ptr, memory_vector[0].offset),
                            offset_increment(memory_base_ptr, memory_vector[1].offset), 
                            offset_increment(memory_base_ptr, memory_vector[2].offset));
                    }
                    // CASE: OTHER EINSUMMABLE
                    else {
                        // std::cout << "Got a other einsummable node" << std::endl;
                        auto cutensor_kernel = build_einsummable(my_einsummable);
                        cutensor_kernel(stream, handle, offset_increment(memory_base_ptr, memory_vector[0].offset), 
                            get_input_mem_ptrs(memory_vector, memory_base_ptr));
                    }
                }

                // after execution, we attach the stream with a callback function
                // get all the metadata needed for the callback
                callback_data_t* data = new callback_data_t;
                data->m_ptr =& m;
                data->cv_ptr =& cv;
                data->node_idx = node_idx;
                data->dependency_count =& dependency_count;
                data->memgraph =& memgraph;
                data->pending_queue =& pending_queue;
                data->num_nodes_remaining =& num_nodes_remaining;
                data->group_id_executing =& group_id_executing;
                data->groupID_to_nodeIDX =& groupID_to_nodeIDX;
                data->mem_ptr = memory_base_ptr;
                
                // add the callback
                cudaStreamAddCallback(
                    stream,
                    [](CUstream_st*, cudaError, void* raw_data) {
                        callback_data_t* data = static_cast<callback_data_t*>(raw_data);
                        callback_data_t& f = *data;
                        f();
                        delete data;
                    },
                    static_cast<void*>(data),
                    0
                );
            }
            else{
                // print a message saying that the operation is not supported and this operation's type
                throw std::runtime_error
                    ("Error: Operation not supported: Type is among the following - move, evict, load");
            }
        }
        
    }
}