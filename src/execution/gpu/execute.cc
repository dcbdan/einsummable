#include "execute.h"
#include "kernels.h"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
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

//   printFloats(tmp, count);
//   printFloats(check, count);
}

// increment the pointer by the byte offset
// NO LONGER NEEDED SINCE EVERYTHING IS IN FLOATS, but it's a good reference
// float* offset_increment(float* ptr, int offset) {
//   return (float*)((char*)ptr + offset);
// }

// update memgraph node when it is finished; modify the dependency counter of the node
// return a vector of ready nodes if there are any
vector<int> node_update(std::map<int, int> &dependency_count, const memgraph_t &memgraph, int node_idx, std::map<int, int> &num_nodes_remaining) {
    // TODO: hard coded index 0 since we only have 1 device
    num_nodes_remaining[0] -= 1;
    vector<int> ready_nodes;
    for (auto out: memgraph.nodes[node_idx].outs){
        dependency_count[out] -= 1;
        // print the node that got decremented
        // printf("Node %d has dependencies decreased\n", out);
        if (dependency_count[out] == 0){
            ready_nodes.push_back(out);
        }
    }
    // print a update message
    printf("Node %d finished execution\n", node_idx);
    // print the number of nodes remaining
    // printf("Number of nodes remaining: %d\n", num_nodes_remaining[0]);
    return ready_nodes;
}

// add a vector of nodes to the queue
void add_to_queue(std::queue<int> &queue, std::vector<int> &nodes) {
    for (auto node: nodes){
        queue.push(node);
    }
    // print how many elements are in the queue
    // std::cout << "Queue size: " << queue.size() << std::endl;
}

// get the dependency count of each node
std::map<int, int> get_dependencies(const memgraph_t &memgraph) {
    std::map<int, int> dependency_count;
    for (int i = 0; i < memgraph.nodes.size(); i++){
        dependency_count[i] = memgraph.nodes[i].inns.size();
    }
    return dependency_count;
}

// checj if all the nodes are finished executing
bool is_complete(std::map<int, int> &dependency_count) {
    auto idx = 0;
    for (auto it = dependency_count.begin(); it != dependency_count.end(); it++){
        if (it->second != 0){
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
        ret.push_back(memory_base_ptr + mem[i].offset);
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

  void operator()() {
    std::mutex& m = *m_ptr;
    auto& cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      // update the queue since this node is finished
      auto new_nodes = node_update(*dependency_count, *memgraph, node_idx, *num_nodes_remaining);
      add_to_queue(*pending_queue, new_nodes);      
    }
    cv.notify_all();
  }
};

void execute(const memgraph_t &memgraph){
    // create a gpu_execute_state_t
    gpu_execute_state_t gpu_execute_state(memgraph);
    gpu_execute_state.run();
}

// function definition of gpu_execute_state_t.run()
void gpu_execute_state_t::run() {

    while (true){
        // if the num_nodes_remaining is 0, then we are done
        if (is_complete(num_nodes_remaining)){
            // if the dependency count for all node is 0, then we are done
            // else throw an error
            if (!is_complete(dependency_count)){
                std::cout << "Error: All nodes finished execution but the dependency count doesn't match." << std::endl;
                exit(1);
            }
            else if (pending_queue.size() != 0) {
                std::cout << "Error: All nodes finished execution but there are still nodes in the queue." << std::endl;
                exit(1);
            }
            else{
                int num_elements = 100;
                std::cout << "Input 1: ";
                printFloatGPU(memory_base_ptr, num_elements);
                std::cout << "Input 2: ";
                printFloatGPU(memory_base_ptr + 100, num_elements);
                std::cout << "Output: ";
                printFloatGPU(memory_base_ptr + 200, num_elements);
                std::cout << "All nodes finished execution." << std::endl;
                exit(0);
            }
            exit(0);
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
            // get the first element in the queue
            auto node_idx = pending_queue.front();
            auto node = memgraph.nodes[node_idx];
            // remove the first element from the queue
            pending_queue.pop();
            // execute the node
            if (node.op.is_input() || node.op.is_del() || node.op.is_partialize()) {
                std::unique_lock lk(m);
                // do nothing but update the memgraph execution since that node is finished
                auto new_nodes = node_update(dependency_count, memgraph, node_idx, num_nodes_remaining);
                add_to_queue(pending_queue, new_nodes);
                lk.unlock();
            }
            else if (node.op.is_apply()){
                // create a cuda stream since for apply we need to execute that on a cuda stream always
                // TODO: may need to keep a pool of streams
                cudaStream_t stream = cuda_create_stream();

                auto memory_vector = node.op.get_apply().mems;

                if (node.op.is_touch()) {
                    // CASE: TOUCH
                    auto touch_kernel = build_touch(node.op.get_touch());
                    touch_kernel(stream, memory_base_ptr + memory_vector[0].offset, memory_base_ptr + memory_vector[1].offset);
                }
                else {
                    auto my_einsummable = node.op.get_einsummable();
                    if (my_einsummable.is_contraction()){
                        // CASE: CONTRACTION
                        // merge the adjacent dims
                        einsummable_t my_einsum_merged = my_einsummable.merge_adjacent_dims();
                        // print an error if we didn't find my_einsum_merged in the map
                        auto einsum_iter = einsum_to_contraction.find(my_einsum_merged);
                        if (einsum_iter == einsum_to_contraction.end()){
                            std::cout << "Error: contraction descriptor found in the map, Node idx: "<< node_idx << std::endl;
                        }
                        auto contraction_descriptor = einsum_iter->second;
                        // print offsets
                        printFloatGPU(memory_base_ptr, 100);
                        printFloatGPU(memory_base_ptr + 100, 100);
                        std::cout << "Offsets: " << memory_vector[0].offset << ", " << memory_vector[1].offset << ", " << memory_vector[2].offset << std::endl;
                        execute_contraction(stream, handle, &contraction_descriptor, 
                            memory_base_ptr + memory_vector[0].offset,
                            memory_base_ptr + memory_vector[1].offset, 
                            memory_base_ptr + memory_vector[2].offset);
                    }
                    else {
                        // CASE: OTHER EINSUMMABLE
                        auto cutensor_kernel = build_einsummable(my_einsummable);
                        cutensor_kernel(stream, handle, memory_base_ptr + memory_vector[0].offset, 
                            get_input_mem_ptrs(memory_vector, memory_base_ptr));
                    }
                }
                // after execution, we attach the stream with a callback function
                // get all the metadata needed for the callback
                callback_data_t* data = new callback_data_t;
                data->m_ptr = &m;
                data->cv_ptr = &cv;
                data->node_idx = node_idx;
                data->dependency_count = &dependency_count;
                data->memgraph = &memgraph;
                data->pending_queue = &pending_queue;
                data->num_nodes_remaining = &num_nodes_remaining;

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
                std::cout << "Error: Operation not supported: Type is among the following - move, evict, load" << std::endl;
                exit(1);
                // also updating just to check the loop
                std::unique_lock lk(m);
                auto new_nodes = node_update(dependency_count, memgraph, node_idx, num_nodes_remaining);
                add_to_queue(pending_queue, new_nodes);
                lk.unlock();
            }
        }
        
    }
}