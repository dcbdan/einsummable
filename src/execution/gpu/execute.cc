#include "execute.h"
#include <cstdlib>
#include <driver_types.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <vector>


using std::thread;
using std::queue;

cudaStream_t cuda_create_stream() {
  cudaStream_t ret;
  if(cudaStreamCreate(&ret) != cudaSuccess) {
    throw std::runtime_error("cuda_create_stream");
  }
  return ret;
}

// update memgraph node when it is finished; modify the dependency counter of the node
// return a vector of ready nodes if there are any
vector<int> node_update(std::map<int, int> &dependency_count, const memgraph_t &memgraph, int node_idx) {
    vector<int> ready_nodes;
    for (auto out: memgraph.nodes[node_idx].outs){
        dependency_count[out] -= 1;
        if (dependency_count[out] == 0){
            ready_nodes.push_back(out);
        }
    }
    // print a update message
    std::cout << "Node " << node_idx << " finished execution." << std::endl;
    return ready_nodes;
}

void add_to_queue(std::queue<int> &queue, std::vector<int> &nodes) {
    for (auto node: nodes){
        queue.push(node);
    }
    // print how many elements are in the queue
    // std::cout << "Queue size: " << queue.size() << std::endl;
}

std::map<int, int> get_dependencies(const memgraph_t &memgraph) {
    std::map<int, int> dependency_count;
    for (int i = 0; i < memgraph.nodes.size(); i++){
        dependency_count[i] = memgraph.nodes[i].inns.size();
    }
    return dependency_count;
}

bool is_complete(std::map<int, int> &dependency_count) {
    for (auto it = dependency_count.begin(); it != dependency_count.end(); it++){
        if (it->second != 0){
            return false;
        }
    }
    return true;
}

// calling cuda malloc to allocate memory for a given size
void* cuda_malloc(size_t size) {
  void* ret;
  if(cudaMalloc(&ret, size) != cudaSuccess) {
    throw std::runtime_error("cuda_malloc");
  }
  return ret;
}

// get a callback data struct that keeps track of the current node that finished execution
// Has all the data structures required to update things
struct callback_data_t {
  std::mutex* m_ptr;
  std::condition_variable* cv_ptr;
  std::map<int, int>* dependency_count;
  const memgraph_t* memgraph = new memgraph_t();
  std::queue<int>* pending_queue;
  int node_idx;

  void operator()() {
    std::mutex& m = *m_ptr;
    auto& cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      // update the queue since this node is finished
      auto new_nodes = node_update(*dependency_count, *memgraph, node_idx);
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
        std::mutex m;
        // wait till some node finishes and notifies the cv
        std::condition_variable cv;
        {
            std::unique_lock lk(m);
            cv.wait(lk, [&]{
                return pending_queue.size() > 0;
            });
        }
        // execute things that are in the apply_queue until the queue is empty
        while (pending_queue.size() != 0){
            // get the first element in the queue
            auto node_idx = pending_queue.front();
            auto node = memgraph.nodes[node_idx];
            // remove the first element from the queue
            pending_queue.pop();
            // execute the node
            // TODO: get the mapping from the node id to the cutensor plan
            if (node.op.is_input() || node.op.is_del()){
                // do nothing but add the node to the finished queue
                auto new_nodes = node_update(dependency_count, memgraph, node_idx);
                add_to_queue(pending_queue, new_nodes);
            }
            else if (node.op.is_apply()){
                // create a cuda stream since for apply we need to execute that on a cuda stream always
                cudaStream_t stream = cuda_create_stream();

                // we run the dummy kernel with the stream
                dummy_dispatch(nullptr, nullptr, stream);

                // after execution, we attach the stream with a callback function
                // get all the metadata needed for the callback
                callback_data_t* data = new callback_data_t;
                data->m_ptr = &m;
                data->cv_ptr = &cv;
                data->node_idx = node_idx;
                data->dependency_count = &dependency_count;
                data->memgraph = &memgraph;
                data->pending_queue = &pending_queue;
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
                // std::cout << "Operation not supported: Type is among the following - move, evict, load" << std::endl;\

                // also updating just to check the loop
                auto new_nodes = node_update(dependency_count, memgraph, node_idx);
                add_to_queue(pending_queue, new_nodes);
            }
        }
        // if the num_nodes_remaining is 0, then we are done
        if (is_complete(num_nodes_remaining)){
            // if the dependency count for all node is 0, then we are done
            // else throw an error
            if (is_complete(dependency_count)){
                std::cout << "All nodes finished execution." << std::endl;
                break;
            }
            else{
                std::cout << "Error: All nodes finished execution but there are still nodes in the queue." << std::endl;
                exit(0);
            }
            exit(0);
        }
    }
}