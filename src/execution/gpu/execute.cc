#include "execute.h"
#include "kernels.h"

#include <driver_types.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>


using std::thread;
using std::queue;

cudaStream_t cuda_create_stream() {
  cudaStream_t ret;
  if(cudaStreamCreate(&ret) != cudaSuccess) {
    throw std::runtime_error("cuda_create_stream");
  }
  return ret;
}

void execute(const memgraph_t &memgraph, buffer_t &gpu_memory){
    // create a gpu_execute_state_t
    gpu_execute_state_t gpu_execute_state(memgraph, gpu_memory);
    gpu_execute_state.run();
}

// we need to create a struct that saves all the callback data
// Things that are needed:
// 1. the node that is being executed / finished
// 2. the cuda stream?
// TODO: think about what else is needed
struct callback_data_t {
  memgraph_t::node_t node;

  memgraph_t::node_t get_node() const {
    return node;
  }
};

// function definition of gpu_execute_state_t.run()
void gpu_execute_state_t::run(){

    while (true){
        // execute things that are in the apply_queue until the queue is empty
        while (pending_queue.size() != 0){
            // get the first element in the queue
            auto node = pending_queue.front();
            // remove the first element from the queue
            pending_queue.pop();
            // execute the node
            // TODO: get the mapping from the node id to the cutensor plan
            if (node.op.is_input() || node.op.is_del()){
                // do nothing
            }
            else if (node.op.is_apply()){
                // create a cuda stream since for apply we need to execute that on a cuda stream always
                cudaStream_t stream = cuda_create_stream();

                // Case - Contraction: if we get a plan, just run contraction
                if (is_contraction(node)){
                    // double check if we can find this node in the cutensor_plans
                    if (cutensor_plans.find(node) == cutensor_plans.end()){
                        throw std::runtime_error("gpu_execute_state_t::run(): cannot find the cutensor plan for contraction");
                    }
                    
                    // get the cutensor plan
                    cutensorContractionPlan_t plan = cutensor_plans[node];

                    // cutensor execute this plan with the stream
                    typedef float floatTypeCompute;
                    floatTypeCompute alpha = (floatTypeCompute)1.0f;
                    floatTypeCompute beta  = (floatTypeCompute)1.0f;

                    // get the apply_t object
                    memgraph_t::apply_t apply = node.op.get_apply();

                    // get memory locations of the input and output tensors
                    auto A_d = gpu_memory->data + apply.mems[1].offset;
                    auto B_d = gpu_memory->data + apply.mems[2].offset;
                    auto C_d = gpu_memory->data + apply.mems[0].offset;

                    cutensorStatus_t err;
                    // TODO: workspace and the workspace size are optional, but still need to look into it to see if needed
                    err = cutensorContraction(handle,
                            &plan,
                     (void*)&alpha, A_d,
                                    B_d,
                     (void*)&beta,  C_d,
                                    C_d,
                            nullptr, 0, stream /* stream */);

                    // attach the callback to the stream; TODO: decide if putting it here makes sense
                    callback_data_t* data = new callback_data_t;
                    data->node = node;
                    cudaStreamAddCallback(stream, node);
                }
                else if (is_reduction(node)){
                    execute_reduction(handle, node, gpu_memory, memgraph, stream);
                }
                else {
                    // print a message saying we need to invoke our custom kernel here
                    std::cout << "Need to invoke custom kernel here" << std::endl;
                }
            }
            else{
                // print a message saying that the operation is not supported and this operation's type
                std::cout << "Operation not supported: Type is among the following - move, evict, load" << std::endl;
            }
        }
        
        // if we get a cuda callback, update the states and check if we have more things to run
        // if (cuda_callback){
        // }
    }
}