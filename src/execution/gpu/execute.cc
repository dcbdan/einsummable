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
    return ready_nodes;
}

void add_to_queue(std::queue<int> &queue, std::vector<int> &nodes) {
    for (auto node: nodes){
        queue.push(node);
    }
}

// ------------- CUTENSOR KERNEL GENERATION ---------------

// helper function that given input metadata and stream, executes the tensor reduction
// unlike contracion, this would immediately execute the reduction with the given stream
cutensorStatus_t execute_reduction(cutensorHandle_t* handle, memgraph_t::node_t node, buffer_t gpu_memory, 
                                    const memgraph_t & memgraph, cudaStream_t stream) {

    // get the apply_t object and the einsummable from apply_t
    memgraph_t::apply_t apply = node.op.get_apply();
    auto einsum = std::get<einsummable_t>(apply.op);
    
    // *****************************************************
    // Reduction: performs the operation alpha * opReduce(opA(A)) + beta * opC(C)
    // getting all the metadata needed to generate the cutensor plan
    auto nmodeA = einsum.inns[0].size();
    auto nmodeC = 0;

    // CUDA types
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    auto extent_A = get_int64_t(einsum.inns[0]);
    int64_t* extent_C = nullptr;

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descA,
                nmodeA,
                extent_A.data(),
                NULL,/*stride*/
                typeA, CUTENSOR_OP_IDENTITY ) );

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descC,
                nmodeC,
                extent_C,
                NULL,/*stride*/
                typeC, CUTENSOR_OP_IDENTITY ) );
    auto A_d = gpu_memory->data + apply.mems[1].offset;
    auto C_d = gpu_memory->data + apply.mems[0].offset;
    // get workspace size
    uint64_t workspaceSize;
    void *workspace = nullptr;
    auto A_mode = create_mode(nmodeA);
    auto C_mode = create_mode(nmodeC);
    
    HANDLE_ERROR(cutensorReductionGetWorkspaceSize(handle, 
                 A_d, &descA, A_mode.data(),
                 C_d, &descC, C_mode.data(),
                 C_d, &descC, C_mode.data(),
                 cutensorOperator_t::CUTENSOR_OP_ADD, typeCompute, &workspaceSize));
    // check if we can get a workspace; TODO: check if this conflicts with our exiting memgraph
    if (workspaceSize > 0)
    {
        if (cudaSuccess != cudaMalloc(&workspace, workspaceSize))
        {
            workspace = nullptr;
            workspaceSize = 0;
        }
    } 

    // performs the reduction
    typedef float floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;
    HANDLE_ERROR(cutensorReduction(handle, &alpha, A_d, 
            &descA, A_mode.data(), &beta, C_d, 
            &descC, C_mode.data(), C_d, &descC, 
            C_mode.data(), cutensorOperator_t::CUTENSOR_OP_ADD, typeCompute, 
            workspace, workspaceSize, stream));
}


void execute_contraction(cutensorContractionPlan_t plan, cudaStream_t stream, buffer_t &gpu_memory, 
                        const memgraph_t &memgraph, int node_idx, cutensorHandle_t &handle) {
    // get the node
    auto node = memgraph.nodes[node_idx];
    
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
}

// we need to create a struct that saves all the callback data
// Things that are needed:
// 1. the node that is being executed / finished
// 2. the cuda stream?
// TODO: think about what else is needed

struct callback_data_t {
  std::function<void(int,int)>* update_state_ptr;
  std::mutex* m_ptr;
  std::condition_variable* cv_ptr;
  std::map<int, int> dependency_count;
  const memgraph_t memgraph;
  std::queue<int> pending_queue;
  int node_idx;

  void operator()() {
    std::mutex& m = *m_ptr;
    auto& update_state = *update_state_ptr;
    auto& cv = *cv_ptr;
    {
      std::unique_lock lk(m);
      add_to_queue(node_update(dependency_count, memgraph, node_idx));
    }
    cv.notify_all();
  }
};

// function definition of gpu_execute_state_t.run()
void gpu_execute_state_t::run() {

    while (true){
        std::mutex m;
        std::condition_variable cv;
        {
            std::unique_lock lk(m);
            cv.wait(lk, [&]{
                return pending_queue.size() != 0;
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
                // do nothing
            }
            else if (node.op.is_apply()){
                // create a cuda stream since for apply we need to execute that on a cuda stream always
                cudaStream_t stream = cuda_create_stream();
                // // *** Place actual node checks and kernel generation here
                // // Case - Contraction: if we get a plan, just run contraction
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
                }
                else if (is_reduction(node)){
                    execute_reduction(handle, node, gpu_memory, memgraph, stream);
                }
                else {
                    // print a message saying we need to invoke our custom kernel here
                    std::cout << "Need to invoke custom kernel here" << std::endl;
                }

                // we run the dummy kernel with the stream
                dummy_dispatch(nullptr, nullptr, stream);

                // after execution, we attach the stream with a callback function
                // get all the metadata needed for the callback
                callback_data_t* data = new callback_data_t;
                data->node_idx = node_idx;
                data->dependency_count = dependency_count;
                data->memgraph = memgraph;
                data->pending_queue = pending_queue;
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
                std::cout << "Operation not supported: Type is among the following - move, evict, load" << std::endl;
            }
        }
    }
}