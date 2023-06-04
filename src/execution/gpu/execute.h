#pragma once
#include "../../base/setup.h"
#include "../../einsummable/memgraph.h"
#include "../../einsummable/reference.h" // buffer_t
#include "../../../libcutensor/include/cutensor.h"
#include "device_launch_parameters.h"
#include "dummy_kernels.h"

#include <cstddef>
#include <map>
#include <thread>
#include <cuda_runtime.h>
#include <variant>

using memgraph_t = memgraph_t;

// we need to define HANDLE_ERROR properly since it's not included in the header file 
// defined: (https://docs.nvidia.com/cuda/cutensor/getting_started.html#determine-algorithm-and-workspace)
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

// ------------- CUTENSOR HELPER FUNCTIONS ---------------

// get a vector of ints, return a vector of int64_t
vector<int64_t> get_int64_t(const vector<int> &vec) {
    vector<int64_t> int64_t_vec;
    for (auto v: vec){
        int64_t_vec.push_back(v);
    }
    return int64_t_vec;
}

// create an array from 1 to n
vector<int>create_mode(int n) {
    vector<int> mode;
    for (int i = 1; i <= n; i++){
        mode.push_back(i);
    }
    return mode;
}

// ------------- NODE CHECKS ---------------

// return true if the node is a reduction, false otherwise
// conditions of the reduction:
// 1. there is one input
// 2. castable is a +
// 3. out rank is less than the join shape size
bool is_reduction(memgraph_t::node_t node) {
    // reduction needs to be an apply node
    if (!node.op.is_apply()){
        return false;
    }

    memgraph_t::apply_t apply = node.op.get_apply();
    // we are not supporting if the apply is a touch
    if (std::holds_alternative<touch_t>(apply.op)){
        return false;
    }

    auto einsum = std::get<einsummable_t>(apply.op);
    // 1. there is one input
    if (einsum.inns.size() != 1){
        return false;
    }
    // 2. castable is a +
    if (einsum.castable != castable_t::add){
        return false;
    }
    // 3. out rank is less than the join shape size
    if (einsum.out_rank >= einsum.join_shape.size()){
        return false;
    }

    return true;
}

// return true if the node is a contraction, false otherwise
bool is_contraction(memgraph_t::node_t node) {
    // contraction needs to be an apply node
    if (!node.op.is_apply()){
        return false;
    }

    memgraph_t::apply_t apply = node.op.get_apply();
    // we are not supporting if the apply is a touch
    if (std::holds_alternative<touch_t>(apply.op)){
        return false;
    }

    auto einsum = std::get<einsummable_t>(apply.op);
    // check if the castable is a multiply and the einsum is a castable
    // if (!(einsum.castable) || einsum.castable != castable_t::mul){}
    if (einsum.castable != castable_t::add){
        return false;
    }

    if (!einsum.join.is_mul()){
        return false;
    }

    if (einsum.out_rank == einsum.join_shape.size()){
        return false;
    }

    return true;
}

// return true if the node is a touch, false otherwise
bool is_touch(memgraph_t::node_t node) {
    // touch needs to be an apply node
    if (!node.op.is_apply()){
        return false;
    }

    memgraph_t::apply_t apply = node.op.get_apply();
    // we are not supporting if the apply is a touch
    if (std::holds_alternative<touch_t>(apply.op)){
        return true;
    }
    return false;
}


// input: node from memgraph and all other necessary metadata, output: cutensor plan
// we are saving this plan in a map so that we can precompute the plan and save it before executing the memgraph
cutensorContractionPlan_t cutensor_plan_from_node(cutensorHandle_t* handle, memgraph_t::node_t node, 
    buffer_t gpu_memory, const memgraph_t & memgraph)
    {
    // get the apply_t object and the einsummable from apply_t
    memgraph_t::apply_t apply = node.op.get_apply();
    auto einsum = std::get<einsummable_t>(apply.op);
    
    // *****************************************************
    // contraction: performs the operation alpha * A * B + beta * C
    // getting all the metadata needed to generate the cutensor plan
    auto nmodeA = einsum.inns[0].size();
    auto nmodeB = einsum.inns[1].size();
    auto nmodeC = 0;
    if (einsum.inns.size() == 3){
        nmodeC = einsum.inns[2].size();
    }
    // CUDA types
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    auto extent_A = get_int64_t(einsum.inns[0]);
    auto extent_B = get_int64_t(einsum.inns[1]);
    int64_t* extent_C = nullptr;
    if (apply.mems.size() == 4){
        extent_C = get_int64_t(einsum.inns[2]).data();
    }
    
    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descA,
                nmodeA,
                extent_A.data(),
                NULL,/*stride*/
                typeA, CUTENSOR_OP_IDENTITY ) );

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descB,
                nmodeB,
                extent_B.data(),
                NULL,/*stride*/
                typeB, CUTENSOR_OP_IDENTITY ) );

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descC,
                nmodeC,
                extent_C,
                NULL,/*stride*/
                typeC, CUTENSOR_OP_IDENTITY ) );

    // printf("Initialize cuTENSOR and tensor descriptors\n");

    /* ***************************** */

    // get the data/memory pointers of the input tensors
    auto A_d = gpu_memory->data + apply.mems[1].offset;
    auto B_d = gpu_memory->data + apply.mems[2].offset;
    auto C_d = gpu_memory->data + apply.mems[0].offset;
    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                A_d,
                &descA,
                &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                B_d,
                &descB,
                &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                C_d,
                &descC,
                &alignmentRequirementC) );

    printf("Query best alignment requirement for our pointers\n");

    /* ***************************** */

    // Create the Contraction Descriptor
    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR( cutensorInitContractionDescriptor( handle,
                &desc,
                &descA, create_mode(nmodeA).data(), alignmentRequirementA,
                &descB, create_mode(nmodeB).data(), alignmentRequirementB,
                &descC, create_mode(nmodeC).data(), alignmentRequirementC,
                &descC, create_mode(nmodeC).data(), alignmentRequirementC,
                typeCompute) );

    printf("Initialize contraction descriptor\n");

    /* ***************************** */

    // Set the algorithm to use
    cutensorContractionFind_t find;
    HANDLE_ERROR( cutensorInitContractionFind(
                handle, &find,
                CUTENSOR_ALGO_DEFAULT) );

    // printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( cutensorContractionGetWorkspaceSize(handle,
                &desc,
                &find,
                CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    // printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */

    // Create Contraction Plan
    cutensorContractionPlan_t plan;
    HANDLE_ERROR( cutensorInitContractionPlan(handle,
                                                &plan,
                                                &desc,
                                                &find,
                                                worksize) );

    // printf("Create plan for contraction\n");

    return plan;

}

// ---------------- Memgraph traversal ----------------
// given a memgraph, traverse the graph and get all the dependencies of a node represented by node.inns()
// return a map from the node to the number of its dependencies
std::map<int, int> get_dependencies(const memgraph_t &memgraph) {
    std::map<int, int> dependency_count;
    for (int i = 0; i < memgraph.nodes.size(); i++){
        dependency_count[i] = memgraph.nodes[i].inns.size();
    }
    return dependency_count;
}

struct gpu_execute_state_t 
{
    memgraph_t memgraph;
    buffer_t gpu_memory;
    // maintain a queue of tasks that are pending to be executed
    std::queue<int> pending_queue;

    // if we are not modifying the original memgraph, we can use a map to store the dependency count of each node
    // key: node index, value: dependency count of that node
    // whenever a node finishes execution, we decrement the dependency count of all its output
    // if the dependency count of a node reaches 0, we add it to the pending_queue
    std::map<int, int> dependency_count;

    // cutensor related:
    cutensorHandle_t* handle;
    
    // a map from the node of the memgraph to the cutensor plan 
    // we only have plans for operation contraction, so we only need to store the plans for those nodes
    // remember to throw an error if the node is not a contraction but trying to access the map
    map<int, cutensorContractionPlan_t> cutensor_plans;

    gpu_execute_state_t(const memgraph_t input_memgraph, buffer_t &input_gpu_memory): memgraph(input_memgraph), gpu_memory(input_gpu_memory){

        // create a cutensor handle
        HANDLE_ERROR( cutensorCreate(&handle) );

        dependency_count = get_dependencies(memgraph);

        // check all elements from the memgraph and add the nodes with no dependencies to the apply_queue
        for (int i = 0; i < memgraph.nodes.size(); i++)
        {
            if (memgraph.nodes[i].inns.size() == 0)
            {
                pending_queue.push(i);
            }

            if (is_contraction(memgraph.nodes[i]))
            {
                cutensor_plans[i] = cutensor_plan_from_node(handle, memgraph.nodes[i], gpu_memory, memgraph);
            }
        }
    }

    void run();
    
};

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute(
  const memgraph_t &memgraph,
  buffer_t& gpu_memory);
