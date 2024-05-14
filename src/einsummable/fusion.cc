#include "fusion.h"

std::unordered_map<einsummable_t, cudaGraphExec_t> fusion_t::cudaGraphExec_handles = {};

// Find pairs of nodes where an 'exp' node is followed by a 'ReLU' node.
std::vector<std::pair<int, int>> fusion_t::findFusibleNodes(const graph_t& graph) 
{
    std::vector<std::pair<int, int>> fusiblePairs;

    for (int i = 0; i < graph.nodes.size(); i++) {
        const auto& node = graph.nodes[i];

        if (node.op.is_einsummable()) 
        {
            const auto& einsum = node.op.get_einsummable();
            if (is_exp(einsum)) // Check for 'exp' operation
            {  
                for (int child_id : node.outs)  // Get the child nodes
                { 
                    const auto& child_node = graph.nodes[child_id];
                    if (child_node.op.is_einsummable() && is_relu(child_node.op.get_einsummable())) // Check for 'ReLU'
                    {  
                        fusiblePairs.emplace_back(i, child_id);
                    }
                }
            }
        }
    }

    return fusiblePairs;
}

graph_t fusion_t::apply(const graph_t& originalGraph) 
{
    auto fusiblePairs = findFusibleNodes(originalGraph);
    return fuseNodes(originalGraph, fusiblePairs);
}



graph_t fusion_t::fuseNodes(const graph_t& graph, const std::vector<std::pair<int, int>>& fusiblePairs) 
{
    // Start with a copy of the original graph
    graph_t newGraph = graph; 

    for (const auto& pair : fusiblePairs) 
    {
        auto& exp_node = newGraph.nodes[pair.first];
        auto& relu_node = newGraph.nodes[pair.second];

        // Create fused einsummable
        einsummable_t fused = createFusedEinsummable(exp_node.op.get_einsummable(), relu_node.op.get_einsummable());

        // Replace the 'exp' node with the fused operation in the new graph
        newGraph.nodes[pair.first].op = graph_t::op_t(fused);

        newGraph.nodes[pair.first].op.set_save(true);

        // Remove the 'ReLU' node from the new graph, and fix connections
        newGraph.removeNode(pair.second);
    }

    return newGraph;
}

// Create a fused einsummable that represents 'exp' followed by 'ReLU'.
einsummable_t fusion_t::createFusedEinsummable(const einsummable_t& exp, const einsummable_t& relu) 
{
    // Ensure the operations are 'exp' followed by 'ReLU'
    if (!is_exp(exp) || !is_relu(relu)) {
        throw std::runtime_error("Unsupported operations for fusion. Expected 'exp' followed by 'ReLU'.");
    }

    // Check if both operations are compatible for fusion
    if (exp.join_shape != relu.join_shape || exp.inns != relu.inns || exp.out_rank != relu.out_rank) {
        throw std::runtime_error("Exp is not compatable with ReLU");
    }

    // Create a new operation type that represents the fused 'exp' and 'ReLU'.
    scalarop_t fused_op = scalarop_t::make_exp_relu(); 

    // The fused einsummable will have the same input tensors as the 'exp' operation,
    // and it will use the newly created scalar operation.
    return einsummable_t(exp.join_shape, exp.inns, exp.out_rank, fused_op);
}

// TODO: Find a more Robust implementation
bool fusion_t::is_exp(const einsummable_t& e) {
    if (e.inns.size() == 1) {  // Assuming exp is a unary operation
        scalarop_t op = e.join;
        op = op.simplify();
        auto op_str = op.to_cppstr();
        // Check if the operation string starts with "_exp(x0)"
        if (op_str.find("_exp(x0)") == 0) { // TODO: This is NOT a good way. It can Fail easily
            return true;
        }
    }
    return false;
}

// TODO: Find a more Robust implementation
bool fusion_t::is_relu(const einsummable_t& e) {
    if (e.inns.size() == 1) {  // Assuming ReLU is a unary operation
        scalarop_t op = e.join;
        op = op.simplify();
        auto op_str = op.to_cppstr();
        // Check if the operation string represents ReLU, assuming ReLU is represented as "(x0 >= 0 ? x0 : 0)"
        if (op_str.find("(f32|0>=x0?f32|0:x0)") == 0) { // TODO: This is NOT a good way. It can Fail easily
            return true;
        }
    }
    return false;
}

///////////////////////////////////////////////////////////////////////
// Helper function for "Kernel-Fusion"
bool fusion_t::is_fused_exp_relu(const einsummable_t& e) {
    scalarop_t op = e.join.simplify();
    auto op_str = op.to_cppstr();
    if (op_str.find("(f32|0>=_exp(x0)?f32|0:_exp(x0))") == 0) {
        return true;
    }
    return false;
}

// uint64_t kernel_manager_t::calculate_workspace_size_for_fused_exp_relu(einsummable_t const& einsummable) {
//     uint64_t a_size = einsummable.join_shape[0]; 
//     uint64_t worksize = a_size * sizeof(float);  
//     return worksize;
// }


// CUDA kernel definitions
// __global__ void exp_kernel(const float* in, float* out, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         out[idx] = expf(in[idx]); // Exponential function
//     }
// }

// __global__ void relu_kernel(const float* in, float* out, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         out[idx] = in[idx] > 0 ? in[idx] : 0; // ReLU function
//     }
// }


// void fusion_t::graph_build() {
//     int numElements = 100*100;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

//     // Host and device memory allocation
//     float* hostInput = new float[numElements];
//     for (int i = 0; i < numElements; i++) {
//         hostInput[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
//     }

//     float *deviceInput, *deviceOutput;
//     cudaMalloc(&deviceInput, numElements * sizeof(float));
//     cudaMalloc(&deviceOutput, numElements * sizeof(float));
//     cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);

//     // Kernel node parameters
//     void* expKernelArgs[] = { (void*)&deviceInput, (void*)&deviceOutput, &numElements };
//     void* reluKernelArgs[] = { (void*)&deviceOutput, (void*)&deviceOutput, &numElements };

//     cudaKernelNodeParams expParams;
//     expParams.func = (void*)exp_kernel;
//     expParams.gridDim = dim3(blocksPerGrid);
//     expParams.blockDim = dim3(threadsPerBlock);
//     expParams.sharedMemBytes = 0;
//     expParams.kernelParams = expKernelArgs;
//     expParams.extra = nullptr;

//     cudaKernelNodeParams reluParams;
//     reluParams.func = (void*)relu_kernel;
//     reluParams.gridDim = dim3(blocksPerGrid);
//     reluParams.blockDim = dim3(threadsPerBlock);
//     reluParams.sharedMemBytes = 0;
//     reluParams.kernelParams = reluKernelArgs;
//     reluParams.extra = nullptr;

//     // Create and populate CUDA graph
//     cudaGraph_t graph;
//     cudaGraphNode_t expNode, reluNode;
//     cudaGraphCreate(&graph, 0);
//     cudaGraphAddKernelNode(&expNode, graph, nullptr, 0, &expParams);
//     cudaGraphAddKernelNode(&reluNode, graph, nullptr, 0, &reluParams);
//     cudaGraphNode_t dependencies[1] = { expNode };
//     cudaGraphAddDependencies(graph, &reluNode, dependencies, 1);

//     cudaGraphExec_t graphExec;
//     cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
//     cudaGraphExec_handles.insert({einsummable, graphExec});



//     // Clean up
//     cudaFree(deviceInput);
//     cudaFree(deviceOutput);
//     delete[] hostInput;

// }

// void fusion_t::graph_capture(cudaStream_t captureStream) {
//   // Y = exp(X) followed by Z = RELU(Y)
//   // Goal is to run exp followed by relu cutensor kernels so that I can capture them in a cuda graph
//   uint64_t a = 100;
//   uint64_t b = 100;
  
//   einsummable_t exp_einsum({a, b}, vector<vector<int>>{{0,1}}, 2, scalarop_t::make_exp());
//   einsummable_t relu_einsum({a, b}, vector<vector<int>>{{0,1}}, 2, scalarop_t::make_relu());

//   dtype_t dtype = default_dtype();
//   dbuffer_t X = make_dbuffer(dtype, a*b);
//   X.random();
//   dbuffer_t Y = make_dbuffer(dtype, a*b);
//   Y.zeros();
//   dbuffer_t Z = make_dbuffer(dtype, a*b);
//   Z.zeros();
// //   dbuffer_t Y_ref = reference_einsummable(exp_einsum, {X});
  
// //   cutensorHandle_t handle;
// //   cutensorCreate(&handle);
// //   cudaStream_t stream;
// //   cudaStreamCreate(&stream);
  
//   size_t sizeX = X.size();
//   size_t sizeY = Y.size();
//   size_t sizeZ = Z.size();

//   void *x, *y, *z;
//   cudaError_t err;

// //   cudaMallocAsync((void**)&x, sizeX, captureStream);
// //   cudaMallocAsync((void**)&y, sizeY, captureStream);
// //   cudaMallocAsync((void**)&z, sizeZ, captureStream);

// //   cudaMalloc((void**)&x, sizeX);
// //   cudaMalloc((void**)&y, sizeY);
// //   cudaMalloc((void**)&z, sizeZ);


//   float* X_ptr = (float*)X.ptr();
//   float* Y_ptr = (float*)Y.ptr();
//   float* Z_ptr = (float*)Z.ptr();

// //   cudaMemcpy(x, X.ptr(), sizeX, cudaMemcpyHostToDevice);
// //   cudaMemcpy(y, Y.ptr(), sizeY, cudaMemcpyHostToDevice);
// //   cudaMemcpy(z, Z.ptr(), sizeZ, cudaMemcpyHostToDevice);


// //   cudaMemcpyAsync(x, X.ptr(), sizeX, cudaMemcpyHostToDevice, captureStream);
// //   cudaMemcpyAsync(y, Y.ptr(), sizeY, cudaMemcpyHostToDevice, captureStream);
// //   cudaMemcpyAsync(z, Z.ptr(), sizeZ, cudaMemcpyHostToDevice, captureStream);

  
//   std::cout << "Right before calling kernel_manager_t km;\n";
// //   int count;
// //   cudaGetDeviceCount(&count);
// //   std::cout << count << "\n";
//   kernel_manager_t km(0);
//   std::cout << "Right after calling kernel_manager_t km;\n";
//   auto exp_workspace_info = km.build(exp_einsum);
//   auto relu_workspace_info = km.build(relu_einsum);

//   vector<void const*> inns, interms;
//   inns.push_back(x);
//   interms.push_back(y);
//   km(exp_einsum,captureStream,y,inns);
//   km(relu_einsum,captureStream,z,interms);

// //   cudaStreamDestroy(stream);


// //   cudaMemcpyAsync(Z.ptr(), z, sizeZ, cudaMemcpyDeviceToHost, captureStream);
// //   cudaMemcpy(Z.ptr(), z, sizeZ, cudaMemcpyDeviceToHost);


// //   if(!is_close(Y_ref, Y)) {
// //     printf("EXP KERNEL:\n");
// //     DOUT(Y_ref);
// //     DOUT(Y);
// //     printf("EXP Kernel ARE NOT CLOSE!\n");
// //   }else{
// //     std::cout << "Elementwise operation successful for dtype "<< dtype << " of EXP Kernel" <<std::endl;
// //   }
// //   cudaFreeAsync(x, captureStream);
// //   cudaFreeAsync(y, captureStream);
// //   cudaFreeAsync(z, captureStream);
// }

void fusion_t::build_cuda_graph_for_fused_exp_relu(einsummable_t const& einsummable) {
    
    uint64_t a = 10'000;
    uint64_t b = 10'000;

    uint64_t numElements = a*b;

    // Host and device memory allocation
    float* hostInput = new float[numElements];
    for (int i = 0; i < numElements; i++) {
        hostInput[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    float *deviceInput, *deviceOutput;
    cudaMalloc(&deviceInput, numElements * sizeof(float));
    cudaMalloc(&deviceOutput, numElements * sizeof(float));
    cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // // Kernel node parameters
    // void* expKernelArgs[] = { (void*)&deviceInput, (void*)&deviceOutput, &numElements };
    // void* reluKernelArgs[] = { (void*)&deviceOutput, (void*)&deviceOutput, &numElements };

    // cudaKernelNodeParams expParams;
    // expParams.func = (void*)exp_kernel;
    // expParams.gridDim = dim3(blocksPerGrid);
    // expParams.blockDim = dim3(threadsPerBlock);
    // expParams.sharedMemBytes = 0;
    // expParams.kernelParams = expKernelArgs;
    // expParams.extra = nullptr;

    // cudaKernelNodeParams reluParams;
    // reluParams.func = (void*)relu_kernel;
    // reluParams.gridDim = dim3(blocksPerGrid);
    // reluParams.blockDim = dim3(threadsPerBlock);
    // reluParams.sharedMemBytes = 0;
    // reluParams.kernelParams = reluKernelArgs;
    // reluParams.extra = nullptr;

    // // Create and populate CUDA graph
    // cudaGraph_t graph;
    // cudaGraphNode_t expNode, reluNode;
    // cudaGraphCreate(&graph, 0);
    // cudaGraphAddKernelNode(&expNode, graph, nullptr, 0, &expParams);
    // cudaGraphAddKernelNode(&reluNode, graph, nullptr, 0, &reluParams);
    // // cudaGraphNode_t dependencies[1] = { expNode };
    // cudaGraphAddDependencies(graph, &reluNode, &expNode, 1);

    // cudaGraphExec_t graphExec;
    // cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    // cudaGraphExec_handles.insert({einsummable, graphExec});

    // Create a stream for capturing
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Start capturing on the stream
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Call the wrapper functions that encapsulate the CUDA kernels
    elementwise_exp(deviceInput, deviceOutput, stream, numElements);
    elementwise_relu(deviceOutput, deviceOutput, stream, numElements);

    // End capturing
    cudaStreamEndCapture(stream, &graph);


    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphDebugDotPrint(graph, "/home/sleem/einsummable/cuda_graph.dot", cudaGraphDebugDotFlagsVerbose);

    // Store the executable graph
    cudaGraphExec_handles.insert({einsummable, graphExec});



    // Clean up
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete[] hostInput;
    cudaStreamDestroy(stream);





  // cudaGraph_t graph;
  // cudaGraphExec_t graphExec;
  // // cudaStream_t captureStream;
  // cudaError_t err;
  // err = cudaStreamCreate(&captureStream);
  // if (err != cudaSuccess) {
  //   std::cerr << "Failed [cudaStreamCreate]: " << cudaGetErrorString(err) << std::endl;
  //   return;
  // }

//   err = cudaGraphCreate(&graph, 0);
//   if (err != cudaSuccess) {
//     std::cerr << "Failed [cudaGraphCreate]: " << cudaGetErrorString(err) << std::endl;
//     return;
//   }

  

  // err = cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);
  // if (err != cudaSuccess) {
  //   std::cerr << "Failed [cudaStreamBeginCapture]: " << cudaGetErrorString(err) << std::endl;
  //   return;
  // }


  // TODO: Should be a general way to do it --> Chris's Idea is sing vector<einsummable_t> that contains the fused nodes
  // Y = exp(X) followed by Z = RELU(Y)
  // graph_capture(captureStream);

//   optional<list_simple_scalarop_t> maybe_expOp = list_simple_scalarop_t::make(exp_einsum.join);
//   if (maybe_expOp) {
//     list_simple_scalarop_t const& expOp = maybe_expOp.value(); 
//     auto exp_plan = make_elementwise_plans(expOp, exp_einsum.join_shape,
//                                       exp_einsum.inns, exp_einsum.out_rank);
//   };

//   optional<list_simple_scalarop_t> maybe_reluOp = list_simple_scalarop_t::make(relu_einsum.join);
//   if (maybe_reluOp) {
//     list_simple_scalarop_t const& reluOp = maybe_reluOp.value(); 
//     auto exp_plan = make_elementwise_plans(reluOp, relu_einsum.join_shape,
//                                       relu_einsum.inns, relu_einsum.out_rank);
//   };


  // err = cudaStreamEndCapture(captureStream, &graph);
  // if (err != cudaSuccess) {
  //   std::cerr << "Failed [cudaStreamEndCapture]: " << cudaGetErrorString(err) << std::endl;
  //   return;
  // }
  
  // err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
  // if (err != cudaSuccess) {
  //   std::cerr << "Failed [cudaGraphInstantiate]: " << cudaGetErrorString(err) << std::endl;
  //   return;
  // }

  // // Print the graph to a dot file
  //   // Save the DOT representation to a file
  //   err = cudaGraphDebugDotPrint(graph, "/home/sleem/einsummable/cuda_graph.dot", cudaGraphDebugDotFlagsVerbose);
  //   if (err != cudaSuccess) {
  //       std::cerr << "Failed to print graph to dot file: " << cudaGetErrorString(err) << std::endl;
  //   }

  // cudaGraphExec_handles.insert({einsummable, graphExec});

  // // Parameters for kernel nodes (EXP and RELU)
  // cudaKernelNodeParams expParams = {/* initialize with cutensor or custom kernel for EXP */};
  // cudaKernelNodeParams reluParams = {/* initialize with cutensor or custom kernel for RELU */};

  // // Add nodes to the graph
  // cudaGraphNode_t expNode, reluNode;
  // cudaGraphAddKernelNode(&expNode, graph, nullptr, 0, &expParams);
  // cudaGraphAddKernelNode(&reluNode, graph, nullptr, 0, &reluParams);

  // // Set dependencies: EXP must execute before RELU
  // cudaGraphAddDependencies(graph, &expNode, &reluNode, 1);

  // Instantiate and store the graph
  // cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

  // You might need a wrapper or manager for CUDA Graph Exec handles
  // cudaGraphExec_handles.insert({einsummable, graphExec});

  // // Assuming workspace calculation (needs implementation)
  // uint64_t workspace_size = calculate_workspace_size_for_fused_exp_relu(einsummable);
  // return workspace_info_t(workspace_size);
}
///////////////////////////////////////////////////////////////////////

void fusion_t::execute_cuda_graph_for_fused_exp_relu(einsummable_t const& einsum, cudaStream_t stream) {
    cudaGraphExec_t graphExec = cudaGraphExec_handles.at(einsum);    
    cudaError_t err;
    err = cudaGraphLaunch(graphExec, stream);
    if (err != cudaSuccess) {
    std::cerr << "Failed [cudaGraphLaunch]: " << cudaGetErrorString(err) << std::endl;
    return;
    }
    std::cout << "LAUNCHED\n";
    cudaStreamSynchronize(stream);
}
