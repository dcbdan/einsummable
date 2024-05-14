#pragma once

#include "graph.h"
#include "../base/setup.h"
#include "einsummable.h"
#include "graph.h"
#include <algorithm>
// #include <unordered_map>

#include "simplescalarop.h"
#include "dbuffer.h"
#include "reference.h"


// #include "../../einsummable/taskgraph.h" // touch_t

#include <thread>

#include "../engine/gpu/gpu_kernel_manager.h"
#include "../engine/gpu/cuda_kernels.h"
#include "../engine/gpu/kernels.h"
#include "../engine/gpu/utility.h"
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <stdexcept>
#include <iostream>
#include <cstdlib> // For rand()

class fusion_t {
public:
    graph_t apply(const graph_t& graph);

    static bool is_exp(const einsummable_t& e);
    static bool is_relu(const einsummable_t& e);
    static bool is_fused_exp_relu(const einsummable_t& e);

    std::vector<std::pair<int, int>> findFusibleNodes(const graph_t& graph);
    einsummable_t createFusedEinsummable(const einsummable_t& non_elementwise, const einsummable_t& elementwise);
    graph_t fuseNodes(const graph_t& graph, const std::vector<std::pair<int, int>>& fusiblePairs);

    // uint64_t calculate_workspace_size_for_fused_exp_relu(einsummable_t const& einsummable);
    // static void graph_build();
    // static void graph_capture(cudaStream_t captureStream);
    static void build_cuda_graph_for_fused_exp_relu(einsummable_t const& einsummable);
    static void execute_cuda_graph_for_fused_exp_relu(einsummable_t const& einsum, cudaStream_t stream);
private:
    static std::unordered_map<einsummable_t, cudaGraphExec_t> cudaGraphExec_handles;

    
};
