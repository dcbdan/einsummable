#pragma once

#include "../../base/setup.h"
#include <cuda_runtime.h>
#include "cutensor.h"

#include <memory>
#include "cublas_v2.h"

// utility functions
cudaStream_t cuda_create_stream();
void const*  offset_increment(void const* ptr, uint64_t offset);
void*        offset_increment(void* ptr, uint64_t offset);
void*        gpu_allocate_memory(size_t size, int device_id);

// debug
void printFloatCPU(const float* cpu_ptr, int count);
void printFloatGPU(const void* gpu_ptr, int count);
void init_value(float* ptr, int count, float value);
void handle_cutensor_error(cutensorStatus_t error, string msg = "");

void handle_cuda_error(cudaError_t error, string msg = "");

void handle_cublas_error(cublasStatus_t error, string msg = "");

struct cuda_stream_t {
    cuda_stream_t()
    {
        DOUT("cuda_stream_t: constructor");
        handle_cuda_error(cudaStreamCreate(&stream), "cuda_stream_t: constructor");
    }
    ~cuda_stream_t()
    {
        DOUT("cuda_stream_t: destructor");
        handle_cuda_error(cudaStreamDestroy(stream), "cuda_stream_t: destructor");
    }

    cudaStream_t stream;
};

using cuda_stream_ptr_t = std::shared_ptr<cuda_stream_t>;
