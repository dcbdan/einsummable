#pragma once

#include "../../base/setup.h"
#include <cuda_runtime.h>
#include "cutensor.h"

#include <memory>

// utility functions
cudaStream_t cuda_create_stream();
void* offset_increment(const void *ptr, int offset);
void* gpu_allocate_memory(size_t size, int device_id);

// debug
void printFloatCPU(const float *cpu_ptr, int count);
void printFloatGPU(const void *gpu_ptr, int count);
void init_value(float *ptr, int count, float value);
void checkAlignment(cutensorHandle_t *handle, float *ptr,
                    cutensorTensorDescriptor_t desc);

void handle_cutensor_error(cutensorStatus_t error, string msg = "");

void handle_cuda_error(cudaError_t error, string msg = "");

struct cuda_stream_t {
  cuda_stream_t() {
    handle_cuda_error(cudaStreamCreate(&stream), "cuda_stream_t: constructor");
  }
  ~cuda_stream_t() {
    handle_cuda_error(cudaStreamDestroy(stream), "cuda_stream_t: destructor");
  }

  cudaStream_t stream;
};

using cuda_stream_ptr_t = std::shared_ptr<cuda_stream_t>;
