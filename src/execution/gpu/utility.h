#pragma once

#include "../../base/setup.h"
#include <cuda_runtime.h>
#include "cutensor.h"

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

