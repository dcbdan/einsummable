#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <driver_types.h>

void touch1_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t, cudaStream_t,int,int);

void touch2_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,cudaStream_t,int,int);

void touch3_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     cudaStream_t,int,int);
void touch4_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,
                     cudaStream_t,int,int);

void elementwise_power(float* out, const float* in,
cudaStream_t stream, double pow, uint64_t size);

void scale_and_increment(float* out, const float* in,
cudaStream_t stream, float scale, float increment, uint64_t size);

void custom_elementwise_1(void* out, const void* in,
cudaStream_t stream, uint64_t size);

void fillTri_dispatch(void* mem, uint64_t nrow, uint64_t ncol, uint64_t start,
  uint64_t lower, uint64_t upper, cudaStream_t stream, int dtype_info);

void fill_constant_dispatch(void* mem, uint64_t nelem, uint64_t value,
  cudaStream_t stream, int dtype_info);

void elementwise_exp(float* out, const float* in, 
cudaStream_t stream, int n);

void elementwise_relu(float* out, const float* in, 
cudaStream_t stream, int n);
