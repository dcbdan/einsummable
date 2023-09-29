#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuComplex.h>

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
