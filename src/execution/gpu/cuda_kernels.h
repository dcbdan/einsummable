#pragma once
#include <cstdint>
#include <cuda_fp16.h>

void touch1_dispatch(float*, const float*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t, cudaStream_t,int);

void touch2_dispatch(float*, const float*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,cudaStream_t,int);

void touch3_dispatch(float*, const float*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     cudaStream_t,int);
void touch4_dispatch(float*, const float*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,
                     cudaStream_t,int);

void elementwise_power(float* out, const float* in,
cudaStream_t stream, double pow, uint64_t size);

void scale_and_increment(float* out, const float* in,
cudaStream_t stream, float scale, float increment, uint64_t size);

void custom_elementwise_1(void* out, const void* in,
cudaStream_t stream, uint64_t size);
