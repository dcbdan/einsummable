#include "cuda_kernels.h"
#include <cstdint>
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cuda/atomic>
#include <sys/types.h>
#include <cuda_fp16.h>

struct FunctorCopy {
  __device__ void operator()(__half& a, const __half& b) const {
    a = b;
  }

  __device__ void operator()(float& a, const float& b) const {
    a = b;
  }

  __device__ void operator()(double& a, const double& b) const {
    a = b;
  }

  __device__ void operator()(cuFloatComplex& a, const cuFloatComplex& b) const {
    a = b;
  }
};

struct FunctorAdd {
  __device__ void operator()(__half& a, const __half& b) const {
    atomicAdd(&a, b);
  }

  __device__ void operator()(float& a, const float& b) const {
    atomicAdd(&a, b);
  }

  __device__ void operator()(double& a, const double& b) const {
    atomicAdd(&a, b);
  }

  __device__ void operator()(cuFloatComplex& a, const cuFloatComplex& b) const {
    float* ar = (float*)&a;
    float* ai = ar + 1;
    float const* br = (float const*)&b;
    float const* bi = br + 1;
    atomicAdd(ar, *br);
    atomicAdd(ai, *bi);
  }
};

struct FunctorMin {
  __device__ void operator()(__half& a, const __half& b) const {
    //return __hlt(a, b) ? a : b;
    while(a > b) {
      a = b;
    }
  }

  __device__ void operator()(float& a, const float& b) const {
    //return fminf(a,b);
    while(a > b) {
      a = b;
    }
  }

  __device__ void operator()(double& a, const double& b) const {
    //return fmin(a,b);
    while(a > b) {
      a = b;
    }
  }

  __device__ void operator()(cuFloatComplex& a, const cuFloatComplex& b) const {}
};

struct FunctorMax {
  __device__ void operator()(__half& a, const __half& b) const {
    //return __hgt(a, b) ? a : b;
    while(a < b) {
      a = b;
    }
  }

  __device__ void operator()(float& a, const float& b) const {
    //return fmaxf(a,b);
    while(a < b) {
      a = b;
    }
  }

  __device__ void operator()(double& a, const double& b) const {
    //return fmax(a,b);
    while(a < b) {
      a = b;
    }
  }

  __device__ void operator()(cuFloatComplex& a, const cuFloatComplex& b) const {}
};


template <typename Functor>
__global__ void touch1(
  void* out, const void* in,
  uint64_t t0_offset_inn,
  uint64_t t0_offset_out,
  uint64_t t0_size,
  uint64_t t0_d_inn,
  uint64_t t0_d_out,
  Functor f, int dtype_info)
{
  uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t inIndex = t0_offset_inn + index;
  uint64_t outIndex = t0_offset_out + index;

  if(index<t0_size){
    if(dtype_info==0){
      f(((__half*)out)[outIndex],((__half*)in)[inIndex]);
    }else if(dtype_info==1){
      f(((float*)out)[outIndex],((float*)in)[inIndex]);
    }else if(dtype_info==2){
      f(((double*)out)[outIndex],((double*)in)[inIndex]);
    }else if(dtype_info==3){
      f(((cuFloatComplex*)out)[outIndex],((cuFloatComplex*)in)[inIndex]);    
    }
  }
}

template <typename Functor>
__global__ void touch2(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out,
  uint64_t t0_size, uint64_t t1_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  Functor f, int dtype_info)
{
  uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t inRow = t0_offset_inn + row;
  uint64_t inCol = t1_offset_inn + col;
  uint64_t outRow = t0_offset_out + row;
  uint64_t outCol = t1_offset_out + col;

  if (row < t0_size && col < t1_size) {
    uint64_t inIndex = inRow * t1_d_inn + inCol;
    uint64_t outIndex = outRow * t1_d_out + outCol;
    if(dtype_info==0){
      f(((__half*)out)[outIndex],((__half*)in)[inIndex]);
    }else if(dtype_info==1){
      f(((float*)out)[outIndex],((float*)in)[inIndex]);
    }else if(dtype_info==2){
      f(((double*)out)[outIndex],((double*)in)[inIndex]);
    }else if(dtype_info==3){
      f(((cuFloatComplex*)out)[outIndex],((cuFloatComplex*)in)[inIndex]);
    }
  }
}

template <typename Functor>
__global__ void touch3(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  Functor f, int dtype_info)
{
  uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;

  uint64_t inX = t0_offset_inn + xDim;
  uint64_t inY = t1_offset_inn + yDim;
  uint64_t inZ = t2_offset_inn + zDim;
  uint64_t outX = t0_offset_out + xDim;
  uint64_t outY = t1_offset_out + yDim;
  uint64_t outZ = t2_offset_out + zDim;

  if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
    uint64_t inIndex = inX * t1_d_inn *t2_d_inn+ inY*t2_d_inn+inZ;
    uint64_t outIndex = outX * t1_d_out *t2_d_out+ outY*t2_d_out+outZ;
    if(dtype_info==0){
      f(((__half*)out)[outIndex],((__half*)in)[inIndex]);
    }else if(dtype_info==1){
      f(((float*)out)[outIndex],((float*)in)[inIndex]);
    }else if(dtype_info==2){
      f(((double*)out)[outIndex],((double*)in)[inIndex]);
    }else if(dtype_info==3){
      f(((cuFloatComplex*)out)[outIndex],((cuFloatComplex*)in)[inIndex]);
    }
  }
}

template <typename Functor>
__global__ void touch4(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn, uint64_t t3_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out, uint64_t t3_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size, uint64_t t3_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn,uint64_t t2_d_out,
  uint64_t t3_d_inn,uint64_t t3_d_out,
  Functor f, int dtype_info)
{
  uint64_t xDim = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t yDim = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t zDim = blockIdx.z * blockDim.z + threadIdx.z;

  uint64_t inX = t0_offset_inn + xDim;
  uint64_t inY = t1_offset_inn + yDim;
  uint64_t inZ = t2_offset_inn + zDim;
  uint64_t outX = t0_offset_out + xDim;
  uint64_t outY = t1_offset_out + yDim;
  uint64_t outZ = t2_offset_out + zDim;

  if (xDim<t0_size&&yDim<t1_size&&zDim<t2_size) {
    for(uint64_t wDim = 0; wDim < t3_size; wDim++){
      uint64_t inW = t3_offset_inn + wDim;
      uint64_t outW = t3_offset_out + wDim;

      uint64_t inIndex =
        inX * t1_d_inn * t2_d_inn * t3_d_inn +
        inY * t2_d_inn * t3_d_inn +
        inZ * t3_d_inn + inW;
      uint64_t outIndex =
        outX * t1_d_out * t2_d_out * t3_d_out +
        outY * t2_d_out * t3_d_out +
        outZ * t3_d_out + outW;
      if(dtype_info==0){
        f(((__half*)out)[outIndex],((__half*)in)[inIndex]);
      }else if(dtype_info==1){
        f(((float*)out)[outIndex],((float*)in)[inIndex]);
      }else if(dtype_info==2){
        f(((double*)out)[outIndex],((double*)in)[inIndex]);
      }else if(dtype_info==3){
        f(((cuFloatComplex*)out)[outIndex],((cuFloatComplex*)in)[inIndex]);
      }
    }
  }
}

void touch1_dispatch(
  void* out, const void* in,
  uint64_t t0_offset_inn,
  uint64_t t0_offset_out,
  uint64_t t0_size,
  uint64_t t0_d_inn,
  uint64_t t0_d_out,
  cudaStream_t stream,
  int choice, int dtype_info)
{
  dim3 blockSize(256);
  dim3 gridSize((t0_size + blockSize.x - 1) / blockSize.x);


  if(choice==0) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorCopy(),dtype_info);
  }
  else if(choice==1) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorAdd(),dtype_info);
  }
  else if(choice==2) {
  //  touch1<<<gridSize, blockSize,0,stream>>>(
  //    out, in,
  //    t0_offset_inn,
  //    t0_offset_out,
  //    t0_size,
  //    t0_d_inn,
  //    t0_d_out,
  //    FunctorMul(),dtype_info);
  }
  else if(choice==3) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorMin(),dtype_info);
  }
  else if(choice==4) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorMax(),dtype_info);
  }
}

void touch2_dispatch(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out,
  uint64_t t0_size, uint64_t t1_size,
  uint64_t t1_d_inn,
  uint64_t t1_d_out,
  cudaStream_t stream,
  int choice, int dtype_info)
{
  dim3 blockSize(16, 16);
  dim3 gridSize((t1_size + blockSize.x - 1) / blockSize.x,
                (t0_size + blockSize.y - 1) / blockSize.y);

  if(choice==0) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorCopy(),dtype_info);
  }
  else if(choice==1) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorAdd(),dtype_info);
  }
  else if(choice==2) {
  //  touch2<<<gridSize, blockSize,0,stream>>>(
  //    out, in,
  //    t0_offset_inn, t1_offset_inn,
  //    t0_offset_out, t1_offset_out,
  //    t0_size, t1_size,
  //    t1_d_inn, t1_d_out,
  //    FunctorMul(),dtype_info);
  }
  else if(choice==3) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorMin(),dtype_info);
  }
  else if(choice==4) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorMax(),dtype_info);
  }
}

void touch3_dispatch(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  cudaStream_t stream,
  int choice, int dtype_info)
{
  dim3 blockSize(8, 8, 8);
  dim3 gridSize((t0_size + blockSize.x - 1) / blockSize.x,
                (t1_size + blockSize.y - 1) / blockSize.y,
                (t2_size + blockSize.z - 1) / blockSize.z);

  if(choice==0) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorCopy(),dtype_info);
  }
  else if(choice==1) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorAdd(),dtype_info);
  }
  else if(choice==2) {
  //  touch3<<<gridSize, blockSize,0,stream>>>(
  //    out, in,
  //    t0_offset_inn, t1_offset_inn, t2_offset_inn,
  //    t0_offset_out, t1_offset_out, t2_offset_out,
  //    t0_size, t1_size, t2_size,
  //    t1_d_inn, t1_d_out,
  //    t2_d_inn, t2_d_out,
  //    FunctorMul(),dtype_info);
  }
  else if(choice==3) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorMin(),dtype_info);
  }
  else if(choice==4) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorMax(),dtype_info);
  }
}

void touch4_dispatch(
  void* out, const void* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn, uint64_t t3_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out, uint64_t t3_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size, uint64_t t3_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  uint64_t t3_d_inn, uint64_t t3_d_out,
  cudaStream_t stream,
  int choice, int dtype_info)
{
  dim3 blockSize(8, 8, 8);
  dim3 gridSize((t0_size + blockSize.x - 1) / blockSize.x,
                (t1_size + blockSize.y - 1) / blockSize.y,
                (t2_size + blockSize.z - 1) / blockSize.z);

  if(choice==0) {
    touch4<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
      t1_offset_out, t2_offset_out, t3_offset_out,
      t0_size, t1_size, t2_size, t3_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      t3_d_inn, t3_d_out,
      FunctorCopy(),dtype_info);
  }
  else if(choice==1) {
    touch4<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
      t1_offset_out, t2_offset_out, t3_offset_out,
      t0_size, t1_size, t2_size, t3_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      t3_d_inn, t3_d_out,
      FunctorAdd(),dtype_info);
  }
  else if(choice==2) {
  //  touch4<<<gridSize, blockSize,0,stream>>>(
  //    out, in,
  //    t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
  //    t1_offset_out, t2_offset_out, t3_offset_out,
  //    t0_size, t1_size, t2_size, t3_size,
  //    t1_d_inn, t1_d_out,
  //    t2_d_inn, t2_d_out,
  //    t3_d_inn, t3_d_out,
  //    FunctorMul(),dtype_info);
  }
  else if(choice==3) {
    touch4<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
      t1_offset_out, t2_offset_out, t3_offset_out,
      t0_size, t1_size, t2_size, t3_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      t3_d_inn, t3_d_out,
      FunctorMin(),dtype_info);
  }
  else if(choice==4) {
    touch4<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
      t1_offset_out, t2_offset_out, t3_offset_out,
      t0_size, t1_size, t2_size, t3_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      t3_d_inn, t3_d_out,
      FunctorMax(),dtype_info);
  }
}

__global__ void power(const float* in, float* out, uint64_t size, double pow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = powf(in[idx], pow);
  }
}

void elementwise_power(float* out, const float* in,
cudaStream_t stream, double pow, uint64_t size){
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  power<<<numBlocks, blockSize>>>(in, out, size, pow);

}

__global__ void increment_scale(const float* in, float* out,
uint64_t size, float scale, float increment) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    out[idx] = scale * in[idx] + increment;
  }
}

void scale_and_increment(float* out, const float* in,
cudaStream_t stream, float scale, float increment, uint64_t size){
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  increment_scale<<<numBlocks, blockSize>>>(in, out, size, scale, increment);

}

__global__ void custom_elementwise(const __half* in, __half* out,
uint64_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {

    __half x = in[idx];
    __half y = out[idx];
    float x0 = __half2float(in[idx]);
    __hmul(x,y);
    __hadd(x,y);
    __hgt(x,y);
    __hlt(x,y);
    out[idx] = __float2half(x0*powf(1.0f+expf(-1.0f*x0),-1.0f));
 }
}

void custom_elementwise_1(void* out, const void* in,
cudaStream_t stream, uint64_t size){
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;
  custom_elementwise<<<numBlocks, blockSize>>>((__half*)in, (__half*)out, size);

}

// for row in range(nrow):
//   for column in range(ncol):
//     ret[row, column] = lower if (row - start >= column) else upper

__global__ void fill_lowerTri(
  void* mem, uint64_t nrow, uint64_t ncol, uint64_t start,
  uint64_t lower, uint64_t upper, int dtype_info)
{
  uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("row: %lu, col: %lu\n", row, col);

  if (row < nrow && col < ncol) {
    uint64_t index = row * ncol + col;

    // printf("row: %lu, row-start: %lu, col: %lu, lower: %d\n", row, row-start, col, row - start >= col);
    if(dtype_info==0){
      ((__half*)mem)[index] = row - start >= col ? *(reinterpret_cast<__half const*>(&lower)) 
                                                    : *(reinterpret_cast<__half const*>(&upper));
    }else if(dtype_info==1){
      ((float*)mem)[index] = row - start >= col ? *(reinterpret_cast<float const*>(&lower)) 
                                                : *(reinterpret_cast<float const*>(&upper));
    }else if(dtype_info==2){
      ((double*)mem)[index] = row - start >= col ? *(reinterpret_cast<double const*>(&lower)) 
                                                : *(reinterpret_cast<double const*>(&upper));
    }else if (dtype_info==3){
      ((cuFloatComplex*)mem)[index] = row - start >= col ? *(reinterpret_cast<cuFloatComplex const*>(&lower)) 
                                                        : *(reinterpret_cast<cuFloatComplex const*>(&upper));
    }
  } 
}

void fillTri_dispatch(void* mem, uint64_t nrow, uint64_t ncol, uint64_t start,
  uint64_t lower, uint64_t upper, cudaStream_t stream, int dtype_info)
{
  dim3 blockSize(nrow, ncol);
  dim3 gridSize((ncol + blockSize.x - 1) / blockSize.x,
                (nrow + blockSize.y - 1) / blockSize.y);

  fill_lowerTri<<<gridSize, blockSize,0,stream>>>
    (mem, nrow, ncol, start, lower, upper, dtype_info);
}

__global__ void fill_constant(void* mem, uint64_t nelem, uint64_t value, int dtype_info){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index<nelem){
    if(dtype_info==0){
      ((__half*)mem)[index] = *(reinterpret_cast<__half const*>(&value));
    }else if(dtype_info==1){
      ((float*)mem)[index] = *(reinterpret_cast<float const*>(&value));
    }else if(dtype_info==2){
      ((double*)mem)[index] = *(reinterpret_cast<double const*>(&value));
    }
    else if(dtype_info==3){
      ((cuFloatComplex*)mem)[index] = *(reinterpret_cast<cuFloatComplex const*>(&value));
    }
    else{
      printf("ERROR: CUDA_KERNEL: dtype_info not supported\n");
    }
  }
}

void fill_constant_dispatch(void* mem, uint64_t nelem, uint64_t value,
  cudaStream_t stream, int dtype_info){
  int blockSize = 256;
  int gridSize = (nelem + blockSize - 1) / blockSize;
  // printf("nelem: %lu value %f\n", nelem, *((float*)value));
  // printf("reinterpret cast value %f\n", *reinterpret_cast<float const*>(value));
  // printf("reinterpret cast value 2 %f\n", *reinterpret_cast<float const*>(&value));
  fill_constant<<<gridSize, blockSize,0,stream>>>
    (mem, nelem, value, dtype_info);
}

// compare mem[i, j] with compare [i], if mem[i, j] == compare[i], assign out[i, j] = value_true
// else assign out[i, j] = value_false
__global__ void conditional_assignment(void* out, void const* mem, uint64_t rows, uint64_t columns,
 void const* compare, uint64_t value_true, uint64_t value_false, int dtype_info){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index<rows * columns){
    uint64_t row = index / columns;
    // uint64_t col = index % columns;
    uint64_t compare_index = row;
    if(dtype_info==0){
      ((__half*)out)[index] = ((__half*)mem)[index] == ((__half*)compare)[compare_index] ? 
        *(reinterpret_cast<__half const*>(&value_true)) : *(reinterpret_cast<__half const*>(&value_false));
    }else if(dtype_info==1){
      ((float*)out)[index] = ((float*)mem)[index] == ((float*)compare)[compare_index] ? 
        *(reinterpret_cast<float const*>(&value_true)) : *(reinterpret_cast<float const*>(&value_false));
    }else if(dtype_info==2){
      ((double*)out)[index] = ((double*)mem)[index] == ((double*)compare)[compare_index] ? 
         *(reinterpret_cast<double const*>(&value_true)) : *(reinterpret_cast<double const*>(&value_false));
    }
    else if(dtype_info==3){
      cuFloatComplex* c = (cuFloatComplex*)compare;
      cuFloatComplex* m = (cuFloatComplex*)mem;
      if (c[compare_index].x == m[index].x && c[compare_index].y == m[index].y){
        ((cuFloatComplex*)out)[index] = *(reinterpret_cast<cuFloatComplex const*>(&value_true));
      }else{
        ((cuFloatComplex*)out)[index] = *(reinterpret_cast<cuFloatComplex const*>(&value_false));
      }
    }
    else{
      printf("ERROR: CUDA_KERNEL: dtype_info not supported\n");
    }
  }
 }

void conditional_assignment_dispatch(void* out, void const* mem, uint64_t rows, uint64_t columns,
  void const* compare, uint64_t value_true, uint64_t value_false, cudaStream_t stream, int dtype_info){
  int blockSize = 256;
  int gridSize = (rows * columns + blockSize - 1) / blockSize;
  conditional_assignment<<<gridSize, blockSize,0,stream>>>
    (out, mem, rows, columns, compare, value_true, value_false, dtype_info);
}


__global__ void special_elementwise_mul(void* out, uint64_t a, uint64_t b, const void* x, 
  const void* y, cudaStream_t stream, int dtype_info){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  // fill a x b matrix with two vectors X and Y
  // for i in size(a):
  // for j in size(b):
  //   Out[i,j] = f(X[i], Y[i])
  if (index < a*b){
    uint64_t row = index / b;
    // uint64_t col = index % b;
    if(dtype_info==0){
      ((__half*)out)[index] = __hmul(((__half*)x)[row], ((__half*)y)[row]);
    }else if(dtype_info==1){
      ((float*)out)[index] = ((float*)x)[row] * ((float*)y)[row];
    }else if(dtype_info==2){
      ((double*)out)[index] = ((double*)x)[row] * ((double*)y)[row];
    }
    else if(dtype_info==3){
      ((cuFloatComplex*)out)[index] = cuCmulf(((cuFloatComplex*)x)[row], ((cuFloatComplex*)y)[row]);
    }
    else{
      printf("ERROR: CUDA_KERNEL: dtype_info not supported\n");
    }
  }
}

void special_elementwise_mul_dispatch(void* out, uint64_t a, uint64_t b, const void* x, 
  const void* y, cudaStream_t stream, int dtype_info){
  int blockSize = 256;
  int gridSize = (a*b + blockSize - 1) / blockSize;
  special_elementwise_mul<<<gridSize, blockSize,0,stream>>>
    (out, a, b, x, y, stream, dtype_info);
}

// ab -> a with max reduction
__global__ void special_reduction_max(void* out, uint64_t a, uint64_t b, const void* x, 
  cudaStream_t stream){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  // out[a] = max(x[a][b]) for all b in size(b)
  if (index < a){
    for (uint64_t i = 0; i < b; i++){
      if (i == 0){
        ((float*)out)[index] = ((float*)x)[index * b + i];
      }
      else{
        ((float*)out)[index] = fmaxf(((float*)out)[index], ((float*)x)[index * b + i]);
      }
    }
  }
}

void special_reduction_max_dispatch(void* out, uint64_t a, uint64_t b, const void* x,
  cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (a + blockSize - 1) / blockSize;
  special_reduction_max<<<gridSize, blockSize,0,stream>>>
    (out, a, b, x, stream);
}

// ab -> a with add reduction
__global__ void special_reduction_sum(void* out, uint64_t a, uint64_t b, const void* x, 
  cudaStream_t stream){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  // out[a] = sum(x[a][b]) for all b in size(b)
  if (index < a){
    for (uint64_t i = 0; i < b; i++){
      if (i == 0){
        ((float*)out)[index] = ((float*)x)[index * b + i];
      }
      else{
        ((float*)out)[index] += ((float*)x)[index * b + i];
      }
    }
  }
}

void special_reduction_sum_dispatch(void* out, uint64_t a, uint64_t b, const void* x,
  cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (a + blockSize - 1) / blockSize;
  special_reduction_sum<<<gridSize, blockSize,0,stream>>>
    (out, a, b, x, stream);
}

// ab -> a with add reduction with negated input x
__global__ void special_reduction_negateSum(void* out, uint64_t a, uint64_t b, const void* x, 
  cudaStream_t stream){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  // out[a] = sum(x[a][b]) for all b in size(b)
  if (index < a){
    for (uint64_t i = 0; i < b; i++){
      if (i == 0){
        ((float*)out)[index] = -((float*)x)[index * b + i];
      }
      else{
        ((float*)out)[index] -= ((float*)x)[index * b + i];
      }
    }
  }
}

void special_reduction_negateSum_dispatch(void* out, uint64_t a, uint64_t b, const void* x,
  cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (a + blockSize - 1) / blockSize;
  special_reduction_negateSum<<<gridSize, blockSize,0,stream>>>
    (out, a, b, x, stream);
}

// + ab,a->a | exp[*[constant{f32|0.0883883},+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]]
// out[a] = 0
// for b in range(nb):
//   out[a] += exp(s*(lhs[a,b] - rhs[b])) 
__global__ void softmax_v3_reduction(void* out, const void* lhs, const void* rhs, uint64_t rows, uint64_t cols,
  float constant){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows){
    float sum = 0;
    for (uint64_t i = 0; i < cols; i++){
      float lhs_val = ((float*)lhs)[index * cols + i];
      float rhs_val = ((float*)rhs)[i];
      sum += expf(constant * (lhs_val - rhs_val));
    }
    ((float*)out)[index] = sum;
  }
}

void softmax_v3_reduction_dispatch(void* out, const void* lhs, const void* rhs, uint64_t rows, uint64_t cols,
  float constant, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows * cols + blockSize - 1) / blockSize;
  softmax_v3_reduction<<<gridSize, blockSize,0,stream>>>
    (out, lhs, rhs, rows, cols, constant);
}

// ab,a,a->ab | 
// *[exp[*[constant{f32|0.0883883},+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]]],power{-1}[hole|f32@2]]
__global__ void softmax_v3_elementwise(void* out, const void* lhs, const void* mid,
  const void* rhs, uint64_t rows, uint64_t cols, float constant){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows * cols){
    uint64_t row = index / cols;
    // uint64_t col = index % cols;
    float lhs_val = ((float*)lhs)[index];
    float rhs_val = ((float*)rhs)[row];
    float mid_val = ((float*)mid)[row];
    float exp_val = expf(constant * (lhs_val - mid_val));
    ((float*)out)[index] = fdividef(exp_val, rhs_val);
  }
}

void softmax_v3_elementwise_dispatch(void *out, const void *lhs, const void* mid,
  const void *rhs, uint64_t rows, 
  uint64_t cols, float constant, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows * cols + blockSize - 1) / blockSize;
  softmax_v3_elementwise<<<gridSize, blockSize,0,stream>>>
    (out, lhs, mid, rhs, rows, cols, constant);
}

// + ab,ab,a->a
// *[hole|f32@0,*[hole|f32@1,*[constant{f32|-1},power{-2}[hole|f32@2]]]]
// out[a] = Sum(lhs[a, b] )
__global__ void large_workspace_1(void* out, const void* lhs, const void* mid,
  const void* rhs, uint64_t rows, uint64_t cols){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows){
    float sum = 0;
    for (uint64_t i = 0; i < cols; i++){
      float lhs_val = ((float*)lhs)[index * cols + i];
      float mid_val = ((float*)mid)[index * cols + i];
      float rhs_val = ((float*)rhs)[i];
      sum += fdividef(lhs_val * mid_val , (rhs_val * rhs_val) * -1.0f);
    }
    ((float*)out)[index] = sum;
  }
}

void large_workspace_1_dispatch(void* out, const void* lhs, const void* mid, const void* rhs,
  uint64_t rows, uint64_t cols, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows * cols + blockSize - 1) / blockSize;
  large_workspace_1<<<gridSize, blockSize,0,stream>>>
    (out, lhs, mid, rhs, rows, cols);
}

// ab,a,a->ab | *[*[hole|f32@0,power{-1}[hole|f32@1]],hole|f32@2]
__global__ void large_workspace_2(void* out, const void* lhs, const void* mid,
  const void* rhs, uint64_t rows, uint64_t cols){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows * cols){
    uint64_t row = index / cols;
    // uint64_t col = index % cols;
    float lhs_val = ((float*)lhs)[index];
    float mid_val = ((float*)mid)[row];
    float rhs_val = ((float*)rhs)[row];
    ((float*)out)[index] = fdividef(lhs_val, mid_val) * rhs_val;
  }
}

void large_workspace_2_dispatch(void* out, const void* lhs, const void* mid, const void* rhs,
  uint64_t rows, uint64_t cols, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows * cols + blockSize - 1) / blockSize;
  large_workspace_2<<<gridSize, blockSize,0,stream>>>
    (out, lhs, mid, rhs, rows, cols);
}

// a,a->a | *[hole|f32@0,+[power{-1}[+[constant{f32|1},exp[*[constant{f32|-1},hole|f32@1]]]],*[hole|f32@1,*[constant{f32|-1},*[power{-2}[+[constant{f32|1},exp[*[constant{f32|-1},hole|f32@1]]]],
// *[constant{f32|-1},exp[*[constant{f32|-1},hole|f32@1]]]]]]]]

__global__ void large_workspace_3(void* out, const void* lhs, uint64_t rows, uint64_t cols){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows){
    float sum = 0;
    for (uint64_t i = 0; i < cols; i++){
      float lhs_val = ((float*)lhs)[index * cols + i];
      sum += (1.0f + expf(-1.0f * lhs_val)) * (1.0f - expf(-1.0f * lhs_val)) * 
        (1.0f - expf(-2.0f * (1.0f + expf(-1.0f * lhs_val))) * 
        (1.0f - expf(-1.0f * lhs_val)));
    }
    ((float*)out)[index] = sum;
  }
}

// haven't implemented it yet
void large_workspace_3_dispatch(void* out, const void* lhs, uint64_t rows, 
  uint64_t cols, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows + blockSize - 1) / blockSize;
  large_workspace_1<<<gridSize, blockSize,0,stream>>>
    (out, lhs, lhs, lhs, rows, cols);
}

// +[*[constant{f32|0.5},hole|f32@0],*[constant{f32|-500},hole|f32@1]]
__global__ void large_workspace_4(void* out, const void* lhs, const void* rhs, 
  uint64_t rows, float f1, float f2){
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<rows){
    float lhs_val = ((float*)lhs)[index];
    float rhs_val = ((float*)rhs)[index];
    ((float*)out)[index] = f1 * lhs_val + f2 * rhs_val;
  }
}

void large_workspace_4_dispatch(void* out, const void* lhs, const void* rhs, 
  uint64_t rows, float f1, float f2, cudaStream_t stream){
  int blockSize = 256;
  int gridSize = (rows + blockSize - 1) / blockSize;
  large_workspace_4<<<gridSize, blockSize,0,stream>>>
    (out, lhs, rhs, rows, f1, f2);
}
