#include "cuda_kernels.h"

struct FunctorNone {
  __device__ float operator()(const float& a, const float& b) const {
    return b;
  }
};

struct FunctorAdd {
  __device__ float operator()(const float& a, const float& b) const {
    return a+b;
  }
};

struct FunctorMul {
  __device__ float operator()(const float& a, const float& b) const {
    return a*b;
  }
};

struct FunctorMin {
  __device__ float operator()(const float& a, const float& b) const {
    return fminf(a,b);
  }
};

struct FunctorMax {
  __device__ float operator()(const float& a, const float& b) const {
    return fmaxf(a,b);
  }
};

template <typename Functor>
__global__ void touch1(
  float* out, const float* in,
  uint64_t t0_offset_inn,
  uint64_t t0_offset_out,
  uint64_t t0_size,
  uint64_t t0_d_inn,
  uint64_t t0_d_out,
  Functor f)
{
  uint64_t index =  blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t inIndex = t0_offset_inn + index;
  uint64_t outIndex = t0_offset_out + index;

  if(index<t0_size){
    out[outIndex] = f(out[outIndex],in[inIndex]);
  }
}

template <typename Functor>
__global__ void touch2(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out,
  uint64_t t0_size, uint64_t t1_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  Functor f)
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
    out[outIndex] = f(out[outIndex],in[inIndex]);
  }
}

template <typename Functor>
__global__ void touch3(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  Functor f)
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
    out[outIndex] = f(out[outIndex],in[inIndex]);
  }
}

template <typename Functor>
__global__ void touch4(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn, uint64_t t3_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out, uint64_t t3_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size, uint64_t t3_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn,uint64_t t2_d_out,
  uint64_t t3_d_inn,uint64_t t3_d_out,
  Functor f)
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
      out[outIndex] = f(out[outIndex],in[inIndex]);
    }
  }
}

void touch1_dispatch(
  float* out, const float* in,
  uint64_t t0_offset_inn,
  uint64_t t0_offset_out,
  uint64_t t0_size,
  uint64_t t0_d_inn,
  uint64_t t0_d_out,
  cudaStream_t stream,
  int choice)
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
      FunctorNone());
  }
  else if(choice==1) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorAdd());
  }
  else if(choice==2) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorMul());
  }
  else if(choice==3) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorMin());
  }
  else if(choice==4) {
    touch1<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn,
      t0_offset_out,
      t0_size,
      t0_d_inn,
      t0_d_out,
      FunctorMax());
  }
}

void touch2_dispatch(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out,
  uint64_t t0_size, uint64_t t1_size,
  uint64_t t1_d_inn,
  uint64_t t1_d_out,
  cudaStream_t stream,
  int choice)
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
      FunctorNone());
  }
  else if(choice==1) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorAdd());
  }
  else if(choice==2) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorMul());
  }
  else if(choice==3) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorMin());
  }
  else if(choice==4) {
    touch2<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn,
      t0_offset_out, t1_offset_out,
      t0_size, t1_size,
      t1_d_inn, t1_d_out,
      FunctorMax());
  }
}

void touch3_dispatch(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  cudaStream_t stream,
  int choice)
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
      FunctorNone());
  }
  else if(choice==1) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorAdd());
  }
  else if(choice==2) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorMul());
  }
  else if(choice==3) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorMin());
  }
  else if(choice==4) {
    touch3<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn,
      t0_offset_out, t1_offset_out, t2_offset_out,
      t0_size, t1_size, t2_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      FunctorMax());
  }
}

void touch4_dispatch(
  float* out, const float* in,
  uint64_t t0_offset_inn, uint64_t t1_offset_inn, uint64_t t2_offset_inn, uint64_t t3_offset_inn,
  uint64_t t0_offset_out, uint64_t t1_offset_out, uint64_t t2_offset_out, uint64_t t3_offset_out,
  uint64_t t0_size, uint64_t t1_size, uint64_t t2_size, uint64_t t3_size,
  uint64_t t1_d_inn, uint64_t t1_d_out,
  uint64_t t2_d_inn, uint64_t t2_d_out,
  uint64_t t3_d_inn, uint64_t t3_d_out,
  cudaStream_t stream,
  int choice)
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
      FunctorNone());
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
      FunctorAdd());
  }
  else if(choice==2) {
    touch4<<<gridSize, blockSize,0,stream>>>(
      out, in,
      t0_offset_inn, t1_offset_inn, t2_offset_inn, t3_offset_inn, t0_offset_out,
      t1_offset_out, t2_offset_out, t3_offset_out,
      t0_size, t1_size, t2_size, t3_size,
      t1_d_inn, t1_d_out,
      t2_d_inn, t2_d_out,
      t3_d_inn, t3_d_out,
      FunctorMul());
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
      FunctorMin());
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
      FunctorMax());
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
  powerKernel<<<numBlocks, blockSize>>>(in, out, size, pow);

}

