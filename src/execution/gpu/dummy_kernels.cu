#include "dummy_kernels.h"

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
__global__ void dummy(float* out, const float* in, Functor f) {
    // do nothing since it's a dummy
}

void dummy_dispatch(float* out, const float* in,cudaStream_t stream){
    // call the dummy
    dim3 blockSize(0); 
    dim3 gridSize(0);
    dummy<<<gridSize, blockSize, 0, stream>>>(out, in, FunctorNone());
}
