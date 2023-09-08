#include "gpu_communicator.h"
#include <cuda_runtime_api.h>

gpu_comm_t::gpu_comm_t() {

}

gpu_comm_t::~gpu_comm_t() {

}

void gpu_comm_t::send(void* dst, void* src, size_t size, cudaStream_t stream){
  cudaError_t cudaError = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
  if (cudaError != cudaSuccess) {
    // print the error code and error string
    fprintf(stderr, "cudaMemcpy failed with error: %s\n", cudaGetErrorString(cudaError));
    throw std::runtime_error("cudaMemcpy failed");
  }
}

// RIGHT NOW SEND AND RECV ARE THE SAME
void gpu_comm_t::recv(void* dst, void* src, size_t size, cudaStream_t stream){
  cudaError_t cudaError = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
  if (cudaError != cudaSuccess) {
    // print the error code and error string
    fprintf(stderr, "cudaMemcpy failed with error: %s\n", cudaGetErrorString(cudaError));
    throw std::runtime_error("cudaMemcpy failed");
  }
}