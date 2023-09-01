#include "gpu_communicator.h"

gpu_comm_t::gpu_comm_t() {

}

gpu_comm_t::~gpu_comm_t() {

}

void gpu_comm_t::send(void* dst, void* src, size_t size){
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
}

// RIGHT NOW SEND AND RECV ARE THE SAME
void gpu_comm_t::recv(void* dst, void* src, size_t size){
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed");
  }
}