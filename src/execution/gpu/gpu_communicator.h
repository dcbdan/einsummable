#include "../../base/setup.h"
#include "cutensor.h"
#include <cuda_runtime.h>

// At this point we are doing single machine multi gpu
// So at this point all communications utilizes cudaMemCpy 
// since they do not go through the network

// TODO: extend the class when we get multiple machines 
struct gpu_comm_t {
  gpu_comm_t();
  ~gpu_comm_t();

  void recv(void* dst, void* src, size_t size, cudaStream_t stream);
  void send(void* dst, void* src, size_t size, cudaStream_t stream);
};