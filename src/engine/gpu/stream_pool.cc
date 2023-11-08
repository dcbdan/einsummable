#include "stream_pool.h"
#include "utility.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

streampool_t::streampool_t(int num_streams_per_gpu, int num_gpus){
  DOUT("Creating stream pool...");
  for(int i = 0; i < num_gpus; i++){
    std::queue<cudaStream_t> streams_per_gpu;
    cudaSetDevice(i);
    for(int j = 0; j < num_streams_per_gpu; j++){
      cudaStream_t stream = cuda_create_stream();
      streams_per_gpu.push(stream);
    }
    stream_pools.push_back(streams_per_gpu);
  }
  // DOUT("Number of pools: " << stream_pools.size());
  // DOUT("Number of streams in pool " << 0 << ": " << stream_pools[0].size());
}

streampool_t::streampool_t(){
}

void streampool_t::initialize(int num_streams_per_gpu, int num_gpus){
  DOUT("Creating stream pool...");
  for(int i = 0; i < num_gpus; i++){
    std::queue<cudaStream_t> streams_per_gpu;
    cudaSetDevice(i);
    for(int j = 0; j < num_streams_per_gpu; j++){
      cudaStream_t stream = cuda_create_stream();
      streams_per_gpu.push(stream);
    }
    stream_pools.push_back(streams_per_gpu);
  }
  // DOUT("Number of pools: " << stream_pools.size());
  // DOUT("Number of streams in pool " << 0 << ": " << stream_pools[0].size());
}

streampool_t::~streampool_t(){
  std::unique_lock lk(m);
  DOUT("Destroying stream pool...");
  for (int i = 0; i < stream_pools.size(); i++) {
    cudaSetDevice(i);
    while (!stream_pools[i].empty()) {
      cudaStream_t stream = stream_pools[i].front();
      cudaStreamDestroy(stream);
      stream_pools[i].pop();
    }
  }
}

optional<streampool_resource_t> streampool_t::
  try_to_acquire_impl(streampool_desc_t const& desc){
  
  std::unique_lock lk(m);
  int dev = desc.device;
  if(stream_pools[dev].size() == 0){
    return std::nullopt;
  }
  cudaStream_t stream = stream_pools[dev].front();
  stream_pools[dev].pop();
  DOUT("STREAM POOL: Got stream on device " << dev << " from pool.");
  return streampool_resource_t(stream, dev);
}

void streampool_t::release_impl(streampool_resource_t const& resource){
  std::unique_lock lk(m);
  stream_pools[resource.device].push(resource.stream);
}

