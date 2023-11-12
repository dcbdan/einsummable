#include "stream_pool.h"
#include "utility.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>

// ------------------------ stream pool ------------------------

streampool_t::streampool_t(int num_streams_per_gpu, int num_gpus){
  initialize(num_streams_per_gpu, num_gpus);
}

void streampool_t::initialize(int num_streams_per_gpu, int num_gpus){
  if(stream_pools.size() > 0) {
    throw std::runtime_error("the stream pool must have already been initialized");
  }

  DOUT("Creating stream pool...");
  for(int i = 0; i < num_gpus; i++){
    vector<cudaStream_t> streams_per_gpu;
    cudaSetDevice(i);
    for(int j = 0; j < num_streams_per_gpu; j++){
      cudaStream_t stream = cuda_create_stream();
      streams_per_gpu.push_back(stream);
    }
    stream_pools.push_back(streams_per_gpu);
  }
  // DOUT("Number of pools: " << stream_pools.size());
  // DOUT("Number of streams in pool " << 0 << ": " << stream_pools[0].size());
}

streampool_t::~streampool_t(){
  DOUT("Destroying stream pool...");
  for (int i = 0; i < stream_pools.size(); i++) {
    cudaSetDevice(i);
    for (auto& stream: stream_pools[i]) {
      cudaStreamDestroy(stream);
    }
  }
}

// ------------------------ stream pool manager ------------------------

streampool_manager_t::streampool_manager_t(streampool_t& streampool): stream_pools(streampool){
  DOUT("Creating stream pool manager...");
}

optional<streampool_resource_t> streampool_manager_t::
  try_to_acquire_impl(streampool_desc_t const& desc){
  
  std::unique_lock lk(m);
  int dev = desc.device;
  if(stream_pools.stream_pools[dev].size() == 0){
    return std::nullopt;
  }
  cudaStream_t stream = stream_pools.stream_pools[dev].back();
  stream_pools.stream_pools[dev].pop_back();

  return streampool_resource_t(stream, dev);
}

void streampool_manager_t::release_impl(streampool_resource_t const& resource){
  std::unique_lock lk(m);
  stream_pools.stream_pools[resource.device].push_back(resource.stream);
}

