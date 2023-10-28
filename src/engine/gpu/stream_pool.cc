#include "stream_pool.h"
#include <cuda_runtime_api.h>

streampool_t::streampool_t(int num_streams_per_gpu, int num_gpus){
  for(int i = 0; i < num_gpus; i++){
    vector<cuda_stream_t> streams_per_gpu;
    cudaSetDevice(i);
    for(int j = 0; j < num_streams_per_gpu; j++){
      cuda_stream_t stream;
      streams_per_gpu.push_back(stream);
    }
    stream_pools.push_back(streams_per_gpu);
  }
}

optional<streampool_resource_t> streampool_t::try_to_acquire_impl(int dev){
  std::unique_lock lk(m);
  if(stream_pools[dev].size() == 0){
    return std::nullopt;
  }
  cuda_stream_t stream = stream_pools[dev].back();
  stream_pools[dev].pop_back();
  return streampool_resource_t(stream, dev);
}

void streampool_t::release_impl(streampool_resource_t const& resource){
  std::unique_lock lk(m);
  stream_pools[resource.device].push_back(resource.stream);
}

