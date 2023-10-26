#pragma once
#include "../../base/setup.h"
#include "../resource_manager.h"
#include "utility.h"
#include <cuda_runtime.h>
#include <mutex>
#include <condition_variable>

// stream pool should be similar to thread pool
// except it has a vector of streams for each gpu

struct streampool_resource_t {
  streampool_resource_t(cuda_stream_t s, int dev): stream(s), device(dev){}

  int device;
  cuda_stream_t stream;
};

struct streampool_desc_t {
  int device;
};

struct streampool_t : rm_template_t<
      streampool_desc_t,
      streampool_resource_t>
{
  streampool_t(int num_streams_per_gpu, int num_gpus);

  // since we are using cuda_stream_t, streams should be destroyed when
  // out of scope
  ~streampool_t();

private:
  std::mutex m;

  vector<vector<cuda_stream_t>> stream_pools;

  optional<streampool_resource_t> try_to_acquire_impl(int dev);

  void release_impl(streampool_resource_t const& resource);
};

