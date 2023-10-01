#pragma once
#include "../../base/setup.h"

#include "utility.h"

#include <cstdint>
#include <cuda_runtime.h>

#include <mutex>

struct workspace_manager_t {
  ~workspace_manager_t();
 
  tuple<void*, uint64_t> borrow_workspace(int gpu, uint64_t size);

  void return_workspace(int gpu, void* mem, uint64_t sz);

  std::mutex m;

  vector<vector<tuple<void*, uint64_t>>> data;

  struct resource_t {
    int which;
    void* ptr;
    uint64_t size;

    tuple<void*, uint64_t> as_tuple() const { return {ptr,size}; }
  };

  struct desc_t {
    int which;
    void* ptr;
    uint64_t size;
  };

  optional<resource_t> try_to_acquire(desc_t desc){
    auto which = desc.which;
    auto size = desc.size;

    auto [ptr, sz] = borrow_workspace(which, size);

    return resource_t{ .which = which, .ptr = ptr, .size = sz };
  }

  void release(resource_t resource) {
    return_workspace(resource.which, resource.ptr, resource.size);
  }
};
