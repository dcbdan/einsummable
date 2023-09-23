#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

struct cpu_workspace_manager_t {
  struct desc_t {
    uint64_t size;
  };

  struct resource_t {
    int which;
    void* ptr;
    uint64_t size;

    tuple<void*, uint64_t> as_tuple() const { return {ptr,size}; }
  };

  optional<resource_t>
  try_to_acquire(desc_t desc);

  void release(resource_t resource) {
    release(resource.which);
  }

private:
  std::mutex m_items;
  vector<tuple<bool,buffer_t>> items;

  // must be called with lock around items
  int acquire(uint64_t min_size);

  void release(int which);
};


