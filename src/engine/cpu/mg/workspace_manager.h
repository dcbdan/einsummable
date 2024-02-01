#pragma once
#include "../../../base/setup.h"

#include "../../../base/buffer.h"

#include "../../resource_manager.h"

struct cpu_workspace_resource_t {
  int which;
  void* ptr;
  uint64_t size;

  tuple<void*, uint64_t> as_tuple() const { return {ptr,size}; }
};

struct cpu_workspace_manager_t
  : rm_template_t<uint64_t, cpu_workspace_resource_t>
{
  cpu_workspace_manager_t() {}

private:
  std::mutex m_items;
  vector<tuple<bool,buffer_t>> items;

  // must be called with lock around items
  int acquire(uint64_t min_size);

  void release(int which);
private:
  optional<cpu_workspace_resource_t>
  try_to_acquire_impl(uint64_t const& size);

  void release_impl(cpu_workspace_resource_t const& resource) {
    release(resource.which);
  }
};


