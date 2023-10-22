#pragma once
#include "../../base/setup.h"

#include "../resource_manager.h"
#include "storage.h"

struct gpu_storage_resource_t {
  gpu_storage_t* ptr;
};

// This manager is really just a placeholder for when it may be necc
// to add more manager-like facilities to using cpu_storage_t.
struct gpu_storage_manager_t
  : rm_template_t<unit_t, gpu_storage_resource_t>
{
  gpu_storage_manager_t(gpu_storage_resource_t* p)
    : ptr(p->ptr)
  {}

  static desc_ptr_t make_desc() { return rm_template_t::make_desc(unit_t{}); }

private:
  optional<gpu_storage_resource_t> try_to_acquire_impl(unit_t const&){
    return gpu_storage_resource_t{ .ptr = ptr };
  }

  void release_impl(gpu_storage_resource_t const&) {}

  gpu_storage_t* ptr;
};

