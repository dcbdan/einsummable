#pragma once
#include "../../../base/setup.h"

#include "../../resource_manager.h"
#include "storage.h"

struct cpu_storage_resource_t {
  cpu_storage_t* ptr;
};

// This manager is really just a placeholder for when it may be necc
// to add more manager-like facilities to using cpu_storage_t.
struct cpu_storage_manager_t
  : rm_template_t<unit_t, cpu_storage_resource_t>
{
  cpu_storage_manager_t(cpu_storage_t* p)
    : ptr(p)
  {}

  static desc_ptr_t make_desc() { return rm_template_t::make_desc(unit_t{}); }

private:
  optional<cpu_storage_resource_t> try_to_acquire_impl(unit_t const&){
    return cpu_storage_resource_t{ .ptr = ptr };
  }

  void release_impl(cpu_storage_resource_t const&) {}

  cpu_storage_t* ptr;
};

