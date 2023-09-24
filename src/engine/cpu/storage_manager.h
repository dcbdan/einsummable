#pragma once
#include "../../base/setup.h"

#include "storage.h"

// This manager is really just a placeholder for when it may be necc
// to add more manager-like facilities to using cpu_storage_t.
struct cpu_storage_manager_t {
  struct desc_t {};

  struct resource_t {
    cpu_storage_t* ptr;
  };

  optional<resource_t> try_to_acquire(desc_t){
    return resource_t{ .ptr = ptr };
  }

  void release(resource_t) {}

  cpu_storage_t* ptr;
};

