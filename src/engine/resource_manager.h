#pragma once
#include "../base/setup.h"

#ifdef CPU_EXEC
#include "cpu/workspace_manager.h"
#endif

struct global_buffer_t {
  struct desc_t {};

  struct resource_t {
    void* ptr;
  };

  optional<resource_t> try_to_acquire(desc_t){
    return resource_t{ .ptr = ptr };
  }

  void release(resource_t) {}

  void* ptr;
};

struct resource_manager_t {
  using desc_unit_t = std::variant<
#ifdef CPU_EXEC
    cpu_workspace_manager_t::desc_t,
#endif
    global_buffer_t::desc_t
  >;
  using resource_unit_t = std::variant<
#ifdef CPU_EXEC
    cpu_workspace_manager_t::resource_t,
#endif
    global_buffer_t::resource_t
  >;

  using desc_t = vector<desc_unit_t>;

  using resource_t = vector<resource_unit_t>;

  optional<resource_unit_t> try_to_acquire_unit(desc_unit_t const& unit);
  optional<resource_t> try_to_acquire(desc_t const& desc);

  void release_unit(resource_unit_t const& resource_unit);
  void release(resource_t resource);

private:
  global_buffer_t* global_buffer;
#ifdef CPU_EXEC
  cpu_workspace_manager_t* cpu_workspace_manager;
#endif
};


