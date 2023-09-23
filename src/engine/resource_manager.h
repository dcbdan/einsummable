#pragma once
#include "../base/setup.h"

#ifdef CPU_EXEC
#include "cpu/workspace_manager.h"
#endif

struct global_buffer_t {
  global_buffer_t(void* p)
    : ptr(p)
  {}

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

struct group_manager_t {
  struct desc_t {
    int group_id;
  };
  struct resource_t {
    int group_id;
    bool is_first;
  };

  optional<resource_t> try_to_acquire(desc_t desc);

  void release(resource_t resource);

private:
  std::mutex m;
  set<int> busy_groups;
  set<int> seen_groups;
};

struct resource_manager_t {
  using desc_unit_t = std::variant<
#ifdef CPU_EXEC
    cpu_workspace_manager_t::desc_t,
#endif
    global_buffer_t::desc_t,
    group_manager_t::desc_t
  >;
  using resource_unit_t = std::variant<
#ifdef CPU_EXEC
    cpu_workspace_manager_t::resource_t,
#endif
    global_buffer_t::resource_t,
    group_manager_t::resource_t
  >;

  using desc_t = vector<desc_unit_t>;

  using resource_t = vector<resource_unit_t>;

  optional<resource_unit_t> try_to_acquire_unit(desc_unit_t const& unit);
  optional<resource_t> try_to_acquire(desc_t const& desc);

  void release_unit(resource_unit_t const& resource_unit);
  void release(resource_t resource);

  //  TODO: decide if they should be pointers or references or
  //        shared pointers and how they should be set
#ifdef CPU_EXEC
  cpu_workspace_manager_t* cpu_workspace_manager;
#endif
  group_manager_t* group_manager;
  global_buffer_t* global_buffer;
};


