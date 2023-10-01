#pragma once
#include "../base/setup.h"

#ifdef CPU_EXEC
#include "cpu/workspace_manager.h"
#include "cpu/storage_manager.h"
#endif

#include "notifier.h"
#include "channel_manager.h"

struct global_buffers_t {
  global_buffers_t(void* p)
    : global_buffers_t(vector<void*>{p})
  {}

  global_buffers_t(vector<void*> const& ps)
    : ptrs(ps)
  {}

  struct desc_t {
    int which;
  };

  struct resource_t {
    void* ptr;

    void* at(uint64_t offset) const {
      return reinterpret_cast<void*>(
        reinterpret_cast<uint8_t*>(ptr) + offset
      );
    }
  };

  optional<resource_t> try_to_acquire(desc_t desc){
    return resource_t{ .ptr = ptrs.at(desc.which) };
  }

  void release(resource_t) {}

  vector<void*> ptrs;
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
    cpu_storage_manager_t::desc_t,
#endif
    global_buffers_t::desc_t,
    group_manager_t::desc_t,
    notifier_t::desc_t,
    channel_manager_t::desc_t
  >;
  using resource_unit_t = std::variant<
#ifdef CPU_EXEC
    cpu_workspace_manager_t::resource_t,
    cpu_storage_manager_t::resource_t,
#endif
    global_buffers_t::resource_t,
    group_manager_t::resource_t,
    notifier_t::resource_t,
    channel_manager_t::resource_t
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
  cpu_storage_manager_t* cpu_storage_manager;
#endif
  group_manager_t*   group_manager;
  global_buffers_t*  global_buffers;
  notifier_t*        notifier;
  channel_manager_t* channel_manager;
};


