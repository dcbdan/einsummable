#include "resource_manager.h"

#define TRY_VARIANT_ACQUIRE(type_t, obj) \
  if(std::holds_alternative<type_t>(desc)) { \
    if(obj) { \
      return obj->try_to_acquire(std::get<type_t>(desc)); \
    } else { \
      throw std::runtime_error("obj is nullptr"); \
    } \
  }


optional<resource_manager_t::resource_unit_t>
resource_manager_t::try_to_acquire_unit(
  resource_manager_t::desc_unit_t const& desc)
{
#ifdef CPU_EXEC
  TRY_VARIANT_ACQUIRE(cpu_workspace_manager_t::desc_t, cpu_workspace_manager);
  TRY_VARIANT_ACQUIRE(cpu_storage_manager_t::desc_t, cpu_storage_manager);
#endif
  TRY_VARIANT_ACQUIRE(global_buffer_t::desc_t, global_buffer);
  TRY_VARIANT_ACQUIRE(group_manager_t::desc_t, group_manager);
  throw std::runtime_error("should not reach");
}

optional<resource_manager_t::resource_t>
resource_manager_t::try_to_acquire(desc_t const& desc)
{
  vector<resource_unit_t> ret;
  for(auto const& unit: desc) {
    auto maybe = try_to_acquire_unit(unit);
    if(maybe) {
      ret.push_back(maybe.value());
    } else {
      release(ret);
      return std::nullopt;
    }
  }
  return ret;
}

#define TRY_VARIANT_RELEASE(type_t, obj)  \
  if(std::holds_alternative<type_t>(rsrc)) { \
    if(obj) { \
      return obj->release(std::get<type_t>(rsrc)); \
    } else { \
      throw std::runtime_error("obj is nullptr"); \
    } \
  }

void resource_manager_t::release_unit(
  resource_manager_t::resource_unit_t const& rsrc)
{
#ifdef CPU_EXEC
  TRY_VARIANT_RELEASE(cpu_workspace_manager_t::resource_t, cpu_workspace_manager);
  TRY_VARIANT_RELEASE(cpu_storage_manager_t::resource_t, cpu_storage_manager);
#endif
  TRY_VARIANT_RELEASE(global_buffer_t::resource_t, global_buffer);
  TRY_VARIANT_RELEASE(group_manager_t::resource_t, group_manager);
  throw std::runtime_error("should not reach");
}

void resource_manager_t::release(
  resource_manager_t::resource_t resources)
{
  for(auto const& unit: resources) {
    release_unit(unit);
  }
}

optional<group_manager_t::resource_t>
group_manager_t::try_to_acquire(group_manager_t::desc_t desc) {
  std::unique_lock lk(m);
  if(busy_groups.count(desc.group_id) == 0) {
    busy_groups.insert(desc.group_id);
    return resource_t{
      .group_id = desc.group_id,
      .is_first = (seen_groups.count(desc.group_id) == 0)
    };
  } else {
    return std::nullopt;
  }
}

void group_manager_t::release(group_manager_t::resource_t rsrc) {
  std::unique_lock lk(m);
  if(!busy_groups.erase(rsrc.group_id)) {
    throw std::runtime_error("trying to release a group id that isn't busy");
  }
  seen_groups.insert(rsrc.group_id);
}

