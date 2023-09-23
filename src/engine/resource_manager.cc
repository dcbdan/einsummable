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
  TRY_VARIANT_ACQUIRE(global_buffer_t::desc_t, global_buffer);
#ifdef CPU_EXEC
  TRY_VARIANT_ACQUIRE(cpu_workspace_manager_t::desc_t, cpu_workspace_manager);
#endif
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
  TRY_VARIANT_RELEASE(global_buffer_t::resource_t, global_buffer);
#ifdef CPU_EXEC
  TRY_VARIANT_RELEASE(cpu_workspace_manager_t::resource_t, cpu_workspace_manager);
#endif
  throw std::runtime_error("should not reach");
}

void resource_manager_t::release(
  resource_manager_t::resource_t resource)
{
}

