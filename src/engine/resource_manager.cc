#include "resource_manager.h"

optional<vector<resource_ptr_t>>
resource_manager_t::try_to_acquire_impl(vector<desc_ptr_t> const& descs)
{
  vector<resource_ptr_t> ret;
  ret.reserve(descs.size());
  for(auto desc: descs) {
    resource_ptr_t maybe = try_to_acquire_unit(desc);
    if(maybe) {
      ret.push_back(maybe);
    } else {
      release_impl(ret);
      return std::nullopt;
    }
  }

  return ret;
}

void resource_manager_t::release_impl(
  vector<resource_ptr_t> const& resources)
{
  for(auto resource: resources) {
    release_unit(resource);
  }
}

resource_ptr_t resource_manager_t::try_to_acquire_unit(
  desc_ptr_t desc)
{
  for(auto manager: managers) {
    if(manager->handles(desc)) {
      return manager->try_to_acquire(desc);
    }
  }
  throw std::runtime_error("should not reach: try to acquire unit");
}

void resource_manager_t::release_unit(
  resource_ptr_t resource)
{
  for(auto manager: managers) {
    if(manager->handles(resource)) {
      return manager->release(resource);
    }
  }
  throw std::runtime_error("should not reach: release unit");
}

optional<tuple<int, bool>>
group_manager_t::try_to_acquire_impl(int const& group_id) {
  std::unique_lock lk(m);
  if(busy_groups.count(group_id) == 0) {
    busy_groups.insert(group_id);
    bool is_first = seen_groups.count(group_id) == 0;
    return tuple<int, bool>(group_id, is_first);
  } else {
    return std::nullopt;
  }
}

void group_manager_t::release_impl(tuple<int, bool> const& info) {
  auto const& [group_id, is_first] = info;
  std::unique_lock lk(m);
  if(!busy_groups.erase(group_id)) {
    throw std::runtime_error("trying to release a group id that isn't busy");
  }
  seen_groups.insert(group_id);
}

