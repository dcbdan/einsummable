#include "resource_manager.h"

optional<vector<resource_ptr_t>>
resource_manager_t::try_to_acquire_impl(vector<desc_ptr_t> const& descs)
{
  ghost_t ghost = make_rmacquire_ghost();
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

///
std::mutex rmacquire_total_mutex;
double rmacquire_total = 0.0;

ghost_t make_rmacquire_ghost() {
  return ghost_t(rmacquire_total_mutex, rmacquire_total);
}

double get_rmacquire_total() { return rmacquire_total; }

