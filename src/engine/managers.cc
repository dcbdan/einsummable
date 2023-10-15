#include "managers.h"

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

void threadpool_resource_t::launch(std::function<void()> f) const {
  self->launch(id, f);
}

threadpool_manager_t::threadpool_manager_t(threadpool_t& tp)
  : num_avail(tp.num_runners()), threadpool(tp), id_(0)
{}

optional<threadpool_resource_t>
threadpool_manager_t::try_to_acquire_impl(unit_t const&)
{
  if(num_avail == 0) {
    return std::nullopt;
  }
  num_avail--;

  int new_id = id_;
  id_++;

  return threadpool_resource_t(new_id, this);
}

void threadpool_manager_t::release_impl(threadpool_resource_t const& r) {
  num_avail++;
  was_called.erase(r.id);
}

void threadpool_manager_t::launch(int which, std::function<void()> f) {
  if(was_called.count(which) > 0) {
    throw std::runtime_error("this resource already called launch");
  }
  was_called.insert(which);
  threadpool.insert(f);
}

