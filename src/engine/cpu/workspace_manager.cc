#include "workspace_manager.h"

int cpu_workspace_manager_t::acquire(uint64_t size)
{
  int ret = -1;
  int smallest;
  for(int i = 0; i != items.size(); ++i) {
    auto const& [is_available, buffer] = items[i];
    uint64_t const& sz = buffer->size;
    if(is_available && size <= sz) {
      if(ret == -1) {
        ret = i;
        smallest = sz;
      } else {
        if(sz < smallest) {
          ret = i;
          smallest = sz;
        }
      }
    }
  }
  if(ret == -1) {
    items.emplace_back(false, make_buffer(size));
    ret = items.size()-1;
  } else {
    auto& [is_available, _] = items[ret];
    is_available = false;
  }

  return ret;
}

optional<resource_t>
workspace_manager_t::try_to_acquire(desc_t desc)
{
  std::unique_lock lk(m_items);

  resource_t ret;

  ret.which = acquire(desc.size);

  auto& [_, buffer] = items[ret.which];

  ret.ptr = buffer->raw();
  ret.size = buffer->size;

  return ret;
}

tuple<
  optional<tuple<void*, uint64_t>>,
  optional<int>>
workspace_manager_t::get(einsummable_t const& e)
{
  uint64_t size = kernel_manager.workspace_size(e);
  if(size == 0) {
    return {std::nullopt, std::nullopt};
  }
  std::unique_lock lk(m);
  int ret = -1;
  int smallest;
  for(int i = 0; i != workspace.size(); ++i) {
    auto const& [is_available, buffer] = workspace[i];
    uint64_t const& sz = buffer->size;
    if(is_available && size <= sz) {
      if(ret == -1) {
        ret = i;
        smallest = sz;
      } else {
        if(sz < smallest) {
          ret = i;
          smallest = sz;
        }
      }
    }
  }
  if(ret == -1) {
    workspace.emplace_back(false, make_buffer(size));
    ret = workspace.size()-1;
  } else {
    auto& [is_available, _] = workspace[ret];
    is_available = false;
  }

  auto b = std::get<1>(workspace[ret]);
  using t1 = optional<tuple<void*, uint64_t>>;
  using t2 = optional<int>;
  return tuple<t1,t2>{
    t1({b->data, b->size}),
    t2(ret)
  };
}

void workspace_manager_t::release(int which) {
  std::unique_lock lk(m_items);
  auto& [is_available, _] = items[which];
  is_available = true;
}

