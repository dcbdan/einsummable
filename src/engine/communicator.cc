#include "communicator.h"

optional<channel_manager_t::resource_t>
channel_manager_t::try_to_acquire(channel_manager_t::desc_t desc) {
  // TODO
  return std::nullopt;
}

void channel_manager_t::release(channel_manager_t::resource_t rsrc) {
  // TODO
}

void
channel_manager_t::resource_t::send(void* ptr, uint64_t bytes) const
{
  // TODO
}

void
channel_manager_t::resource_t::recv(void* ptr, uint64_t bytes, int channel) const
{
  // TODO
}

