#include "communicator.h"

void communicator_t::send_sync(int dst, int channel, void const* data, uint64_t size)
{
  // TODO
}

void communicator_t::recv_sync(int src, int channel, void* data, uint64_t size) {
  // TODO
}

void communicator_t::send_int_sync(int dst, int val) {
  // TODO
}
int communicator_t::recv_int_sync(int src) {
  // TODO
}

void communicator_t::barrier_sync() {
  // TODO
}

void communicator_t::set_notify_callback(
  std::function<void(void* data, uint64_t size)> callback)
{
  // TODO maybe need to grab lock
}

void communicator_t::set_notify_recv_size(uint64_t new_recv_size)
{
  // TODO maybe need to grab lock
}

void communicator_t::notify_sync(int dst, void* data, uint64_t size) {
  // TODO need to send to notify on other machine
}

channel_manager_t::channel_manager_t(communicator_t& comm)
  : comm(comm)
{
  // TODO

  //for(int i = 0; i != comm.get_world_size(); ++i) {
  //  avail_channels.push_back(0, 1, ..., num channels)
  //}
}

optional<channel_manager_t::resource_t>
channel_manager_t::try_to_acquire(channel_manager_t::desc_t desc) {
  auto const& [is_send, loc] = desc;
  if(is_send) {
    auto maybe_channel = acquire_channel(loc);
    if(maybe_channel) {
      return resource_t {
        .self = this,
        .loc = loc,
        .channel = maybe_channel.value()
      };
    } else {
      return std::nullopt;
    }
  } else {
    return resource_t {
      .self = this,
      .loc = loc,
      .channel = -1
    };
  }
}

void channel_manager_t::release(channel_manager_t::resource_t rsrc) {
  if(rsrc.channel >= 0) {
    release_channel(rsrc.loc, rsrc.channel);
  }
}

void
channel_manager_t::resource_t::send(void const* ptr, uint64_t num_bytes) const
{
  if(channel < 0) {
    throw std::runtime_error("invalid channel");
  }
  self->comm.send_sync(loc, channel, ptr, num_bytes);
}

void
channel_manager_t::resource_t::recv(void* ptr, uint64_t num_bytes, int channel_) const
{
  self->comm.recv_sync(loc, channel_, ptr, num_bytes);
}

optional<int>
channel_manager_t::acquire_channel(int loc)
{
  std::unique_lock lk(m);

  auto& cs = avail_channels.at(loc);

  if(cs.size() == 0) {
    return std::nullopt;
  }

  optional<int> ret(cs.back());
  cs.pop_back();

  return ret;
}

void channel_manager_t::release_channel(int loc, int channel) {
  std::unique_lock lk(m);

  avail_channels.at(loc).push_back(channel);
}

