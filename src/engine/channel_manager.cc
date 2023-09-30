#include "channel_manager.h"

channel_manager_t::channel_manager_t(communicator_t& comm)
  : comm(comm)
{
  int world_size = comm.get_world_size();
  int this_rank = comm.get_this_rank();
  for(int rank = 0; rank != world_size; ++rank) {
    avail_channels.emplace_back(comm.num_channels(rank));
    auto& xs = avail_channels.back();
    std::iota(xs.begin(), xs.end(), 0);
  }
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
channel_manager_t::resource_t::send(void* ptr, uint64_t bytes) const
{
  if(channel < 0) {
    throw std::runtime_error("invalid channel for send");
  }
  self->comm.send(loc, channel, ptr, bytes);
}

void
channel_manager_t::resource_t::recv(void* ptr, uint64_t bytes, int channel_) const
{
  self->comm.recv(loc, channel_, ptr, bytes);
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
