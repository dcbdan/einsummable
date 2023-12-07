#include "channel_manager.h"

send_channel_manager_t::send_channel_manager_t(communicator_t& comm, int max_count)
  : comm(comm), num_remaining(max_count)
{
  if(num_remaining <= 0) {
    throw std::runtime_error("invalid max_count provided to send_channel_manager");
  }

  int world_size = comm.get_world_size();
  int this_rank = comm.get_this_rank();
  for(int rank = 0; rank != world_size; ++rank) {
    avail_channels.emplace_back(comm.num_channels(rank));
    auto& xs = avail_channels.back();
    std::iota(xs.begin(), xs.end(), 0);
  }
}

optional<send_channel_manager_resource_t>
send_channel_manager_t::try_to_acquire_impl(int const& loc)
{
  auto maybe_channel = acquire_channel(loc);
  if(maybe_channel) {
    return send_channel_manager_resource_t {
      .self = this,
      .loc = loc,
      .channel = maybe_channel.value()
    };
  } else {
    return std::nullopt;
  }
}

void send_channel_manager_t::release_impl(send_channel_manager_resource_t const& rsrc) {
  release_channel(rsrc.loc, rsrc.channel);
}

void
send_channel_manager_resource_t::send(void* ptr, uint64_t bytes) const
{
  self->comm.send(loc, channel, ptr, bytes);
}

optional<int>
send_channel_manager_t::acquire_channel(int loc)
{
  if(num_remaining == 0) {
    DLINEOUT("send_channel_manager: count hit zero!");
    return std::nullopt;
  }

  auto& cs = avail_channels.at(loc);

  if(cs.size() == 0) {
    return std::nullopt;
  }

  num_remaining--;

  optional<int> ret(cs.back());
  cs.pop_back();

  return ret;
}

void send_channel_manager_t::release_channel(int loc, int channel) {
  avail_channels.at(loc).push_back(channel);
  num_remaining++;
}

/////////

recv_channel_manager_t::recv_channel_manager_t(communicator_t& c)
  : comm(c), ready_recvs(c.get_world_size())
{
  for(int rank = 0; rank != c.get_world_size(); ++rank) {
    auto& queues = ready_recvs[rank];
    queues = vector<std::queue<int>>(comm.num_channels(rank));
  }
}

void recv_channel_manager_t::notify(int id, int loc, int channel) {
  std::unique_lock lk(m);

  auto [_, did_insert] = id_to_channel.insert({id, channel});
  if(!did_insert) {
    throw std::runtime_error("this id has already been inserted into id_to_channel!");
  }

  ready_recvs[loc][channel].push(id);
}

optional<recv_channel_manager_resource_t>
recv_channel_manager_t::try_to_acquire_impl(tuple<int,int> const& desc)
{
  auto const& [id, src] = desc;

  std::unique_lock lk(m);

  auto iter = id_to_channel.find(id);
  if(iter == id_to_channel.end()) {
    throw std::nullopt;
  }

  int const& channel = iter->second;

  auto& queue = ready_recvs[src][channel];
  if(queue.size() == 0) {
    return std::nullopt;
  }

  if(queue.front() == id) {
    return recv_channel_manager_resource_t {
      .self = this,
      .id = id,
      .loc = src,
      .channel = channel
    };
  } else {
    return std::nullopt;
  }
}

void recv_channel_manager_t::release_impl(
  recv_channel_manager_resource_t const& r)
{
  // the "resource" will actually have been "released" if and only if
  // this->recv was called
}

void recv_channel_manager_resource_t::recv(void* ptr, uint64_t bytes) const {
  self->recv(id, loc, channel, ptr, bytes);
}

void recv_channel_manager_t::recv(
  int id, int loc, int channel, void* ptr, uint64_t num_bytes)
{
  comm.recv(loc, channel, ptr, num_bytes);

  // Now that the recv has happened, we will update the queue..
  // ("releasing the resource")

  std::unique_lock lk(m);

  id_to_channel.erase(id);

  auto& queue = ready_recvs[loc][channel];
  if(queue.front() != id) {
    throw std::runtime_error("front of queue must be the id");
  }
  queue.pop();
}

