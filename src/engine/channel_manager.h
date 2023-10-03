#pragma once
#include "../base/setup.h"

#include "resource_manager.h"
#include "communicator.h"

struct channel_manager_t;

struct channel_manager_resource_t {
  void send(void* ptr, uint64_t bytes) const;
  void recv(void* ptr, uint64_t bytes, int channel) const;

  int get_channel() const { return channel; }

  channel_manager_t* self;
  int loc;
  int channel; // -1 if no send channel reserved;
               // then it can only recv
};

struct channel_manager_t
  : rm_template_t<tuple<bool, int>, channel_manager_resource_t>
    // desc_t { bool is_send, int the_loc }
{
  channel_manager_t(communicator_t& comm);

private:
  communicator_t& comm;

  std::mutex m;
  vector<vector<int>> avail_channels;

  optional<int> acquire_channel(int loc);
  void release_channel(int loc, int channel);

  friend class channel_manager_resource_t;

private:
  optional<channel_manager_resource_t>
  try_to_acquire_impl(
    tuple<bool, int> const& is_send_and_the_loc);

  void release_impl(channel_manager_resource_t const& rsrc);
};
