#pragma once
#include "../base/setup.h"

#include "resource_manager.h"
#include "communicator.h"

struct send_channel_manager_t;

struct send_channel_manager_resource_t {
  void send(void* ptr, uint64_t bytes) const;

  send_channel_manager_t* self;
  int loc;
  int channel;
};

struct send_channel_manager_t
  : rm_template_t<int, send_channel_manager_resource_t>
    // desc_t { int the_loc }
{
  send_channel_manager_t(communicator_t& comm, int max_count);

private:
  communicator_t& comm;

  vector<vector<int>> avail_channels;

  optional<int> acquire_channel(int loc);
  void release_channel(int loc, int channel);

  friend class send_channel_manager_resource_t;

private:
  optional<send_channel_manager_resource_t>
  try_to_acquire_impl(int const& loc);

  void release_impl(send_channel_manager_resource_t const& rsrc);

private:
  int num_remaining;
};

/////////

struct recv_channel_manager_t;

struct recv_channel_manager_resource_t {
  void recv(void* ptr, uint64_t bytes) const;

  recv_channel_manager_t* self;
  int id;
  int loc;
  int channel;
};

struct recv_channel_manager_t
  : rm_template_t<tuple<int, int>, recv_channel_manager_resource_t>
    // desc_t { int the_id, int src_loc }
{
  recv_channel_manager_t(communicator_t& comm);

  // the only way for any resources to be acquirable is to
  // call this method
  void notify(int id, int loc, int channel);

private:
  communicator_t& comm;

  // Note that resource managers don't have to be thread safe, but
  // this will be accessed from the notifier, so must be safe here
  std::mutex m;

  // loc -> channel -> queue of ids that are ready
  vector<vector<std::queue<int>>> ready_recvs;

  // id -> channel
  map<int, int> id_to_channel;

  friend class recv_channel_manager_resource_t;

  void recv(int id, int loc, int channel, void* ptr, uint64_t num_bytes);

private:
  optional<recv_channel_manager_resource_t>
  try_to_acquire_impl(tuple<int,int> const& desc);

  void release_impl(recv_channel_manager_resource_t const& rsrc);
};

