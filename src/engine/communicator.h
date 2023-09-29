#pragma once
#include "../base/setup.h"

struct communicator_t {
  // TODO: constructor
  communicator_t() {}
  // - connect to all the other workers and get setup
  // - have a notify recver for each location
  // - launch a notify thread that isn't killed until the destructor
  //   is called

  int get_this_rank()  const { return this_rank;  }
  int get_world_size() const { return world_size; }

  void send_sync(int dst, void const* data, uint64_t size);
  void recv_sync(int src, void* data,       uint64_t size);

  void send_int_sync(int dst, int val);
  int  recv_int_sync(int src);

  void barrier_sync();

  void set_notify_callback(std::function<void(void* data, uint64_t size)> callback);
  void set_notify_recv_size(uint64_t);
  void notify_sync(int dst, void* data, uint64_t size);

private:
  int this_rank;
  int world_size;

  uint64_t notify_recv_size;
  std::function<void(void* data, uint64_t size)> notify_callback;
};

struct channel_manager_t {
  channel_manager_t(communicator_t& comm);

  struct desc_t {
    bool send;
    int loc;
  };

  struct resource_t {
    void send(void* ptr, uint64_t bytes) const;
    void recv(void* ptr, uint64_t bytes, int channel) const;

    int get_channel() const { return channel; }

    channel_manager_t* self;
    int loc;
    int channel; // -1 if no send channel reserved;
                 // then it can only recv
  };

  optional<resource_t> try_to_acquire(desc_t desc);

  void release(resource_t rsrc);

private:
  communicator_t& comm;

  std::mutex m;
  vector<vector<int>> avail_channels;

  optional<int> acquire_channel(int loc);
  void release_channel(int loc, int channel);
};
