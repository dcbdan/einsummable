#pragma once
#include "../base/setup.h"

struct communicator_t {
  // TODO
};

struct channel_manager_t {
  struct desc_t {
    bool send;
    int loc;
  };

  struct resource_t {
    void send(void* ptr, uint64_t bytes) const;
    void recv(void* ptr, uint64_t bytes, int channel) const;

    int get_channel() const { return channel; }
  private:
    channel_manager_t* self;
    int loc;
    int channel; // -1 if no send channel reserved;
                 // then it can only recv
  };

  optional<resource_t> try_to_acquire(desc_t desc);

  void release(resource_t rsrc);

private:
  communicator_t* comm;
};
