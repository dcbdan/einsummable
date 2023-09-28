#pragma once
#include "../base/setup.h"

#include "communicator.h"

struct notifier_t {
  struct desc_t {};

  struct resource_t {
    notifier_t* self;
  };

  // Called on the recv side
  void notify_recv_ready(int id);
  void wait_send_ready(int id);
  int get_channel(int id);

  // Called on the send side
  void wait_recv_ready(int id);
  void send_ready(int id, int channel);

  optional<resource_t> try_to_acquire(desc_t){
    return resource_t{ .self = this };
  }

  void release(resource_t) {}
};
