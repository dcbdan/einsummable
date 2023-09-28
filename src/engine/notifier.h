#pragma once
#include "../base/setup.h"

#include "communicator.h"

#include <future>

struct notifier_t {
  struct desc_t {};

  struct resource_t {
    notifier_t* self;
  };

  void notify_recv_ready(int id);

  std::future<void> get_recv_ready(int id);

  std::future<int> get_channel(int id);

  void notify_send_ready(int id, int channel);

  // Not much "resource" to manage--really this is just state
  optional<resource_t> try_to_acquire(desc_t){
    return resource_t{ .self = this };
  }

  void release(resource_t) {}
};
