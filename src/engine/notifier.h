#pragma once
#include "../base/setup.h"

#include "communicator.h"

#include <future>
#include <mutex>

struct notifier_t {
  // starts a notifier on comm
  notifier_t(communicator_t& comm);

  // stops the notifier
  ~notifier_t();

  struct desc_t {};

  struct resource_t {
    notifier_t* self;
  };

  // Called on the recv side
  void notify_recv_ready(int dst, int id);
  void wait_send_ready(int id);
  int get_channel(int id);

  // Called on the send side
  void wait_recv_ready(int id);
  void notify_send_ready(int dst, int id, int channel);

  optional<resource_t> try_to_acquire(desc_t){
    return resource_t{ .self = this };
  }

  void release(resource_t) {}

private:
  communicator_t& comm;

  struct msg_t {
    enum {
      recv_ready,
      send_ready
    } msg_type;
    union {
      struct {
        int id;
      } recv_info;
      struct {
        int id;
        int channel;
      } send_info;
    } msg;
  };

  void process(msg_t const& msg);

  std::mutex m;

  map<int, int> id_to_channel;

  map<int, std::promise<void>> send_promises;
  map<int, std::promise<void>> recv_promises;

  std::future<void> get_send_future(int id);
  std::future<void> get_recv_future(int id);
};
