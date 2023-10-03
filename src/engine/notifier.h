#pragma once
#include "../base/setup.h"

#include "resource_manager.h"
#include "communicator.h"

#include <future>
#include <mutex>

struct notifier_t;

struct notifier_resource_t {
  notifier_t* self;
};

struct notifier_t
  : rm_template_t<unit_t, notifier_resource_t>
{
  // starts a notifier on comm
  notifier_t(communicator_t& comm);

  // stops the notifier
  ~notifier_t();

  // Called on the recv side
  void notify_recv_ready(int dst, int id);
  void wait_send_ready(int id);
  int get_channel(int id);

  // Called on the send side
  void wait_recv_ready(int id);
  void notify_send_ready(int dst, int id, int channel);

private:
  communicator_t& comm;

  struct msg_t {
    enum {
      recv_ready,
      send_ready,
      stop
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

private:
  optional<notifier_resource_t> try_to_acquire_impl(unit_t const&){
    return notifier_resource_t{ .self = this };
  }

  void release_impl(notifier_resource_t const&) {}
};
