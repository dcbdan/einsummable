#pragma once
#include "../base/setup.h"

#include "resource_manager.h"
#include "communicator.h"
#include "channel_manager.h"

#include <mutex>

#include <fstream>

struct notifier_t;

struct notifier_resource_t {
  notifier_t* self;
};

struct notifier_t
  : rm_template_t<unit_t, notifier_resource_t>
{
  // starts a notifier on comm
  notifier_t(
    communicator_t& comm,
    recv_channel_manager_t& recv_channel_manager);

  // stops the notifier
  ~notifier_t();

  // Called on the recv side
  void notify_recv_ready(int dst, int id);
  void wait_send_ready(int id, std::function<void()> callback);

  // Called on the send side
  void wait_recv_ready(int id, std::function<void()> callback);
  void notify_send_ready(int dst, int id, int channel);

private:
  communicator_t& comm;
  recv_channel_manager_t& recv_channel_manager;

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
        int loc;
        int channel;
      } send_info;
    } msg;
  };

  void process(msg_t const& msg);

  // making sure this is thread safe since different
  // threads can access
  std::mutex m;

  // For each id on either the send or recv side,
  // we either have a callback that we will call when that arrives,
  // or we insert an empty function to signal that the id is already
  // ready
  map<int, std::function<void()>> send_promises;
  map<int, std::function<void()>> recv_promises;

private:
  optional<notifier_resource_t> try_to_acquire_impl(unit_t const&){
    return notifier_resource_t{ .self = this };
  }

  void release_impl(notifier_resource_t const&) {}

  std::ofstream print;
};
