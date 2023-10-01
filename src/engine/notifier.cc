#include "notifier.h"

notifier_t::notifier_t(communicator_t& cm)
  : comm(cm)
{
  comm.start_listen_notify(
    sizeof(msg_t),
    [this](vector<uint8_t> data) {
      if(data.size() != sizeof(msg_t)) {
        throw std::runtime_error("recv notify of incorrect size");
      }
      msg_t& msg = *reinterpret_cast<msg_t*>(data.data());
      if(msg.msg_type == msg_t::stop) {
        return true;
      } else {
        this->process(msg);
        return false;
      }
  });
}

notifier_t::~notifier_t() {
  int this_rank = comm.get_this_rank();
  int world_size = comm.get_world_size();

  msg_t msg;
  msg.msg_type = msg_t::stop;

  for(int rank = 0; rank != world_size; ++rank) {
    if(rank != this_rank) {
      comm.notify(rank, reinterpret_cast<void*>(&msg), sizeof(msg));
    }
  }

  comm.stop_listen_notify();
}

void notifier_t::notify_recv_ready(int dst, int id) {
  msg_t msg;
  msg.msg_type = msg_t::recv_ready;
  msg.msg.recv_info.id = id;

  comm.notify(dst, reinterpret_cast<void*>(&msg), sizeof(msg));
}

void notifier_t::wait_send_ready(int id) {
  std::future<void> f = get_send_future(id);
  return f.get();
}

int notifier_t::get_channel(int id) {
  std::unique_lock lk(m);

  auto iter = id_to_channel.find(id);
  if(iter == id_to_channel.end()) {
    throw std::runtime_error("cannot get channel; not here!");
  }
  int ret = iter->second;
  id_to_channel.erase(iter);
  return ret;
}

void notifier_t::wait_recv_ready(int id) {
  std::future<void> f = get_recv_future(id);
  return f.get();
}

void notifier_t::notify_send_ready(int dst, int id, int channel) {
  msg_t msg;
  msg.msg_type = msg_t::send_ready;
  msg.msg.send_info.id = id;
  msg.msg.send_info.channel = channel;

  comm.notify(dst, reinterpret_cast<void*>(&msg), sizeof(msg));
}

void notifier_t::process(notifier_t::msg_t const& msg) {
  std::unique_lock lk(m);

  if(msg.msg_type == msg_t::recv_ready) {
    int const& id = msg.msg.recv_info.id;

    auto [iter, _0] = recv_promises.insert({id, std::promise<void>()});
    iter->second.set_value();
  } else if(msg.msg_type == msg_t::send_ready) {
    auto const& [id, channel] = msg.msg.send_info;

    auto [_1, did_insert_channel] = id_to_channel.insert({id, channel});
    if(!did_insert_channel) {
      throw std::runtime_error("this id already has a channel value!");
    }

    auto [iter, _2] = send_promises.insert({id, std::promise<void>()});
    iter->second.set_value();

  } else {
    throw std::runtime_error("invalid notifier msg type");
  }
}

std::future<void> notifier_t::get_send_future(int id) {
  std::unique_lock lk(m);

  auto [iter, _] = send_promises.insert({id, std::promise<void>()});
  return iter->second.get_future();
}

std::future<void> notifier_t::get_recv_future(int id) {
  std::unique_lock lk(m);

  auto [iter, _] = recv_promises.insert({id, std::promise<void>()});
  return iter->second.get_future();
}

