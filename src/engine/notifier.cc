#include "notifier.h"

int _filecount_ = 0;

notifier_t::notifier_t(communicator_t& cm, recv_channel_manager_t& rcm)
  : comm(cm), recv_channel_manager(rcm)
{
  string filename = "notifier_rnk" + write_with_ss(comm.get_this_rank()) + 
    "_cnt" + write_with_ss(_filecount_++);
  print = std::ofstream(filename);
  DLINEOUT(filename);

  bool constant_poll = false;
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
    },
    constant_poll);
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
  if(print.is_open()) {
    print << "notify_recv_ready(dst,id) " << dst << " " << id << std::endl;
  }
}

void notifier_t::wait_send_ready(int id, std::function<void()> callback) {
  std::unique_lock lk(m);
  auto [_0, did_insert_callback] = send_promises.insert({id, callback});
  if(!did_insert_callback) {
    // there was a dummy callback in send_promises, so the event has happened
    callback();
    if(print.is_open()) {
      print << "wait_send_ready(id) " << id << std::endl;
    }
  }
}

void notifier_t::wait_recv_ready(int id, std::function<void()> callback) {
  std::unique_lock lk(m);
  auto [_0, did_insert_callback] = recv_promises.insert({id, callback});
  if(!did_insert_callback) {
    // there was a dummy callback in recv_promises, so the event has happened
    callback();
    if(print.is_open()) {
      print << "wait_recv_ready(id) " << id << std::endl;
    }
  }
}

void notifier_t::notify_send_ready(int dst, int id, int channel) {
  msg_t msg;
  msg.msg_type = msg_t::send_ready;
  msg.msg.send_info.id = id;
  msg.msg.send_info.loc = comm.get_this_rank();
  msg.msg.send_info.channel = channel;

  comm.notify(dst, reinterpret_cast<void*>(&msg), sizeof(msg));
  if(print.is_open()) {
    print << "notify_send_ready(dst,id,channel) " 
      << dst << " " << id << " " << channel << std::endl;
  }
}

void notifier_t::process(notifier_t::msg_t const& msg) {
  std::unique_lock lk(m);

  if(msg.msg_type == msg_t::recv_ready) {
    int const& id = msg.msg.recv_info.id;

    auto [iter, did_insert_dummy] = 
      recv_promises.insert({id, std::function<void()>()});
    if(!did_insert_dummy) {
      auto& callback = iter->second;
      callback();
      if(print.is_open()) {
        print << "process_recv_ready(id) " << id << std::endl;
      }
    } else {
      if(print.is_open()) {
        print << "got_recv_ready(id) " << id << std::endl;
      }
    }
  } else if(msg.msg_type == msg_t::send_ready) {
    auto const& [id, src, channel] = msg.msg.send_info;

    // Make sure to notify the recv channel before setting the promise!
    recv_channel_manager.notify(id, src, channel);

    auto [iter, did_insert_dummy] = 
      send_promises.insert({id, std::function<void()>()});
    if(!did_insert_dummy) {
      auto& callback = iter->second;
      callback();
      if(print.is_open()) {
        print << "process_send_ready(id,src,channel) " << id << " " << src << " " << channel << std::endl;
      }
    } else {
      if(print.is_open()) {
        print << "got_send_ready(id,src,channel) " << id << " " << src << " " << channel << std::endl;
      }
    }
  } else {
    throw std::runtime_error("invalid notifier msg type");
  }
}

