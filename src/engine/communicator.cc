#include "communicator.h"
#include "comm_util.h"

#include <cstring>
#include <unistd.h>

#include <thread>

#include <sys/epoll.h>

// TODO: put this somewhere
static uint16_t server_port     = 13337;
static sa_family_t ai_family    = AF_INET;
double _timeout = 10.0;

// Server (rank == 0):
//   for(int rank = 1; rank != world_size; ++rank) {
//     connect oob
//     recv: send addrs from rank
//     recv: recv addrs from rank
//     setup endpoints between 0 and rank
//     send: rank
//     send: send peers < rank
//     send: recv peers < rank
//     close oob
//     connect ucx to rank
//   }
//   for(int rank = 1; rank != world_size; ++rank) {
//     (using ucx wire)
//     send: send peers > rank
//     send: recv peers > rank
//   }
// Client (rank > 0):
//   connect oob
//   send: send addrs
//   send: recv addrs
//   recv: rank
//   recv: send peers < rank
//   recv: recv peers < rank
//   setup endpoints between rank and <rank
//   close oob
//   connect to everything < rank
//   (using ucx wire)
//   recv: send peers > rank
//   recv: recv peers > rank
//   setup endpoints between rank and >rank
communicator_t::communicator_t(
  string addr_zero, bool is_server, int world_size_, int n_channels):
    world_size(world_size_), is_listening(false)
{
  _setup_context();

  // Note: wire_t can't be copied because the memory it contains is possibly
  //       being used by ucx to setup things. So reserve the memory and make
  //       sure the copy constructor is not called.
  auto& n_send_wires = notify_channel.send_wires;
  auto& n_recv_wires = notify_channel.recv_wires;
  n_send_wires.reserve(world_size - 1);
  n_recv_wires.reserve(world_size - 1);
  for(int i = 0; i != world_size - 1; ++i) {
    n_send_wires.emplace_back(this);
    n_recv_wires.emplace_back(this);
  }

  if(is_server) {
    this_rank = 0;
    // To get the addr on send wire_t at r to s:
    //   all_send_addr[r][_to_idx(s, r)]
    // To get the addr on recv wire_t at r from s:
    //   all_recv_addr[r][_to_idx(s, r)]
    vector<vector<addr_data_t>> all_send_addr;
    vector<vector<addr_data_t>> all_recv_addr;

    // Add rank zero to all_*_addr
    {
      all_send_addr.emplace_back();
      for(auto& wire: n_send_wires) {
        all_send_addr[0].push_back(wire.make_addr_data());
      }

      all_recv_addr.emplace_back();
      for(auto& wire: n_recv_wires) {
        all_recv_addr[0].push_back(wire.make_addr_data());
      }
    }

    for(int rank = 1; rank != world_size; ++rank) {
      int oob_sock = -1;
      oob_sock = connect_common(NULL, server_port, ai_family);
      if(oob_sock == -1) {
        throw std::runtime_error("_server_recv_addrs failed to connect");
      }

      all_send_addr.push_back(_oob_recv_addrs(oob_sock));
      all_recv_addr.push_back(_oob_recv_addrs(oob_sock));

      // Setup the endpoints here on the server
      {
        auto& send_peer = all_recv_addr[rank][0];
        n_send_wires[_to_idx(rank)].create_endpoint(send_peer);
      }

      {
        auto& recv_peer = all_send_addr[rank][0];
        n_recv_wires[_to_idx(rank)].create_endpoint(recv_peer);
      }

      _oob_send_int(oob_sock, rank);

      {
        // The peers for send wires are recv wire addresses
        vector<addr_data_t> send_peers = _get_peers_less(all_recv_addr, rank);
        _oob_send_addrs(oob_sock, send_peers);
      }

      {
        // The peers for recv wires are send wire addresses
        vector<addr_data_t> recv_peers = _get_peers_less(all_send_addr, rank);
        _oob_send_addrs(oob_sock, recv_peers);
      }

      close(oob_sock);
    }

    for(int rank = 1; rank < world_size-1; ++rank) {
      {
        vector<addr_data_t> send_peers = _get_peers_more(all_recv_addr, rank);
        _wire_send_addrs(rank, send_peers);
      }

      {
        vector<addr_data_t> recv_peers = _get_peers_more(all_send_addr, rank);
        _wire_send_addrs(rank, recv_peers);
      }
    }
  } else {
    int oob_sock = -1;
    for(int i = 0; i != 30; ++i) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(2000ms);
      oob_sock = connect_common(addr_zero.c_str(), server_port, ai_family);
      if(oob_sock != -1) {
        break;
      }
    }
    if(oob_sock == -1) {
      throw std::runtime_error("_client_send_addrs failed to connect");
    }

    {
      vector<addr_data_t> send_addr;
      for(auto& wire: n_send_wires) {
        send_addr.push_back(wire.make_addr_data());
      }
      _oob_send_addrs(oob_sock, send_addr);
    }

    {
      vector<addr_data_t> recv_addr;
      for(auto& wire: n_recv_wires) {
        recv_addr.push_back(wire.make_addr_data());
      }
      _oob_send_addrs(oob_sock, recv_addr);
    }

    this_rank = _oob_recv_int(oob_sock);

    {
      vector<addr_data_t> less_send_peers = _oob_recv_addrs(oob_sock);
      for(int rank = 0; rank != this_rank; ++rank) {
        auto& addr = less_send_peers.at(rank);
        n_send_wires[_to_idx(rank)].create_endpoint(addr);
      }
    }

    {
      vector<addr_data_t> less_recv_peers = _oob_recv_addrs(oob_sock);
      for(int rank = 0; rank != this_rank; ++rank) {
        auto& addr = less_recv_peers.at(rank);
        n_recv_wires[_to_idx(rank)].create_endpoint(addr);
      }
    }

    close(oob_sock);

    if(this_rank < world_size - 1) {
      {
        vector<addr_data_t> more_send_peers = _wire_recv_addrs(0);
        auto iter = more_send_peers.begin();
        for(int rank = this_rank + 1; rank != world_size; ++rank, ++iter) {
          auto& addr = *iter;
          n_send_wires[_to_idx(rank)].create_endpoint(addr);
        }
      }
      {
        vector<addr_data_t> more_recv_peers = _wire_recv_addrs(0);
        auto iter = more_recv_peers.begin();
        for(int rank = this_rank + 1; rank != world_size; ++rank, ++iter) {
          auto& addr = *iter;
          n_recv_wires[_to_idx(rank)].create_endpoint(addr);
        }
      }
    }
  }

  barrier(notify_channel);

  if(world_size > 1) {
    // we will be waiting for this many messages to arrive
    int num_msgs = n_channels * (world_size - 1) * 2;
    _notify_processor_t processor(num_msgs);

    start_listen_notify(
      sizeof(_notify_processor_t::msg_t),
      [&](vector<uint8_t> data) {
        using msg_t = _notify_processor_t::msg_t;
        msg_t& msg = *reinterpret_cast<msg_t*>(data.data());
        if(msg.ep == _notify_processor_t::msg_t::stop) {
          return true;
        } else {
          processor.process(msg);
          return false;
        }
      }
    );

    // set up all the wires
    for(int channel = 0; channel != n_channels; ++channel) {
      stream_channels.emplace_back();
      auto& send_wires = stream_channels.back().send_wires;
      auto& recv_wires = stream_channels.back().recv_wires;
      send_wires.reserve(world_size - 1);
      recv_wires.reserve(world_size - 1);
      for(int i = 0; i != world_size - 1; ++i) {
        send_wires.emplace_back(this);
        recv_wires.emplace_back(this);
      }
    }

    // notify everyone what their endpoint will be using
    // by notifying with a bunch of _notify_processor_t::msg_t objects
    {
      _notify_processor_t::msg_t msg;
      for(int rank = 0; rank != world_size; ++rank) {
        if(rank == this_rank) {
          continue;
        }

        for(int channel = 0; channel != n_channels; ++channel) {
          msg.rank = this_rank;
          msg.channel = channel;

          {
            auto& wire = get_stream_send_wire(rank, channel);
            msg.ep = _notify_processor_t::msg_t::recv_ep;
            msg.write_addr(wire.make_addr_data());

            notify(rank, reinterpret_cast<void*>(&msg), sizeof(msg));
          }

          {
            auto& wire = get_stream_recv_wire(rank, channel);
            msg.ep = _notify_processor_t::msg_t::send_ep;

            msg.write_addr(wire.make_addr_data());

            notify(rank, reinterpret_cast<void*>(&msg), sizeof(msg));
          }
        }

        msg.ep = _notify_processor_t::msg_t::stop;
        notify(rank, reinterpret_cast<void*>(&msg), sizeof(msg));
      }
    }

    // wait until we have all the addresses and where they come from
    processor.wait();

    // now that we have all the other addresses, setup all the endpoints
    for(auto const& msg: processor.get_recvd_messages()) {
      if(msg.ep == _notify_processor_t::msg_t::recv_ep) {
        get_stream_recv_wire(msg.rank, msg.channel).create_endpoint(msg.get_addr());
      } else if(msg.ep == _notify_processor_t::msg_t::send_ep) {
        get_stream_send_wire(msg.rank, msg.channel).create_endpoint(msg.get_addr());
      }
    }

    stop_listen_notify();
  }
}

communicator_t::~communicator_t() {
  notify_channel.close();

  barrier();

  for(auto& channel: stream_channels) {
    channel.close();
  }

  _close_context();
}

void communicator_t::send(int dst, int channel, void const* data, uint64_t size) {
  get_stream_send_wire(dst, channel).send(data, size);
}

void communicator_t::recv(int src, int channel, void* data, uint64_t size) {
  get_stream_recv_wire(src, channel).recv(data, size);
}

std::future<void> communicator_t::send_async(
  int dst, int channel, void const* data, uint64_t size)
{
  std::promise<void> barrier_;
  std::future<void> ret = barrier_.get_future();
  std::thread t(
    [dst, channel, data, size, this](std::promise<void>&& barrier)
    {
      this->send(dst, channel, data, size);
      barrier.set_value();
    },
    std::move(barrier_)
  );
  t.detach();
  return std::move(ret);
}

std::future<void> communicator_t::recv_async(
  int src, int channel, void* data, uint64_t size)
{
  std::promise<void> barrier_;
  std::future<void> ret = barrier_.get_future();
  std::thread t(
    [src, channel, data, size, this](std::promise<void>&& barrier)
    {
      this->recv(src, channel, data, size);
      barrier.set_value();
    },
    std::move(barrier_)
  );
  t.detach();
  return std::move(ret);
}

void communicator_t::send_int(int dst, int channel, int val) {
  send(dst, channel, &val, sizeof(int));
}

int communicator_t::recv_int(int src, int channel) {
  int ret;
  recv(src, channel, &ret, sizeof(int));
  return ret;
}

void communicator_t::start_listen_notify(
  uint64_t msg_size,
  std::function<bool(vector<uint8_t> data)> callback,
  bool constant_poll)
{
  if(is_listening) {
    throw std::runtime_error("is already listening!");
  }
  is_listening = true;

  vector<wire_t>& recvs = notify_channel.recv_wires;

  if(constant_poll) {
    listen_thread = std::thread([this, msg_size, callback, &recvs] {
      vector<std::unique_ptr<wire_t::listener_t>> listeners;
      for(int i = 0; i != world_size - 1; ++i) {
        listeners.push_back(recvs[i].make_listener(msg_size));
        listeners.back()->start();
      }

      while(listeners.size() > 0) {
        auto iter = listeners.begin();
        while(iter != listeners.end()) {
          auto& listener = *iter;
          if(listener->progress()) {
            if(callback(listener->payload())) {
              // this message informed us that we should stop
              iter = listeners.erase(iter);
            } else {
              listener->start();
              iter++;
            }
          } else {
            // no progress was made, ok, go to the next
            // listener
            iter++;
          }
        }
      }
    });
  } else {
    listen_thread = std::thread([this, msg_size, callback, &recvs] {
      int epoll_fd = epoll_create(1);
      if(epoll_fd == -1) {
          throw std::runtime_error("Failed to create epoll instance.");
      }

      vector<std::unique_ptr<wire_t::listener_t>> listeners;

      vector<struct epoll_event> evs(world_size-1);

      // build the listeners, add then to epoll_fd and
      // post a recv
      for(int i = 0; i != world_size - 1; ++i) {
        listeners.push_back(recvs[i].make_listener(msg_size));

        auto* listener = listeners.back().get();

        auto& ev = evs[i];
        ev.data.ptr = listener;
        ev.events = EPOLLIN;

        // Get the event file descriptor from the UCX worker
        int efd = listener->get_worker_efd();

        if(epoll_ctl(epoll_fd, EPOLL_CTL_ADD, efd, &ev) == -1) {
          throw std::runtime_error("Failed to add to epoll.");
        }

        listener->start();
      }

      // arm all the workers
      auto iter = listeners.begin();
      while(iter != listeners.end()) {
        auto* listener = iter->get();

        // the listener can be done
        // (1) after listener->start() or
        // (2) after listerner->arm_worker()

        if(!listener->get_is_done()) {
          listener->arm_worker();
        }

        if(listener->get_is_done()) {
          if(callback(listener->payload())) {
            int efd = listener->get_worker_efd();
            if(epoll_ctl(epoll_fd, EPOLL_CTL_DEL, efd, NULL) == -1) {
              throw std::runtime_error("could not del with epoll_ctl");
            }
            listeners.erase(iter);
          } else {
            listener->start();
          }
        } else {
          iter++;
        }
      }

      std::vector<struct epoll_event> events(world_size - 1);

      while(listeners.size() > 0) {
        int nfds = epoll_wait(epoll_fd, events.data(), events.size(), -1);
        if (nfds == -1) {
          throw std::runtime_error("epoll_wait failed.");
        }

        for (int n = 0; n < nfds; ++n) {
          auto* listener = static_cast<communicator_t::wire_t::listener_t*>(
            events[n].data.ptr);

          bool do_erase = false;

          // v1: does a few of these iterations
          while(!listener->progress()) {}

          // v2: calls epoll_wait a ton
          //if(!listener->progress()) {
          //  continue;
          //}

          // v3: Nothing ever progresses with this version
          //if(!listener->get_is_done()) {
          //  continue;
          //}

          // v4: a combination of v1 and v2
          //if(!listener->get_is_done()) {
          //  while(ucp_worker_progress(listener->get_worker())) {}
          //}
          //if(!listener->get_is_done()) {
          //  continue;
          //}

          // now the listener is done
          if(callback(listener->payload())) {
            do_erase = true;
          } else {
            while(true) {
              listener->start();

              if(!listener->get_is_done()) {
                listener->arm_worker();
              }

              if(listener->get_is_done()) {
                if(callback(listener->payload())) {
                  do_erase = true;
                  break;
                } else {
                  // we need to start and arm again
                }
              } else {
                // we are armed and ready to keep listening
                break;
              }
            }
          }

          if(do_erase) {
            int efd = listener->get_worker_efd();
            if(epoll_ctl(epoll_fd, EPOLL_CTL_DEL, efd, NULL) == -1) {
              throw std::runtime_error("could not del with epoll_ctl");
            }
            auto iter = std::find_if(listeners.begin(), listeners.end(),
                [listener](const auto& p) { return p.get() == listener; });
            listeners.erase(iter);
          }
        }
      }

      close(epoll_fd);
    });
  }
}

void communicator_t::stop_listen_notify() {
  if(is_listening) {
    listen_thread.join();
    is_listening = false;
  }
}

void communicator_t::notify(int dst, void const* data, uint64_t size) {
  // TODO: is notify mutex necc? Should have one on each notify send wire?
  std::unique_lock lk(notify_mutex);
  get_notify_send_wire(dst).send(data, size);
}

void communicator_t::barrier() {
  // TODO: this implementation could be better if all
  //       workers were waited on at the same time.

  for(auto& channel: stream_channels) {
    barrier(channel);
  }
}

void communicator_t::barrier(communicator_t::channel_t& channel) {
  char c;
  void* ptr = reinterpret_cast<void*>(&c);
  uint64_t sz = sizeof(char);

  if(this_rank == 0) {
    for(int rank = 1; rank != world_size; ++rank) {
      get_recv_wire(rank, channel).recv(ptr, sz);
    }
    for(int rank = 1; rank != world_size; ++rank) {
      get_send_wire(rank, channel).send(ptr, sz);
    }
  } else {
    get_send_wire(0, channel).send(ptr, sz);
    get_recv_wire(0, channel).recv(ptr, sz);
  }
}

void communicator_t::_setup_context() {
  ucp_params_t ucp_params;
  memset(&ucp_params, 0, sizeof(ucp_params));

  ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES          |
                                 UCP_PARAM_FIELD_NAME              |
                                 UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                                 UCP_PARAM_FIELD_ESTIMATED_NUM_PPN ;
  ucp_params.name              = "c04";
  ucp_params.features          = UCP_FEATURE_STREAM | UCP_FEATURE_WAKEUP;
  ucp_params.mt_workers_shared = 1;
  ucp_params.estimated_num_ppn = 1; // estimate processes per node to 1

  handle_ucs_error(
    ucp_init(&ucp_params, NULL, &ucp_context));

  ucp_context_attr_t attr;
  memset(&attr, 0, sizeof(attr));
  attr.field_mask = UCP_ATTR_FIELD_THREAD_MODE;
  handle_ucs_error(
    ucp_context_query(ucp_context, &attr));
  if(attr.thread_mode == UCS_THREAD_MODE_SINGLE) {
    throw std::runtime_error("ucp context does not have multi threaded enabled");
  }
}

void communicator_t::_close_context() {
  ucp_cleanup(ucp_context);
}

vector<communicator_t::addr_data_t>
communicator_t::_oob_recv_addrs(int oob_sock) {
  vector<addr_data_t> ret;
  int n = _oob_recv_int(oob_sock);
  ret.reserve(n);
  for(int i = 0; i != n; ++i) {
    ret.emplace_back();
    addr_data_t& data = ret.back();

    int msg_size = _oob_recv_int(oob_sock);
    data.resize(msg_size);

    int x =
      read(oob_sock, data.raw(), msg_size);
      //recv(oob_sock, data.raw(), msg_size, MSG_WAITALL);
    if(x != msg_size) {
      throw std::runtime_error("did not recv the message");
    }
  }
  return ret;
}

void communicator_t::_oob_send_addrs(int oob_sock, vector<addr_data_t>& addr)
{
  _oob_send_int(oob_sock, addr.size());
  for(auto& a: addr) {
    int msg_size = a.size();
    _oob_send_int(oob_sock, msg_size);

    int x =
      write(oob_sock, a.raw(), msg_size);
      //send(oob_sock, a.raw(), msg_size, 0);

    if(x != msg_size) {
      throw std::runtime_error("did not send the message");
    }
  }
}

int communicator_t::_oob_recv_int(int oob_sock) {
  int ret;
  int x =
    read(oob_sock, &ret, sizeof(int));
    //recv(oob_sock, &ret, sizeof(int), MSG_WAITALL);
  if(x != sizeof(int)) {
    throw std::runtime_error("oob_recv_int");
  }
  return ret;
}

void communicator_t::_oob_send_int(int oob_sock, int rank) {
  int x =
    write(oob_sock, &rank, sizeof(int));
    //send(oob_sock, &rank, sizeof(int), 0);
  if(x != sizeof(int)) {
    throw std::runtime_error("_oob_send_int");
  }
}

void communicator_t::_wire_send_addrs(int dst, vector<communicator_t::addr_data_t> const& addr) {
  int n = addr.size();
  _wire_send_sync(dst, reinterpret_cast<void const*>(&n), sizeof(int));

  for(auto const& a: addr) {
    int sz = a.size();
    _wire_send_sync(dst, reinterpret_cast<void const*>(&sz), sizeof(int));
    _wire_send_sync(dst, a.raw(), sz);
  }
}

vector<communicator_t::addr_data_t>
communicator_t::_wire_recv_addrs(int src) {
  int n;
  _wire_recv_sync(src, reinterpret_cast<void*>(&n), sizeof(int));

  vector<addr_data_t> ret;
  ret.reserve(n);
  for(int i = 0; i != n; ++i) {
    int sz;
    _wire_recv_sync(src, reinterpret_cast<void*>(&sz), sizeof(int));

    ret.emplace_back();
    addr_data_t& d = ret.back();
    d.resize(sz);
    _wire_recv_sync(src, d.raw(), sz);
  }

  return ret;
}

void communicator_t::_wire_send_sync(int dst, void const* data, uint64_t size) {
  auto& wire = get_notify_send_wire(dst);
  wire.send(data, size);
}

void communicator_t::_wire_recv_sync(int src, void* data, uint64_t size) {
  auto& wire = get_notify_recv_wire(src);
  wire.recv(data, size);
}

vector<communicator_t::addr_data_t>
communicator_t::_get_peers_less(
  vector<vector<communicator_t::addr_data_t>> const& all_addr,
  int this_rank)
{
  vector<addr_data_t> ret;
  ret.reserve(this_rank-1);
  for(int rank = 0; rank != this_rank; ++rank) {
    // Note: this_rank - 1 == _to_idx(this_rank, rank)
    auto const& peer = all_addr[rank][this_rank-1];
    ret.push_back(peer);
  }
  return ret;
}

vector<communicator_t::addr_data_t>
communicator_t::_get_peers_more(
  vector<vector<communicator_t::addr_data_t>> const& all_addr,
  int this_rank)
{
  vector<addr_data_t> ret;

  int world_size = all_addr.size();
  ret.reserve(world_size-this_rank);
  for(int rank = this_rank+1; rank != world_size; ++rank) {
    // Note: this_rank == _to_idx(this_rank, rank)
    auto const& peer = all_addr[rank][this_rank];
    ret.push_back(peer);
  }
  return ret;
}

communicator_t::addr_data_t
communicator_t::_notify_processor_t::msg_t::get_addr() const {
  addr_data_t ret;
  ret.data.resize(addrsize);
  std::copy(addr, addr + addrsize, ret.data.data());
  return ret;
}

void communicator_t::_notify_processor_t::msg_t::write_addr(addr_data_t const& d)
{
  addrsize = d.size();
  if(addrsize > max_addr_size) {
    throw std::runtime_error("max addr size is not large enough!");
  }
  uint8_t const* data = d.data.data();
  std::copy(data, data + addrsize, addr);
}

void communicator_t::_notify_processor_t::process(msg_t const& msg) {
  if(recvd_messages.size() >= num_msg) {
    throw std::runtime_error("too many msgs recvd");
  }

  recvd_messages.push_back(msg);

  if(recvd_messages.size() == num_msg) {
    when_done.set_value();
  }
}

int communicator_t::_to_idx(int rank, int this_rank_) {
  if(rank < this_rank_) {
    return rank;
  } else if(rank > this_rank_) {
    return rank-1;
  } else {
    throw std::runtime_error("invalid call to _to_idx");
  }
}

int communicator_t::_to_idx(int rank) const {
  return _to_idx(rank, this_rank);
}

int communicator_t::_from_idx(int idx, int this_rank_) {
  if(idx < this_rank_) {
    return idx;
  } else if(idx >= this_rank_) {
    return idx+1;
  } else {
    throw std::runtime_error("should not reach");
  }
}

int communicator_t::_from_idx(int idx) const {
  return _from_idx(idx, this_rank);
}

communicator_t::wire_t::wire_t(communicator_t* self)
  : has_endpoint(false)
{
  auto thread_mode = UCS_THREAD_MODE_SERIALIZED;

  // create the worker
  {
    memset(&worker_params, 0, sizeof(worker_params));

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = thread_mode;

    handle_ucs_error(
      ucp_worker_create(self->ucp_context, &worker_params, &worker));
  }

  // create the addr; check the
  {
    memset(&worker_attr, 0, sizeof(worker_attr));

    worker_attr.field_mask =
      UCP_WORKER_ATTR_FIELD_ADDRESS | UCP_WORKER_ATTR_FIELD_THREAD_MODE;

    handle_ucs_error(ucp_worker_query(worker, &worker_attr));

    ucp_addr_size = worker_attr.address_length;
    ucp_addr      = worker_attr.address;

    if(worker_attr.thread_mode != thread_mode) {
      throw std::runtime_error("could not set worker thread mode");
    }
  }
}

communicator_t::wire_t::~wire_t()
{
  // take apart the address
  ucp_worker_release_address(worker, ucp_addr);

  // take apart the worker
  ucp_worker_destroy(worker);
}

void communicator_t::wire_t::create_endpoint(communicator_t::addr_data_t const& addr) {
  ucp_ep_params_t ep_params;

  ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS    |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                              UCP_EP_PARAM_FIELD_ERR_HANDLER       |
                              UCP_EP_PARAM_FIELD_USER_DATA         ;
  ep_params.address         = addr.get();
  ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;

  ep_params.err_handler.cb  = [](void* arg, ucp_ep_h, ucs_status_t) {
    throw std::runtime_error("wire_t could not create endpoint");
  };
  ep_params.err_handler.arg = NULL;
  ep_params.user_data       = reinterpret_cast<void*>(this);

  handle_ucs_error(
    ucp_ep_create(worker, &ep_params, &endpoint));
  has_endpoint = true;
}

communicator_t::addr_data_t communicator_t::wire_t::make_addr_data() {
  addr_data_t ret;
  auto& d = ret.data;
  d.resize(ucp_addr_size);
  uint8_t* raw = reinterpret_cast<uint8_t*>(ucp_addr);
  std::copy(raw, raw + ucp_addr_size, d.begin());
  return ret;
};

void communicator_t::wire_t::send(void const* data, uint64_t size) {
  if(!has_endpoint) {
    throw std::runtime_error("must have endpoint setup");
  }

  bool is_done = false;

  ucp_request_param_t param;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK   |
                       UCP_OP_ATTR_FIELD_USER_DATA  ;

  param.cb.send = [](void*, ucs_status_t status, void* data) {
    bool& is_done = *reinterpret_cast<bool*>(data);
    handle_ucs_error(status, "in send callback");
    is_done = true;
  };

  param.user_data = reinterpret_cast<void*>(&is_done);

  ucs_status_ptr_t status = ucp_stream_send_nbx(endpoint, data, size, &param);

  if(status != NULL) {
    if (UCS_PTR_IS_ERR(status)) {
      throw std::runtime_error(
        "send fail: " + write_with_ss(UCS_PTR_STATUS(status)));
    }

    while(!is_done) {
      if(!ucp_worker_progress(worker)) {
        handle_ucs_error(ucp_worker_wait(worker), "ucp_worker_wait at send");
      }
    }

    ucp_request_free(status);
  }
}

void communicator_t::wire_t::recv(void* data, uint64_t size) {
  if(!has_endpoint) {
    throw std::runtime_error("must have endpoint setup");
  }

  bool is_done = false;

  ucp_request_param_t param;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK   |
                       UCP_OP_ATTR_FIELD_USER_DATA  |
                       UCP_OP_ATTR_FIELD_FLAGS      ;
  param.flags = UCP_STREAM_RECV_FLAG_WAITALL;

  param.cb.recv_stream = [](void*, ucs_status_t status, size_t length, void* data) {
    bool& is_done = *reinterpret_cast<bool*>(data);
    handle_ucs_error(status, "in recv callback");
    is_done = true;
  };

  param.user_data = reinterpret_cast<void*>(&is_done);

  ucs_status_ptr_t status = ucp_stream_recv_nbx(endpoint, data, size, &size, &param);

  if(status != NULL) {
    if (UCS_PTR_IS_ERR(status)) {
      throw std::runtime_error(
        "recv fail: " + write_with_ss(UCS_PTR_STATUS(status)));
    }

    // Two ways to do the wait:
    //   constant polling vs ucp_worker_wait (requires the wakeup feature)

    while(!is_done) {
      if(!ucp_worker_progress(worker)) {
        handle_ucs_error(ucp_worker_wait(worker), "ucp_worker_wait at recv");
      }
    }

    ucp_request_free(status);
  }
}

communicator_t::wire_t::listener_t::listener_t(communicator_t::wire_t& w, uint64_t msg_size):
  data(msg_size), size(msg_size), self(w), request_status(NULL)
{
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK   |
                       UCP_OP_ATTR_FIELD_USER_DATA  |
                       UCP_OP_ATTR_FIELD_FLAGS      ;
  param.flags = UCP_STREAM_RECV_FLAG_WAITALL;
  param.cb.recv_stream = [](void*, ucs_status_t status, size_t length, void* data) {
    auto& self = *reinterpret_cast<listener_t*>(data);
    handle_ucs_error(status, "in notify recv callback");
    self.is_done = true;
  };
  param.user_data = reinterpret_cast<void*>(this);

  _efd = make_worker_efd();
}

communicator_t::wire_t::listener_t::~listener_t() {
  if(request_status != NULL) {
    ucp_request_free(request_status);
  }
}

void communicator_t::wire_t::listener_t::start() {
  is_done = false;

  if(request_status != NULL) {
    ucp_request_free(request_status);
  }

  request_status = ucp_stream_recv_nbx(
    self.endpoint,
    reinterpret_cast<void*>(data.data()),
    size,
    &size,
    &param);

  if(request_status == NULL) {
    if(size != data.size()) {
      throw std::runtime_error("uh-oh: size has been changed");
    }
    is_done = true;
  } else if(UCS_PTR_IS_ERR(request_status)) {
    auto err = UCS_PTR_RAW_STATUS(request_status);
    string err_msg = ucs_status_string(err);
    throw std::runtime_error("listener start recv fail: " + err_msg +
        ". Error code: " + write_with_ss(err));
  } else {
    // gonna have to wait
  }
}

bool communicator_t::wire_t::listener_t::progress() {
  if(is_done) { return true; }

  ucp_worker_progress(self.worker);

  return is_done;
}

int communicator_t::wire_t::listener_t::make_worker_efd() {
  int efd = 0;
  ucs_status_t status = ucp_worker_get_efd(self.worker, &efd);

  if(status != UCS_OK){
    throw std::runtime_error("Failed to get ucp worker efd.");
  }

  return efd;
}

void communicator_t::wire_t::listener_t::arm_worker() {
  while(ucp_worker_progress(self.worker)) {}
  if(is_done) {
    return;
  }
  while (true) {
    ucs_status_t status = ucp_worker_arm(self.worker);
    if (status == UCS_OK) {
      break;
    } else if (status == UCS_ERR_BUSY) {
      ucp_worker_progress(self.worker);
      if(is_done) {
        return;
      }
    }  else {
      // handle other errors
      handle_ucs_error(status, "Unexpected error when arming the worker.");
    }
  }
}
