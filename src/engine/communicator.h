#pragma once
#include "../base/setup.h"

#include <ucp/api/ucp.h>

#include <future>
#include <atomic>
#include <memory>

struct communicator_t {
  communicator_t(string addr_zero, bool is_server, int world_size_, int n_channels = 1);

  ~communicator_t();

  int get_this_rank()  const { return this_rank;  }
  int get_world_size() const { return world_size; }

  int num_channels(int loc) const {
    return loc == this_rank ? 0 : stream_channels.size();
  }

  void send(int dst, int channel, void const* data, uint64_t size);
  void recv(int src, int channel, void* data,       uint64_t size);

  void send_int(int dst, int channel, int val);
  int  recv_int(int src, int channel);

  // TODO: right now, start_listen_notify launches a thread that constantly polls.
  //       maybe this is not what we want
  // TODO: In practice, it appears to be the case that a recv post on a stream cannot
  //       be cancelled. The problem with this is that start_listen_notify originally
  //       just waited until a stop_listen_notify was called and then it would attempt
  //       to cancel the recvs on all posts. But that left dangling recvs and things
  //       would break.
  //
  //       Instead, the callback returns whether or not the wire that got this message
  //       should be stopped. Once all recving wires have been stopped, the thread
  //       launched by start_listen_notify will stop. So currently, if you want to
  //       ever join start_listen_notify, what you should do is send a message to every
  //       other location and the callback should decipher that that message will be the
  //       last message on the wire.
  void start_listen_notify(
    uint64_t msg_size,
    std::function<bool(vector<uint8_t> data)> callback);

  void stop_listen_notify();

  void notify(int dst, void const* data, uint64_t size);

  void barrier();

  void send(int dst, void const* data, uint64_t size) {
    send(dst, 0, data, size);
  }
  void recv(int src, void* data, uint64_t size) {
    recv(src, 0, data, size);
  }

  void send_int(int dst, int val) {
    send_int(dst, 0, val);
  }
  int recv_int(int src) {
    return recv_int(src, 0);
  }

  template <typename T>
  void send_contig_obj(int dst, T const& obj) {
    send(dst, reinterpret_cast<void const*>(&obj), sizeof(obj));
  }
  template <typename T>
  T recv_contig_obj(int dst) {
    T ret;
    recv(dst, reinterpret_cast<void*>(&ret), sizeof(ret));
    return ret;
  }
  template <typename T>
  void broadcast_contig_obj(T const& obj) {
    for(int rank = 0; rank != world_size; ++rank) {
      if(rank == this_rank) {
        continue;
      }
      send_contig_obj(rank, obj);
    }
  }

  template <typename T>
  void send_vector(int dst, vector<T> const& xs) {
    int n = xs.size();
    send_int(dst, n);
    send(
      dst,
      reinterpret_cast<void const*>(xs.data()),
      n*sizeof(T));
  }
  template <typename T>
  vector<T> recv_vector(int src) {
    int n = recv_int(src);
    vector<T> ret(n);
    recv(
      src,
      reinterpret_cast<void*>(ret.data()),
      n*sizeof(T));
    return ret;
  }
  template <typename T>
  void broadcast_vector(vector<T> const& xs) {
    for(int rank = 0; rank != world_size; ++rank) {
      if(rank == this_rank) {
        continue;
      }
      send_vector(rank, xs);
    }
  }

  void send_string(int dst, string const& str) {
    int sz = str.size();
    char const* ptr = str.c_str();
    send_int(dst, sz);
    send(dst, reinterpret_cast<void const*>(ptr), sz);
  }
  string recv_string(int src) {
    int sz = recv_int(src);
    vector<char> data(sz);
    recv(src, reinterpret_cast<void*>(data.data()), sz);
    return string(data.begin(), data.end());
  }
  void broadcast_string(string const& str) {
    for(int rank = 0; rank != world_size; ++rank) {
      if(rank == this_rank) {
        continue;
      }
      send_string(rank, str);
    }
  }

private:
  struct addr_data_t {
    vector<uint8_t> data;

    void resize(uint64_t const& sz) { data.resize(sz); }

    uint64_t size() const { return data.size(); }

    ucp_address_t* get() {
      return reinterpret_cast<ucp_address_t*>(data.data());
    }
    ucp_address_t const* get() const {
      return reinterpret_cast<ucp_address_t const*>(data.data());
    }

    void* raw() {
      return reinterpret_cast<void*>(data.data());
    }
    void const* raw() const {
      return reinterpret_cast<void const*>(data.data());
    }
  };

  struct wire_t {
    wire_t(communicator_t* self);

    wire_t(wire_t const& other) {
      throw std::runtime_error("wire_t should not be copied!");
    }

    ~wire_t();

    void create_endpoint(addr_data_t const& addr);

    addr_data_t make_addr_data();

    ucp_address_t* get_ucp_addr() { return ucp_addr; }
    uint64_t get_ucp_addr_size() { return ucp_addr_size; }

    void send(void const* data, uint64_t size);

    void recv(void* data, uint64_t size);

    struct listener_t {
      listener_t(wire_t& w, uint64_t msg_size);

      ~listener_t();

      void start();

      bool progress();

      vector<uint8_t>& payload() { return data; }

    private:
      ucp_request_param_t param;
      vector<uint8_t> data;
      uint64_t size;
      bool is_done;
      wire_t& self;
      ucs_status_ptr_t status;
    };

    std::unique_ptr<listener_t> make_listener(uint64_t msg_size) {
      return std::make_unique<listener_t>(*this, msg_size);
    }

  private:
    ucp_worker_h worker;

    bool has_endpoint;
    ucp_ep_h endpoint;

    ucp_address_t* ucp_addr;
    uint64_t ucp_addr_size;

    // these are used in the constructor but have to
    // stay alive after the constructor TODO (that is the hypothesis)
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t worker_attr;


    friend class listen_t;
  };

  struct channel_t {
    void close() {
      send_wires.clear();
      recv_wires.clear();
    }

    vector<wire_t> recv_wires;
    vector<wire_t> send_wires;
  };

private:
  int this_rank;
  int world_size;

  channel_t notify_channel;
  vector<channel_t> stream_channels;

  ucp_context_h ucp_context;

  // for the listener thread to check
  bool is_listening;
  std::thread listen_thread;

  ///////////
  // these methods will only occur in the constructor {{{
  void _setup_context();

  static vector<addr_data_t> _oob_recv_addrs(int oob_sock);

  // (addrs is taken by reference but not modified)
  static void _oob_send_addrs(int oob_sock, vector<addr_data_t>& addr);

  static int _oob_recv_int(int oob_sock);

  static void _oob_send_int(int oob_sock, int rank);

  void _wire_send_addrs(int rank, vector<addr_data_t> const& addr);

  vector<addr_data_t> _wire_recv_addrs(int rank);

  void _wire_send_sync(int dst, void const* data, uint64_t size);

  void _wire_recv_sync(int src, void* data, uint64_t size);

  static vector<addr_data_t>
  _get_peers_less(
    vector<vector<addr_data_t>> const& all_addr,
    int rank);

  static vector<addr_data_t>
  _get_peers_more(
    vector<vector<addr_data_t>> const& all_addr,
    int rank);

  struct _notify_processor_t {
    _notify_processor_t(int n):
      num_msg(n)
    {}

    struct msg_t {
      // TODO: hopefully this is big enough to store a ucx address
      static uint64_t const max_addr_size = 500;

      enum {
        recv_ep,
        send_ep,
        stop
      } ep;
      int rank;
      int channel;
      uint64_t addrsize;
      uint8_t addr[max_addr_size];

      addr_data_t get_addr() const;

      void write_addr(addr_data_t const& d);
    };

    void process(msg_t const& msg);

    void wait() { when_done.get_future().get(); }

    vector<msg_t>& get_recvd_messages() { return recvd_messages; }

  private:
    int num_msg;
    vector<msg_t> recvd_messages;
    std::promise<void> when_done;
  };


  // }}}
  ///////////

  void _close_context();

  static int _to_idx(int rank, int this_rank_);

  int _to_idx(int rank) const;

  static int _from_idx(int idx, int this_rank_);

  int _from_idx(int idx) const;

  void barrier(channel_t& channel);

  wire_t& get_send_wire(int rank, channel_t& channel) {
    return channel.send_wires[_to_idx(rank)];
  }
  wire_t& get_recv_wire(int rank, channel_t& channel) {
    return channel.recv_wires[_to_idx(rank)];
  }

  wire_t& get_stream_send_wire(int rank, int channel) {
    return stream_channels[channel].send_wires[_to_idx(rank)];
  }
  wire_t& get_stream_recv_wire(int rank, int channel) {
    return stream_channels[channel].recv_wires[_to_idx(rank)];
  }

  wire_t& get_notify_send_wire(int rank) {
    return notify_channel.send_wires[_to_idx(rank)];
  }
  wire_t& get_notify_recv_wire(int rank) {
    return notify_channel.recv_wires[_to_idx(rank)];
  }

  std::mutex notify_mutex; // TODO: is this needed?
};

