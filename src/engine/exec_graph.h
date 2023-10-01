#pragma once
#include "../base/setup.h"

#include "resource_manager.h"

#include "../einsummable/memgraph.h"

#ifdef CPU_EXEC
#include "cpu/kernel_executor.h"
#include "cpu/workspace_manager.h"
#include "cpu/storage.h"
#endif

struct exec_graph_t {
#ifdef CPU_EXEC
  exec_graph_t(cpu_kernel_executor_t& e)
    : cpu_executor(e)
  {}
#endif

#ifdef CPU_EXEC
  static exec_graph_t make_cpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    cpu_kernel_executor_t& cpu_executor);
#endif

  using desc_t = resource_manager_t::desc_t;
  using desc_unit_t = resource_manager_t::desc_unit_t;

  using rsrc_t = resource_manager_t::resource_t;
  using rsrc_unit_t = resource_manager_t::resource_unit_t;

  struct dummy_t {
    void launch(rsrc_t resource, std::function<void()> callback) const {
      callback();
    }
    desc_t resource_description() const {
      return vector<desc_unit_t>();
    }
    void print(std::ostream& out) const { out << "dummy"; }
  };

#ifdef CPU_EXEC
  struct cpu_einsummable_t {
    cpu_kernel_executor_t& cpu_executor;
    einsummable_t einsummable;
    vector<mem_t> mems;
    uint64_t workspace_size;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
    void print(std::ostream& out) const { out << "cpu_einsummable"; }
  };

  struct cpu_touch_t {
    cpu_kernel_executor_t& cpu_executor;
    touch_t touch;
    int group_id;
    vector<mem_t> mems;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
    void print(std::ostream& out) const { out << "cpu_touch"; }
  };

  struct cpu_evict_t {
    int id;
    mem_t mem;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
    void print(std::ostream& out) const { out << "cpu_evict"; }
  };

  struct cpu_load_t {
    int id;
    mem_t mem;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
    void print(std::ostream& out) const { out << "cpu_load"; }
  };
#endif

  // The communicator object, resource manager and knowing when communication can
  // occur in the state object are all sort of coupled together. So to explain
  // what is going on with the following communication nodes, it is necc to explain how
  // all these parts are fitting together.

  // 1. Note that the state object is not being externally notified of events. So when
  //    some data can be recv'd on some other node is not something the state object
  //    is listening to. That information has to be retrieved in an exec node by
  //    accessing some resource.
  // 2. Note that resources will be acquired for the entire time a node is executing.
  //    As such, if a thread is acquired to send data with but the node ends up
  //    waiting until the corresponding recv side is ready, then a thread is sitting
  //    there idle and unused.
  // 3. The communicator object can send "notify" messages that should be recved quickly
  //    and "stream" messages that are ordered and must have an opposing recv ready.
  //    The stream messages require a full thread to do the work.
  // 4. With the "notify" and "stream" types of ops, in order to send and recv data across
  //    machines, the following handshake process occurs:
  //    * recv side: notify the send side that the recv side is ready
  //    * send side: recv the recv ready notification
  //    * send side: acquire a channel-send resource to use
  //    * send side: notify the recv side what channel to use
  //    * send side: send data over the acquired channel
  //    * recv side: recv notification of which channel
  //    * recv side: recv over channel

  // The following communication nodes implement portions of the handshake while
  // only utilizing the necc resources.

  struct notify_recv_ready_t {
    int id;
    int dst;

    // notify the dst side that recv `id` is ready
    // recv the notification that send `id` is ready with some `channel`
    void launch(rsrc_t resource, std::function<void()> callback) const;

    // resource: the notifier
    desc_t resource_description() const;

    // dependencies: wtvr dependencies until a recv can start

    void print(std::ostream& out) const {
      out << "notify_recv_ready {id = " << id << "}";
    }
  };

  struct wait_recv_ready_t {
    int id;
    int src;

    // wait until the src side says recv `id` is ready
    void launch(rsrc_t resource, std::function<void()> callback) const;

    // resource: the notifier
    desc_t resource_description() const;

    // dependencies: wtvr dependencies until a send can start

    void print(std::ostream& out) const {
      out << "wait_recv_ready {id = " << id << "}";
    }
  };

  struct send_t {
    int id;
    int dst;
    mem_t mem;

    // notify `dst` that `id` is ready and `channel` will be used
    // use a thread to send the data over that channel
    void launch(rsrc_t resource, std::function<void()> callback) const;

    // resources:
    //   the notifer,
    //   send channel,
    //   a thread,
    //   global buffer
    desc_t resource_description() const;

    // dependencies: a single dependencies on a wait_recv_ready_t

    void print(std::ostream& out) const {
      out << "send {id = " << id << "}";
    }
  };

  struct recv_t {
    int id;
    int src;
    mem_t mem;

    // get the `channel` from the notifier
    // use a thread to recv the data over that channel
    void launch(rsrc_t resource, std::function<void()> callback) const;

    // resources:
    //   the notifier,
    //   communicator,
    //   a thread
    //   global buffer
    desc_t resource_description() const;

    // dependencies: a single dependencies on a notify_recv_ready_t

    void print(std::ostream& out) const {
      out << "recv {id = " << id  << "}";
    }
  };

  using op_t = std::variant<
#ifdef CPU_EXEC
    cpu_touch_t,
    cpu_einsummable_t,
    cpu_evict_t,
    cpu_load_t,
#endif
    notify_recv_ready_t,
    wait_recv_ready_t,
    send_t,
    recv_t,
    dummy_t>;

  struct node_t {
    desc_t resource_description() const;

    void launch(rsrc_t resource, std::function<void()> callback) const;

    void print(std::ostream& out) const;

    op_t op;

    vector<int> inns;

    vector<int> outs;
  };

  vector<node_t> nodes;

#ifdef CPU_EXEC
  // When compiling exec graphs, einsummable nodes are built into
  // this particular executor. An executor could be considered a "resource"
  // that gets passed in to the einsummable nodes, but the same
  // cpu executor would have to get passed in. Also the cpu executor's state
  // does not change during execution, only during compilation.
  cpu_kernel_executor_t& cpu_executor;
#endif

private:
  int insert(op_t const& op, vector<int> const& inns);
};
