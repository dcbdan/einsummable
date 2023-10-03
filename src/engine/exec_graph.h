#pragma once
#include "../base/setup.h"

#include "resource_manager.h"

#include "../einsummable/memgraph.h"

#ifdef CPU_EXEC
#include "cpu/kernel_executor.h"
#endif

#ifdef GPU_EXEC
#include "gpu/gpu_kernel_manager.h"
#endif

struct exec_graph_t {
#ifdef CPU_EXEC
  exec_graph_t(cpu_kernel_executor_t& e)
    : cpu_executor(e)
  {}

  static exec_graph_t make_cpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    cpu_kernel_executor_t& cpu_executor);
#endif

#ifdef GPU_EXEC
  exec_graph_t(kernel_manager_t& km)
    : gpu_km(km)
  {}

  static exec_graph_t make_gpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    kernel_manager_t& gpu_km);
#endif

  struct op_base_t {
    virtual void launch(resource_ptr_t resource, std::function<void()> callback) const = 0;
    virtual desc_ptr_t resource_description() const = 0;
    virtual void print(std::ostream& out) const = 0;
  };

  using op_ptr_t = std::shared_ptr<op_base_t>;

  struct node_t {
    op_ptr_t op;

    vector<int> inns;
    vector<int> outs;

    inline desc_ptr_t resource_description() const { return op->resource_description(); }

    inline void launch(resource_ptr_t resource, std::function<void()> callback) const {
      op->launch(resource, callback);
    }

    inline void print(std::ostream& out) const { op->print(out); }
  };

  vector<node_t> nodes;

  struct dummy_t : op_base_t {
    void launch(resource_ptr_t resource, std::function<void()> callback) const {
      callback();
    }
    desc_ptr_t resource_description() const {
      return resource_manager_t::make_desc(vector<desc_ptr_t>());
    }
    void print(std::ostream& out) const { out << "dummy"; }
  };

  struct notify_recv_ready_t : op_base_t {
    notify_recv_ready_t(int a, int b)
      : id(a), dst(b)
    {}

    int id;
    int dst;

    // notify the dst side that recv `id` is ready
    // recv the notification that send `id` is ready with some `channel`
    void launch(resource_ptr_t resource, std::function<void()> callback) const;

    // resource: the notifier
    desc_ptr_t resource_description() const;

    // dependencies: wtvr dependencies until a recv can start

    void print(std::ostream& out) const {
      out << "notify_recv_ready {id = " << id << "}";
    }
  };

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

  struct wait_recv_ready_t : op_base_t {
    wait_recv_ready_t(int a, int b)
      : id(a), src(b)
    {}

    int id;
    int src;

    // wait until the src side says recv `id` is ready
    void launch(resource_ptr_t resource, std::function<void()> callback) const;

    // resource: the notifier
    desc_ptr_t resource_description() const;

    // dependencies: wtvr dependencies until a send can start

    void print(std::ostream& out) const {
      out << "wait_recv_ready {id = " << id << "}";
    }
  };

  // TODO: send and recv need to specify which buffer in global
  //       buffers we're talking about ;; right now the assumption
  //       is always buffer zero, but we want to support gpu to gpu
  //       communication, and maybe we want to do send and recv with
  //       ucx. (Note: ucx communicator doesn't support such a setup since
  //             we don't allow sending to ourselves)

  struct send_t : op_base_t {
    send_t(int a, int b, mem_t const& c)
      : id(a), dst(b), mem(c)
    {}

    int id;
    int dst;
    mem_t mem;

    // notify `dst` that `id` is ready and `channel` will be used
    // use a thread to send the data over that channel
    void launch(resource_ptr_t resource, std::function<void()> callback) const;

    // resources:
    //   the notifer,
    //   send channel,
    //   a thread,
    //   global buffer
    desc_ptr_t resource_description() const;

    // dependencies: a single dependencies on a wait_recv_ready_t

    void print(std::ostream& out) const {
      out << "send {id = " << id << "}";
    }
  };

  struct recv_t : op_base_t {
    recv_t(int a, int b, mem_t const& c)
      : id(a), src(b), mem(c)
    {}

    int id;
    int src;
    mem_t mem;

    // get the `channel` from the notifier
    // use a thread to recv the data over that channel
    void launch(resource_ptr_t resource, std::function<void()> callback) const;

    // resources:
    //   the notifier,
    //   communicator,
    //   a thread
    //   global buffer
    desc_ptr_t resource_description() const;

    // dependencies: a single dependencies on a notify_recv_ready_t

    void print(std::ostream& out) const {
      out << "recv {id = " << id  << "}";
    }
  };

#ifdef CPU_EXEC
  // When compiling exec graphs, einsummable nodes are built into
  // this particular executor. An executor could be considered a "resource"
  // that gets passed in to the einsummable nodes, but the same
  // cpu executor would have to get passed in. Also the cpu executor's state
  // does not change during execution, only during compilation.
  cpu_kernel_executor_t& cpu_executor;
  // TODO: prefer to not have the kernel manager as part of this class...
#endif

#ifdef GPU_EXEC
  kernel_manager_t& gpu_km;
#endif

private:
  int insert(op_ptr_t op, vector<int> const& inns);
};


