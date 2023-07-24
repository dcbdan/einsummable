#pragma once

#pragma once
#include "../../base/setup.h"

#include "../../einsummable/taskgraph.h"
#include "../../einsummable/dbuffer.h"

#include "mpi_class.h"
#include "kernels.h"
#include "workspace.h"

#include <thread>
#include <mutex>
#include <condition_variable>

struct execute_taskgraph_settings_t {
  int num_apply_runner;
  int num_send_runner;
  int num_recv_runner;

  static execute_taskgraph_settings_t default_settings() {
    // standards says hardware_concurrency could return 0
    // if not computable or well defined
    int num_threads = std::max(1u, std::thread::hardware_concurrency());

    // TODO: When the number of send and recv threads are really large,
    //       it scales poorly.
    return execute_taskgraph_settings_t {
      .num_apply_runner = num_threads,
      .num_send_runner  = 4,
      .num_recv_runner  = 4
    };
  }
};

kernel_manager_t make_kernel_manager(taskgraph_t const& taskgraph);

void update_kernel_manager(kernel_manager_t& km, taskgraph_t const& taskgraph);

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute_taskgraph(
  taskgraph_t const& taskgraph,
  execute_taskgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the taskgraph must be single-node
  map<int, buffer_t>& tensors);

void execute_taskgraph_in_order(
  taskgraph_t const& taskgraph,
  vector<_tg_op_t> const& ops_in_order,
  kernel_manager_t const& kernel_manager,
  map<int, buffer_t>& tensors);

vector<_tg_op_t>
random_taskgraph_order(taskgraph_t const& taskgraph);

// TODO
vector<_tg_op_t>
temporal_taskgraph_order(taskgraph_t const& taskgraph);

namespace executetg_ns {
  using std::thread;
  using std::queue;
  using std::unordered_map;
  using std::unordered_set;
  using std::mutex;
  using std::condition_variable;
  using std::unique_lock;

  struct touch_unit_t {
    int partialize_id;
    int unit_id;
  };

  // Possible states for working on a touch unit
  //   state 1: working     & empty queue
  //   state 2: not working & empty queue
  //   state 3: working     & non-empty queue
  // (here working == in pending_applys or being executed)
  struct touch_unit_state_t {
    touch_unit_state_t()
      : busy(false)
    {}

    bool busy;
    queue<int> waiting;
  };

  struct which_touch_t {
    int partialize_id;
    int unit_id;
    int touch_id;

    touch_unit_t as_touch_unit() const {
      return touch_unit_t { partialize_id, unit_id };
    }
  };

  struct state_t {
    state_t(
      mpi_t* mpi,
      taskgraph_t const& taskgraph,
      kernel_manager_t const& kernel_manager,
      map<int, buffer_t>& tensors);

    void event_loop();
    void event_loop_did_node(int id);
    void event_loop_did_touch(which_touch_t const& info);
    void event_loop_launch(int id);
    void event_loop_launch_touch(int inn_id, int partialize_id);
    void event_loop_register_usage(int id);

    void apply_runner(int runner_id);
    void send_runner(int runner_id);
    void recv_runner(int runner_id);

    // Mark an input of this apply for donation. If something is marked for donation,
    // return the corresponding input to the apply.
    optional<int> get_donate(int apply_id);

    int const& get_inn_from_which_touch(which_touch_t const& info) const;
    tuple<int, touch_t> const& get_touch_info(which_touch_t const&) const;

    // grab tensors (and allocate if necc) under m_tensors mutex
    tuple<
      vector<buffer_t>,
      vector<buffer_t> >
    get_buffers(
      vector<tuple<uint64_t, int>> const& which_allocate,
      vector<int> const& which_get);

    int this_rank;
    mpi_t* mpi;
    taskgraph_t const& taskgraph;
    kernel_manager_t const& kernel_manager;

    map<int, buffer_t>& tensors;

    // this holds the kernel_manager and queries it
    workspace_manager_t workspace_manager;

    // the number of remaining events until a node can start
    unordered_map<int, int> num_remaining;
    // For partializes, initially the number of touches remaining
    // For sends, initially 1
    // For applys, initially the number of inputs

    unordered_map<int, int> num_usage_remaining;
    // the number of times a tensor is used until it won't be used again

    // partialize_id -> unit -> (inn, touch) pairs
    unordered_map<int, vector<vector<tuple<int, touch_t>>>> touches;
    // this is read only after initialization

    int num_apply_remaining;
    int num_send_remaining;
    int num_recv_post_remaining;
    int num_did_remaining;

    queue<int> pending_sends;
    queue<int> pending_applys;
    queue<which_touch_t> pending_touches;

    map<touch_unit_t, touch_unit_state_t> units_in_progress;
    // ^ any touch unit with multiple touches must have a key
    //   here until the corresponding partialize is done

    queue<int> did_nodes;
    queue<which_touch_t> did_touches;

    // for offloading did things into the event loop
    queue<int> here_nodes;
    queue<which_touch_t> here_touches;

    // these tensors have been donated, so do not delete them
    set<int> donated;

    // for tensors
    mutex m_tensors;

    // for
    //   did_nodes
    //   did_touches
    mutex m_notify;
    condition_variable cv_notify;

    // for
    //   num_apply_remaining,
    //   pending_applys,
    //   pending_touches
    mutex m_apply;
    condition_variable cv_apply;

    // for
    //   num_send_remaining
    //   pending_sends
    mutex m_send;
    condition_variable cv_send;

    // for num_recv_post_remaining
    mutex m_recv;
    condition_variable cv_recv;

    // for donated
    mutex m_donate;

    // State only updated by the event loop:
    //   num_remaining
    //   num_usage_remaining
    //   units_in_progress
    //   num_did_remaining
    //   here_nodes
    //   here_touches
  };

  bool operator==(touch_unit_t const&, touch_unit_t const&);
  bool operator< (touch_unit_t const&, touch_unit_t const&);

  std::ostream& operator<<(std::ostream&, which_touch_t const&);
}

