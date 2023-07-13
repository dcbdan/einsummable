#include "executetg2.h"

#include "kernels.h"
#include "workspace.h"

#include <thread>
#include <mutex>
#include <condition_variable>

#include <fstream> // TODO

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

bool operator==(touch_unit_t const& lhs, touch_unit_t const& rhs);
bool operator< (touch_unit_t const& lhs, touch_unit_t const& rhs);

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
std::ostream& operator<<(std::ostream&, which_touch_t const&);

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

  // State only updated by the event loop:
  //   num_remaining
  //   num_usage_remaining
  //   units_in_progress
  //   num_did_remaining
  //   here_nodes
  //   here_touches

  mutex m_print; // TODO
  std::ofstream fff;
  void print(int n) { unique_lock lk(m_print); fff << n << ","; fff.flush(); }
};

state_t::state_t(
  mpi_t* mpi_,
  taskgraph_t const& taskgraph_,
  kernel_manager_t const& kernel_manager_,
  map<int, buffer_t>& tensors_)
  : mpi(mpi_), taskgraph(taskgraph_), kernel_manager(kernel_manager_),
    tensors(tensors_), workspace_manager(kernel_manager_),
    num_apply_remaining(0),
    num_send_remaining(0),
    num_recv_post_remaining(0),
    num_did_remaining(0),
    fff("fff.nodes")
{
  this_rank = bool(mpi) ? mpi->this_rank : 0;

  int num_nodes = taskgraph.nodes.size();
  for(int id = 0; id != num_nodes; ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()) {
      if(node.op.out_loc() != this_rank) {
        continue;
      }
      num_did_remaining++;
      here_nodes.push(id);
    } else if(node.op.is_apply()) {
      if(node.op.out_loc() != this_rank) {
        continue;
      }
      num_did_remaining++;
      num_apply_remaining++;
      auto const& inns = node.op.inputs();
      num_remaining[id] = inns.size();
      for(int const& inn: inns) {
        num_usage_remaining[inn]++;
      }
    } else if(node.op.is_move()) {
      auto const& move = node.op.get_move();
      if(move.src == this_rank) {
        num_did_remaining++;
        num_send_remaining++;
        num_remaining[id] = 1;
        num_usage_remaining[move.inn]++;
      } else if(move.dst == this_rank) {
        num_did_remaining++;
        num_recv_post_remaining++;
      }
    } else if(node.op.is_partialize()) {
      if(node.op.out_loc() != this_rank) {
        continue;
      }
      auto const& partialize = node.op.get_partialize();

      touches.insert({id, partialize.as_touches_from()});
      auto const& ts = touches[id];

      int& n_touches = num_remaining[id];
      n_touches = 0;
      for(vector<tuple<int, touch_t>> const& m: ts) {
        n_touches += m.size();
        for(auto const& [inn,_]: m) {
          num_usage_remaining[inn]++;
        }
      }

      // plus one for the partialize
      // and plus one for each touch
      num_did_remaining += (n_touches + 1);

      num_apply_remaining += n_touches;
    } else {
      throw std::runtime_error("missing tg node type in execute taskgraph state init");
    }
  }
}

void state_t::apply_runner(int runner_id)
{
  int which_apply;
  which_touch_t which_touch;
  bool doing_touch;
  while(true) {
    {
      unique_lock lk(m_apply);
      cv_apply.wait(lk, [&, this] {
        if(num_apply_remaining == 0) {
          return true;
        }
        if(pending_touches.size() > 0) {
          doing_touch = true;
          which_touch = pending_touches.front();
          pending_touches.pop();
          return true;
        }
        if(pending_applys.size() > 0) {
          doing_touch = false;
          which_apply = pending_applys.front();
          pending_applys.pop();
          return true;
        }
        return false;
      });
      if(num_apply_remaining == 0) {
        //DLINEOUT("exiting apply runner");
        return;
      }
    }

    if(doing_touch) {
      auto const& [inn_id, touch_op] = get_touch_info(which_touch);

      int const& partialize_id = which_touch.partialize_id;
      uint64_t partialize_size = taskgraph.nodes[partialize_id].op.out_size();

      auto [_ps, _is] = get_buffers(
        { {partialize_size, partialize_id} },
        { inn_id });

      buffer_t& out_buffer = _ps[0];
      buffer_t& inn_buffer = _is[0];

      //DOUT("TOUCH " << which_touch);
      kernel_manager(touch_op, out_buffer->data, inn_buffer->data);

      {
        unique_lock lk(m_notify);
        did_touches.push(which_touch);
      }
    } else {
      auto const& node = taskgraph.nodes[which_apply];
      auto const& [_0, inns, einsummable] = node.op.get_apply();

      auto [workspace, which_workspace] = workspace_manager.get(einsummable);

      auto [_1, inputs] = get_buffers({}, inns);

      buffer_t out_buffer;

      // Can we donate one of the input buffers to
      // this computation?
      // TODO

      // If not, allocate.
      if(!out_buffer) {
        out_buffer = make_buffer(node.op.out_size());
      }

      vector<void const*> raw_inputs;
      raw_inputs.reserve(inputs.size());
      for(auto const& buffer: inputs) {
        raw_inputs.push_back(buffer->data);
      }

      //DOUT("EINSUMMABLE " << which_apply);
      kernel_manager(einsummable, out_buffer->data, raw_inputs, workspace);
      workspace_manager.release(which_workspace);

      // Note: Even if out_buffer was donated, this is fine. When
      //       the donated input gets removed from tensors, the
      //       buffer won't get deleted since its a shared pointer.
      {
        std::unique_lock lk(m_tensors);
        tensors.insert_or_assign(which_apply, out_buffer);
      }

      {
        unique_lock lk(m_notify);
        did_nodes.push(which_apply);
      }
    }

    cv_notify.notify_all();

    {
      unique_lock lk(m_apply);
      num_apply_remaining--;
    }
    cv_apply.notify_all();
  }
}

void state_t::send_runner(int runner_id) {
  int send_id;
  while(true) {
    {
      unique_lock lk(m_send);
      cv_send.wait(lk, [&, this] {
        if(num_send_remaining == 0) {
          return true;
        }
        if(pending_sends.size() > 0) {
          send_id = pending_sends.front();
          pending_sends.pop();
          return true;
        }
        return false;
      });
      if(num_send_remaining == 0) {
        return;
      }
    }

    auto const& node = taskgraph.nodes[send_id];
    auto const& [_0, dst, inn_id, _1] = node.op.get_move();

    mpi->send_int(send_id, dst, mpi->max_tag);

    auto [_, read_buffers] = get_buffers({}, {inn_id});
    auto& buffer = read_buffers[0];

    mpi->send(buffer, dst, send_id);

    {
      unique_lock lk(m_notify);
      did_nodes.push(send_id);
    }
    cv_notify.notify_all();

    {
      unique_lock lk(m_send);
      num_send_remaining--;
    }
    cv_send.notify_all();
  }
}

void state_t::recv_runner(int runner_id) {
  while(true)
  {
    {
      unique_lock lk(m_recv);

      if(num_recv_post_remaining == 0) {
        return;
      }
      num_recv_post_remaining -= 1;
      // (don't do this after here because if there are two
      //  recv runners for one post remaining, then mpi recv could be
      //  called twice waiting for the same data and would would hang)
    }

    int recv_id = mpi->recv_int_from_anywhere(mpi->max_tag);

    auto const& node = taskgraph.nodes[recv_id];
    auto const& [src, _0, _1, size] = node.op.get_move();

    auto [should_be_new_buffers, _] = get_buffers({ {size, recv_id} }, {});
    auto& recv_buffer = should_be_new_buffers[0];

    mpi->recv(recv_buffer, src, recv_id);

    {
      unique_lock lk(m_notify);
      did_nodes.push(recv_id);
    }
    cv_notify.notify_all();
  }
}

void state_t::event_loop()
{
  while(true) {
    while(here_nodes.size() > 0 || here_touches.size() > 0) {
      if(here_nodes.size() > 0) {
        int id = here_nodes.front();
        here_nodes.pop();
        //DOUT("event loop did node " << id);
        event_loop_did_node(id);
        num_did_remaining--;
      }
      if(here_touches.size() > 0) {
        auto info = here_touches.front();
        here_touches.pop();
        //DOUT("event loop did touch " << info);
        event_loop_did_touch(info);
        num_did_remaining--;
      }
    }
    if(num_did_remaining == 0) {
      return;
    }
    unique_lock lk(m_notify);
    cv_notify.wait(lk, [this] {
      if(did_nodes.size() > 0) {
        here_nodes = did_nodes;
        did_nodes = queue<int>();
      }
      if(did_touches.size() > 0) {
        here_touches = did_touches;
        did_touches = queue<which_touch_t>();
      }
      return here_nodes.size() > 0 || here_touches.size() > 0;
    });
  }
}

void state_t::event_loop_did_node(int id)
{
  auto const& node = taskgraph.nodes[id];

  // register the usage for each tensor used
  bool is_send = false;
  if(node.op.is_input()) {
    // there are no inputs to input nodes
  } else if(node.op.is_apply()) {
    for(auto const& inn: node.op.inputs()) {
      event_loop_register_usage(inn);
    }
  } else if(node.op.is_move()) {
    auto const& move = node.op.get_move();
    if(move.src == this_rank) {
      event_loop_register_usage(move.inn);
      is_send = true;
    }
  } else if(node.op.is_partialize()) {
    // usage is registered after touches for partialize nodes,
    // not here

    // however, units_in_progress needs to be cleaned up
    auto const& ts = touches.at(id);
    for(int unit_id = 0; unit_id != ts.size(); ++unit_id) {
      int num_touches_in_unit = ts[unit_id].size();
      if(num_touches_in_unit > 1) {
        units_in_progress.erase(touch_unit_t { id, unit_id });
      }
    }
  } else {
    throw std::runtime_error("missing node type: event loop did node");
  }

  // register the outputs and possibly launch them
  if(!is_send) {
    for(auto const& out: node.outs) {
      auto const& out_node = taskgraph.nodes[out];
      if(out_node.op.is_partialize()) {
        //DOUT("event_loop_launch_touch " << id << " " << out);
        event_loop_launch_touch(id, out);
      } else {
        int& cnt = num_remaining.at(out);
        cnt--;
        if(cnt == 0) {
          event_loop_launch(out);
          num_remaining.erase(out);
        } else if(cnt < 0) {
          throw std::runtime_error("cnt should not be negative!");
        }
      }
    }
  }
}

void state_t::event_loop_did_touch(which_touch_t const& info)
{
  //DLINEOUT("did touch                         " << info);
  auto const& [partialize_id, unit_id, touch_id] = info;

  int const& inn_id = get_inn_from_which_touch(info);
  event_loop_register_usage(inn_id);

  {
    int& cnt = num_remaining.at(partialize_id);
    cnt--;

    if(cnt == 0) {
      // all the partializes have been done, mark this tensor
      // as computed so it can be processed here in the event
      // loop as part of here_nodes
      here_nodes.push(partialize_id);
      num_remaining.erase(partialize_id);
    } else if(cnt < 0) {
      throw std::runtime_error("cnt should not be  negative!");
    }
  }

  // No touch in a unit can happen at the same time. If more touches
  // from this unit are available, add them to pending touches and
  // notify the workers.
  auto iter = units_in_progress.find(info.as_touch_unit());
  if(iter != units_in_progress.end()) {
    touch_unit_state_t& unit_state = iter->second;
    queue<int>& next_touch_ids = unit_state.waiting;
    if(next_touch_ids.size() > 0) {
      int next_touch_id = next_touch_ids.front();
      next_touch_ids.pop();
      //DLINEOUT("inserting from unit state         "
      //  << (which_touch_t { partialize_id, unit_id, next_touch_id }));
      {
        unique_lock lk(m_apply);
        pending_touches.push(which_touch_t { partialize_id, unit_id, next_touch_id });
      }
      cv_apply.notify_all();
    } else {
      unit_state.busy = false;
    }
  }
}

void state_t::event_loop_launch(int id) {
  auto const& node = taskgraph.nodes[id];
  if(node.op.is_apply()) {
    {
      unique_lock lk(m_apply);
      pending_applys.push(id);
    }
    cv_apply.notify_all();
  } else if(node.op.is_move()) {
    {
      unique_lock lk(m_send);
      pending_sends.push(id);
    }
    cv_send.notify_all();
  } else {
    throw std::runtime_error("event loop launch only for apply and send");
  }
}

void state_t::event_loop_launch_touch(int inn_id, int partialize_id) {
  auto const& ts = touches.at(partialize_id);
  for(int unit_id = 0; unit_id != ts.size(); ++unit_id) {
    auto const& items = ts[unit_id];
    if(items.size() == 1) {
      // this unit is a singleton, no need to dance with units_in_progress
      auto const& [touch_inn, _] = items[0];
      if(inn_id == touch_inn) {
        {
          unique_lock lk(m_apply);
          pending_touches.push(which_touch_t { partialize_id, unit_id, 0 });
        }
        cv_apply.notify_all();
      }
    } else {
      touch_unit_t touch_unit { partialize_id, unit_id };
      auto& unit_state = units_in_progress[touch_unit];
      for(int touch_id = 0; touch_id != items.size(); ++touch_id) {
        auto const& [touch_inn, _] = items[touch_id];
        if(inn_id == touch_inn) {
          if(unit_state.busy) {
            //DLINEOUT("having agg wait                   "
            //    << (which_touch_t { partialize_id, unit_id, touch_id}));
            unit_state.waiting.push(touch_id);
          } else {
            //DLINEOUT("launching agg                     " <<
            //    (which_touch_t { partialize_id, unit_id, touch_id}));
            {
              unique_lock lk(m_apply);
              pending_touches.push(which_touch_t { partialize_id, unit_id, touch_id });
            }
            cv_apply.notify_all();
            unit_state.busy = true;
          }
        }
      }
    }
  }
}

void state_t::event_loop_register_usage(int id) {
  auto iter = num_usage_remaining.find(id);
  if(iter == num_usage_remaining.end()) {
    throw std::runtime_error("could not find usage count");
  }

  int& cnt = iter->second;
  cnt--;

  if(cnt == 0) {
    bool const& is_save = taskgraph.nodes[id].is_save;
    if(!is_save) {
      unique_lock lk(m_tensors);
      tensors.erase(id);
    }
    num_usage_remaining.erase(id);
  } else if(cnt < 0) {
    throw std::runtime_error("usage count should not be less than zero");
  }
}

tuple<int, touch_t> const&
state_t::get_touch_info(which_touch_t const& info) const
{
  auto const& [partialize_id, unit_id, touch_id] = info;
  return touches.at(partialize_id)[unit_id][touch_id];
}

int const&
state_t::get_inn_from_which_touch(which_touch_t const& info) const {
  return std::get<0>(get_touch_info(info));
}

tuple<
  vector<buffer_t>,
  vector<buffer_t> >
state_t::get_buffers(
  vector<tuple<uint64_t, int>> const& which_writes,
  vector<int>                  const& which_reads)
{
  std::unique_lock lk(m_tensors);

  vector<buffer_t> writes;
  writes.reserve(which_writes.size());
  for(auto const& [size, id]: which_writes) {
    if(tensors.count(id) == 0) {
      tensors.insert_or_assign(
        id,
        make_buffer(size)
      );
    }
    writes.push_back(tensors.at(id));
  }

  vector<buffer_t> reads;
  reads.reserve(which_reads.size());
  for(auto const& id: which_reads) {
    reads.push_back(tensors.at(id));
  }

  return {writes, reads};
}

void execute_taskgraph_2(
  taskgraph_t const& taskgraph,
  execute_taskgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi,
  map<int, buffer_t>& tensors)
{
  state_t state(mpi, taskgraph, kernel_manager, tensors);

  vector<thread> runners;
  {
    int const& n_apply = settings.num_apply_runner;
    int const& n_send = settings.num_send_runner;
    int const& n_recv = settings.num_recv_runner;
    runners.reserve(n_apply + n_send + n_recv);
    for(int i = 0; i != n_apply; ++i) {
      runners.emplace_back([&state, i](){ return state.apply_runner(i); });
    }
    for(int i = 0; i != n_send; ++i) {
      runners.emplace_back([&state, i](){ return state.send_runner(i); });
    }
    for(int i = 0; i != n_recv; ++i) {
      runners.emplace_back([&state, i](){ return state.recv_runner(i); });
    }
  }

  state.event_loop();

  for(auto& t: runners) {
    t.join();
  }
}

bool operator==(touch_unit_t const& lhs, touch_unit_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator< (touch_unit_t const& lhs, touch_unit_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
std::ostream& operator<<(std::ostream& out, which_touch_t const& info) {
  out << "wt{partialize " << info.partialize_id;
  out << " unit " << info.unit_id;
  out << " touch " << info.touch_id;
  out << "}";
  return out;
}

