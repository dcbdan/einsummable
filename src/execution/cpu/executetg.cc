#include "executetg.h"

kernel_manager_t make_kernel_manager(taskgraph_t const& taskgraph) {
  kernel_manager_t ret;
  update_kernel_manager(ret, taskgraph);
  return ret;
}

void update_kernel_manager(
  kernel_manager_t& ret,
  taskgraph_t const& taskgraph)
{
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_apply()) {
      auto const& e = node.op.get_apply().einsummable;
      if(!ret.build(e)) {
        throw std::runtime_error(
          "could not build a kernel for " + write_with_ss(e));
      }
    }
  }
}

void execute_taskgraph(
  taskgraph_t const& taskgraph,
  execute_taskgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi,
  map<int, buffer_t>& tensors)
{
  executetg_ns::state_t state(mpi, taskgraph, kernel_manager, tensors);

  vector<std::thread> runners;
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

void execute_taskgraph_in_order(
  taskgraph_t const& taskgraph,
  vector<_tg_op_t> const& ops,
  kernel_manager_t const& kernel_manager,
  map<int, buffer_t>& tensors)
{
  map<int, int> usage_counts;
  auto register_usage = [&](int id) {
    auto iter = usage_counts.find(id);
    if(iter == usage_counts.end()) {
      auto const& node = taskgraph.nodes[id];
      if(node.outs.size() == 1) {
        tensors.erase(id);
      } else {
        usage_counts.insert({id, node.outs.size() - 1});
      }
    } else {
      int& cnt = iter->second;
      cnt--;
      if(cnt == 0) {
        auto const& node = taskgraph.nodes[id];
        if(!node.is_save) {
          tensors.erase(id);
        }
        usage_counts.erase(iter);
      }
    }
  };

  // partialize_id -> unit -> (inn, touch) pairs
  map<int, vector<vector<tuple<int, touch_t>>>> touches;
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_partialize()) {
      auto const& p = node.op.get_partialize();
      touches.insert({id, p.as_touches_from()});
    }
  }

  buffer_t workspace = make_buffer(1);

  for(auto const& op: ops) {
    if(op.is_miscop()) {
      int const& id = op.get_task_id();

      auto const& node = taskgraph.nodes[id];
      if(node.op.is_move()) {
        throw std::runtime_error(
          "single threaded execution engine cannot do moves!");
      } else if(node.op.is_partialize()) {
        // this probably should not happen, but assume all that
        // work is done in the apply
      } else if(node.op.is_input()) {
        // nothing to do
      } else if(node.op.is_apply()) {
        auto const& [_0, inns, einsummable] = node.op.get_apply();

        vector<void const*> raw_inputs;
        raw_inputs.reserve(inns.size());
        for(auto const& inn_id: inns) {
          raw_inputs.push_back(tensors.at(inn_id)->raw());
        }

        uint64_t sz = kernel_manager.workspace_size(einsummable);
        if(sz > workspace->size) {
          workspace = make_buffer(sz);
        }

        buffer_t out_buffer = make_buffer(node.op.out_size());

        tuple<void*, uint64_t> ws { workspace->raw(), workspace->size };
        kernel_manager(einsummable, out_buffer->raw(), raw_inputs, ws);
        tensors.insert({id, out_buffer});

        for(auto const& inn_id: inns) {
          register_usage(inn_id);
        }
      } else {
        throw std::runtime_error("missing tg type");
      }
    } else {
      auto const& [id, unit_id, touch_id] = op.get_touchop();
      auto const& [inn_id, touch_op] = touches.at(id)[unit_id][touch_id];

      buffer_t out_buffer;
      {
        auto iter = tensors.find(id);
        if(iter == tensors.end()) {
          uint64_t sz = taskgraph.out_size(id);
          out_buffer = make_buffer(sz);
          tensors.insert({id, out_buffer});
        } else {
          out_buffer = iter->second;
        }
      }

      buffer_t const& inn_buffer = tensors.at(inn_id);
      kernel_manager(touch_op, out_buffer->raw(), inn_buffer->raw());

      register_usage(inn_id);
    }
  }
}

vector<_tg_op_t>
_make_taskgraph_order(taskgraph_t const& taskgraph, bool always_random)
{
  auto get_tensors_worked_on_at_op = [&taskgraph](_tg_op_t const& op) {
    set<int> ts;
    if(op.is_miscop()) {
      int const& tid = op.get_task_id();
      auto const& node = taskgraph.nodes[tid];
      if(node.op.is_apply()) {
        auto const& inns = node.op.get_apply().inns;
        ts = set<int>(inns.begin(), inns.end());
        ts.insert(tid);
      } else if(node.op.is_move()) {
        auto const& m = node.op.get_move();
        ts = set<int>{m.inn, tid};
      } else {
        throw std::runtime_error("must be apply or move");
      }
    } else if(op.is_touchop()) {
      auto const& [tid, unit_id, touch_id] = op.get_touchop();
      auto const& node = taskgraph.nodes[tid];
      auto const& p = node.op.get_partialize();
      int inn = p.get_inn_at(unit_id, touch_id);
      ts = set<int>{inn, tid};
    } else {
      throw std::runtime_error("should not reach");
    }
    return ts;
  };

  vector<_tg_op_t> ret;
  ret.reserve(2*taskgraph.nodes.size());

  vector<_tg_op_t> pending;

  // partialize id -> number of touches still needed to complete
  // all other id  -> number of inputs left until can start
  vector<int> num_rem;

  map<int, int> num_touch_rem;

  int num_completed = 0;

  using touchop_t = _tg_op_t::touchop_t;
  using miscop_t  = _tg_op_t::miscop_t;

  // inn id -> touches it is a part of
  map<int, vector<touchop_t>> id_to_touches;

  auto register_complete = [&](int id) {
    num_completed++;

    // add the corresponding touches
    auto iter = id_to_touches.find(id);
    if(iter != id_to_touches.end()) {
      for(auto const& touch: iter->second) {
        pending.push_back(touch);
      }
      id_to_touches.erase(iter);
    }

    // for all non partialize nodes, a dependency has been finished
    // (partialize nodes have their dependency satisified only after
    //  touches)
    auto const& node = taskgraph.nodes[id];
    for(int const& out: node.outs) {
      auto const& out_node = taskgraph.nodes[out];
      if(!out_node.op.is_partialize()) {
        int& cnt = num_rem[out];
        cnt--;
        if(cnt == 0) {
          pending.push_back(miscop_t { .task_id = out });
        }
      }
    }
  };
  auto register_complete_touch = [&](int partialize_id) {
    int& cnt = num_rem[partialize_id];
    cnt--;
    if(cnt == 0) {
      register_complete(partialize_id);
    }
  };

  {
    num_rem.reserve(taskgraph.nodes.size());
    vector<int> inputs;
    for(int id = 0; id != taskgraph.nodes.size(); ++id) {
      auto const& node = taskgraph.nodes[id];

      int n_inn = node.op.inputs().size();
      if(node.op.is_partialize()) {
        num_rem.push_back(0);
        int& num_touch_rem = num_rem.back();

        // fill out num_touch_rem and id_to_touches
        auto const& p = node.op.get_partialize();
        for(int unit_id = 0; unit_id != p.units.size(); ++unit_id) {
          auto const& unit = p.units[unit_id];
          for(int touch_id = 0; touch_id != unit.inputs.size(); ++touch_id) {
            auto const& inn_id = unit.inputs[touch_id].id;
            id_to_touches[inn_id].push_back(touchop_t {
              .task_id = id,
              .unit_id = unit_id,
              .touch_id = touch_id
            });

            num_touch_rem++;
          }
        }
      } else {
        num_rem.push_back(n_inn);
        if(n_inn == 0) {
          if(!node.op.is_input()) {
            throw std::runtime_error(
              "how can a node with zero inputs not be an input node?");
          }
          // this is an input node
          inputs.push_back(id);
        }
      }
    }
    // now that num rem has been setup, register that the
    // input tensors are ready
    for(auto const& id: inputs) {
      register_complete(id);
    }
  }

  while(pending.size() != 0) {
    _tg_op_t op;
    if(!always_random && ret.size() > 0) {
      // get the tensors being being worked with
      _tg_op_t const& last_op = ret.back();
      set<int> ts = get_tensors_worked_on_at_op(last_op);

      // just pick the first op that has shared bytes
      bool set_the_next_op = false;
      for(auto iter = pending.begin(); iter != pending.end(); ++iter) {
        auto const& p = *iter;
        set<int> ts_at_p = get_tensors_worked_on_at_op(p);
        set<int> in_both = set_intersection(ts, ts_at_p);
        if(in_both.size() > 0) {
          op = p;
          pending.erase(iter);
          set_the_next_op = true;
          break;
        }
      }
      if(!set_the_next_op) {
        // if none, just pick something random
        op = vector_random_pop(pending);
      }
    } else {
      op = vector_random_pop(pending);
    }

    ret.push_back(op);
    if(op.is_miscop()) {
      int const& id = op.get_task_id();
      register_complete(id);
    } else if(op.is_touchop()) {
      int const& partialize_id = op.get_task_id();
      register_complete_touch(partialize_id);
    } else {
      throw std::runtime_error("should not happen");
    }
  }

  if(num_completed != taskgraph.nodes.size()) {
    throw std::runtime_error("num completed != taskgraph nodes size");
  }

  return ret;
}

vector<_tg_op_t>
random_taskgraph_order(taskgraph_t const& taskgraph) {
  return _make_taskgraph_order(taskgraph, true);
}

vector<_tg_op_t>
temporal_taskgraph_order(taskgraph_t const& taskgraph) {
  return _make_taskgraph_order(taskgraph, false);
}

namespace executetg_ns {

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
    num_did_remaining(0)
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
      auto maybe_donate_inn = get_donate(which_apply);
      if(maybe_donate_inn) {
        int const& which_inn = maybe_donate_inn.value();
        out_buffer = inputs[which_inn];
      } else {
        // No donation is happening, allocate new memory
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

    cv_notify.notify_one();

    bool fini;
    {
      unique_lock lk(m_apply);
      num_apply_remaining--;
      fini = num_apply_remaining == 0;
    }
    if(fini) {
      cv_apply.notify_all();
    }
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
    cv_notify.notify_one();

    bool fini;
    {
      unique_lock lk(m_send);
      num_send_remaining--;
      fini = (num_send_remaining == 0);
    }
    if(fini) {
      cv_send.notify_all();
    }
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
    cv_notify.notify_one();
  }
}

optional<int> state_t::get_donate(int apply_id) {
  auto const& node  = taskgraph.nodes[apply_id];
  auto const& apply = node.op.get_apply();
  auto const& e     = apply.einsummable;
  auto const& inns  = apply.inns;

  // Get all inputs that the kernel manager says can be used as
  // both an input and an output.
  vector<int> donatables = kernel_manager.donatables(e);

  // Now make sure that the input is only used once.
  // (It might be better to instead make sure that this is the _last_ usage of the
  //  input, but that would require more concurrency management.)
  optional<int> ret;
  int donate_inn_id;
  for(int const& which_inn: donatables) {
    auto const& inn_id = inns[which_inn];
    auto const& inn_node = taskgraph.nodes[inn_id];
    if(inn_node.outs.size() == 1) {
      donate_inn_id = inn_id;
      ret = which_inn;
      break;
    }
  }

  if(!ret) {
    return ret;
  }

  {
    unique_lock lk(m_donate);
    donated.insert(donate_inn_id);
  }
  //DLINEOUT("num donated: " << donated.size());
  return ret;
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
      cv_apply.notify_one();
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
    cv_apply.notify_one();
  } else if(node.op.is_move()) {
    {
      unique_lock lk(m_send);
      pending_sends.push(id);
    }
    cv_send.notify_one();
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
        cv_apply.notify_one();
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
            cv_apply.notify_one();
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

    bool was_donated;
    {
      unique_lock lk(m_donate);
      auto iter = donated.find(id);
      was_donated = iter != donated.end();
      if(was_donated) {
        donated.erase(iter);
      }
    }

    if(!is_save && !was_donated) {
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

}

