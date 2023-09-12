#include "bcost.h"

cluster_settings_t::cluster_settings_t(int n_node, int n_worker_per) {
  start_cost = 1e-4;
  speed_per_byte = vector<vector<double>>(n_node, vector<double>(n_node, 1e9));
  for(int i = 0; i != n_node; ++i) {
    speed_per_byte[i][i] *= 3;
  }
  nworkers_per_node = vector<int>(n_node, n_worker_per);
}

double bytes_cost(
  taskgraph_t const& taskgraph,
  cluster_settings_t const& settings)
{
  bytes_cost_state_t state(taskgraph, settings);

  while(state.step()) {}

  return state.time;
}

bytes_cost_state_t::bytes_cost_state_t(
  taskgraph_t const& tg,
  cluster_settings_t const& sts)
  : taskgraph(tg), settings(sts),
    time(0.0),
    num_avail_workers(sts.nworkers_per_node),
    counts(tg.nodes.size(), 0),
    num_finished(0)
{
  for(int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
    auto const& node = taskgraph.nodes[tid];
    int& cnt = counts[tid];
    cnt = node.op.inputs().size();
    if(cnt == 0) {
      pending_tasks.push_back(tid);
    }
  }
}

bool bytes_cost_state_t::step() {
  // Step 1: start all the work we can
  // Step 2: finish the next worker
  vector<int> cannot_start_tasks;
  while(pending_tasks.size() != 0) {
    int tid = pending_tasks.back();
    pending_tasks.pop_back();

    auto const& node = taskgraph.nodes[tid];
    auto const& op = node.op;
    if(op.is_input()) {
      decrement(node.outs);
    } else {
      int loc = op.out_loc();
      if(num_avail_workers[loc] == 0) {
        cannot_start_tasks.push_back(tid);
      } else {
        num_avail_workers[loc] -= 1;
        double cost = get_cost(op);
        busy_workers.push(bytes_cost_busy_t {
          .finish = time + cost,
          .tid = tid,
          .loc = loc
        });
      }
    }
  }

  pending_tasks = cannot_start_tasks;

  if(busy_workers.size() > 0) {
    auto [new_time, tid, loc] = busy_workers.top();
    busy_workers.pop();

    time = new_time;
    decrement(taskgraph.nodes[tid].outs);
    num_avail_workers[loc]++;

    return true; // yes, completed another task
  } else {
    if(pending_tasks.size() > 0) {
      throw std::runtime_error("pending tasks should be empty");
    }
    for(int const& cnt: counts) {
      if(cnt != 0) {
        throw std::runtime_error("some thing hasn't even started");
      }
    }
    if(num_finished != taskgraph.nodes.size()) {
      throw std::runtime_error("did not finish all nodes");
    }
    return false; // no, no more tasks to complete
  }

}

void bytes_cost_state_t::decrement(set<int> const& outs) {
  num_finished++;
  for(auto const& out: outs) {
    int& cnt = counts[out];
    cnt -= 1;
    if(cnt == 0) {
      pending_tasks.push_back(out);
    }
  }
}

double bytes_cost_state_t::get_cost(taskgraph_t::op_t const& op)
{
  if(op.is_apply()) {
    // compute the number of bytes touched by this op
    // (all the inputs + the output)
    auto const& apply = op.get_apply();
    auto const& e = apply.einsummable;
    uint64_t total = 0;
    auto inn_shapes = e.inn_shapes();
    for(int i = 0; i != inn_shapes.size(); ++i) {
      auto maybe_dtype = e.join.inn_dtype(i);
      dtype_t dtype = maybe_dtype ? maybe_dtype.value() : dtype_t::f64;
      total += dtype_size(dtype) * product(inn_shapes[i]);
    }
    total += e.out_size();

    int const& loc = apply.loc;

    return settings.start_cost +
      double(total) / settings.speed_per_byte[loc][loc];
  } else if(op.is_move()) {
    // touch the bytes on the input side and on the output side
    auto const& [src,dst,_,total] = op.get_move();
    return settings.start_cost +
      double(total) / settings.speed_per_byte[src][src] +
      double(total) / settings.speed_per_byte[src][dst];
  } else if(op.is_partialize()) {
    auto const& partialize = op.get_partialize();
    uint64_t total = 0;

    for(auto const& [_, touch]: partialize.as_touches_from_flat()) {
      total += product(vector_from_each_member(touch.selection, uint64_t, d_inn));
      total += product(vector_from_each_member(touch.selection, uint64_t, d_out));
    }

    total *= dtype_size(partialize.dtype);

    int const& loc = partialize.loc;
    return settings.start_cost +
      double(total) / settings.speed_per_byte[loc][loc];
  } else if(op.is_input()) {
    throw std::runtime_error("should not be getting cost for input op");
    return 0.0;
  } else {
    throw std::runtime_error("should not reach: bcost");
  }
}

bool operator>(bytes_cost_busy_t const& lhs, bytes_cost_busy_t const& rhs) {
  return lhs.finish > rhs.finish;
}

