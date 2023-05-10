#include "coster.h"

cluster_t cluster_t::make(
  vector<cluster_t::device_t> const& devices,
  vector<cluster_t::connection_t> const& cs)
{
  // Remove duplicate connections in cs
  // by summing the corresponding bandwidth

  vector<connection_t> connections;
  map<tuple<int, int>, int> to_connection;

  for(auto const& c: cs) {
    auto key = c.key();

    if(to_connection.count(key) > 0) {
      connections[to_connection[key]].bandwidth += c.bandwidth;
    } else {
      connections.push_back(c);
      to_connection.insert({key, connections.size()-1});
    }
  }

  return cluster_t {
    .devices = devices,
    .connections = connections,
    .to_connection = to_connection
  };
}

double cluster_t::move(int src, int dst, uint64_t bytes) const {
  connection_t const& c = connections[to_connection.at({src,dst})];
  return (1.0 / c.bandwidth) * bytes;
}

double cluster_t::compute(int loc, uint64_t flops) const {
  device_t const& d = devices[loc];
  return (1.0 / d.compute) * flops;
}

costgraph_t costgraph_t::make(twolayergraph_t const& twolayer) {
  costgraph_t ret;

  map<tuple<int, int>, int> ridloc_to_costid;
  map<int, tuple<int, int>> jid_to_costidloc;

  for(auto const& [id, is_join]: twolayer.order) {
    if(is_join) {
      auto const& jid = id;
      auto const& join = twolayer.joins[jid];

      auto const& loc = twolayer.join_location(join);

      vector<int> deps;
      deps.reserve(join.deps.size());
      for(auto const& rid: join.deps) {
        deps.push_back(ridloc_to_costid.at({rid, loc}));
      }

      jid_to_costidloc[jid] = {
        ret.insert_compute(loc, join.flops, join.util, deps),
        loc
      };
    } else {
      auto const& rid = id;
      auto refi = twolayer.refinements[rid];

      // deduce where this refi will be used
      std::set<int> usage_locs;
      for(auto const& out_jid: refi.outs) {
        usage_locs.insert(twolayer.join_location(out_jid));
      }

      // insert moves as necc and fill out ridloc_to_costid
      for(auto const& dst: usage_locs) {
        vector<int> deps;
        for(auto const& [bytes, jids]: refi.units) {
          for(auto const& jid: jids) {
            auto const& [inn_costid, src] = jid_to_costidloc.at(jid);
            if(src == dst) {
              deps.push_back(inn_costid);
            } else {
              deps.push_back(ret.insert_move(src, dst, bytes, inn_costid));
            }
          }
        }
        int costid = ret.insert_barrier(deps);
        ridloc_to_costid.insert({ {rid, dst}, costid });
      }
    }
  }

  return ret;
}

costgraph_t costgraph_t::make_from_taskgraph(taskgraph_t const& taskgraph)
{
  costgraph_t costgraph;

  map<int, int> taskid_to_costid;

  for(auto const& taskid: taskgraph.get_order()) {
    auto const& node = taskgraph.nodes[taskid].op;

    if(node.is_input()) {
      auto const& input = node.get_input();
      int costid = costgraph.insert_compute(input.loc, 0, 1, {});
      taskid_to_costid.insert({taskid, costid});
    } else if(node.is_apply()) {
      auto const& apply = node.get_apply();

      vector<int> inns;
      inns.reserve(apply.inns.size());
      for(auto const& task_inn: apply.inns) {
        inns.push_back(taskid_to_costid.at(task_inn));
      }

      int costid = costgraph.insert_compute(
        apply.loc,
        product(apply.einsummable.join_shape),
        1,
        inns);
      taskid_to_costid.insert({taskid, costid});
    } else if(node.is_move()) {
      auto const& move = node.get_move();
      int costid = costgraph.insert_move(
        move.src, move.dst, move.size, taskid_to_costid.at(move.inn));
      taskid_to_costid.insert({taskid, costid});
    } else if(node.is_partialize()) {
      auto touches = node.get_touches();

      vector<int> inns;
      for(auto const& ts: touches) {
        for(auto const& [task_inn, touch]: ts) {
          inns.push_back(taskid_to_costid.at(task_inn));
        }
      }

      int costid = costgraph.insert_barrier(inns);
      taskid_to_costid.insert({taskid, costid});
    } else {
      throw std::runtime_error("should not reach: make_from_taskgraph");
    }
  }

  return costgraph;
}

int costgraph_t::insert_compute(
  int loc, uint64_t flops, int util,
  vector<int> const& deps)
{
  return insert(
    compute_t {
      .loc = loc,
      .flops = flops,
      .util = util
    },
    deps);
}

int costgraph_t::insert_move(int src, int dst, uint64_t bytes, int id)
{
  return insert(
    move_t {
      .src = src,
      .dst = dst,
      .bytes = bytes
    },
    {id}
  );
}

int costgraph_t::insert_barrier(vector<int> const& deps)
{
  return insert(barrier_t(), deps);
}

int costgraph_t::insert(
  op_t op,
  vector<int> const& deps)
{
  int ret = nodes.size();

  nodes.push_back(node_t {
    .id = ret,
    .inns = std::set(deps.begin(), deps.end()),
    .outs = {},
    .op = op
  });

  for(auto const& inn: nodes.back().inns) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

template <typename T>
struct worker_map_t {
  worker_map_t(cluster_t const& c):
    cluster(c),
    info(c.devices.size() + c.connections.size())
  {}

  T& operator()(int loc) {
    return info[loc];
  }
  T& operator()(int src, int dst) {
    return info[cluster.devices.size() + cluster.to_connection.at({src,dst})];
  }
  T& operator()(costgraph_t::op_t const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)) {
      return this->operator()(std::get<costgraph_t::compute_t>(x));
    } else if(std::holds_alternative<costgraph_t::move_t>(x)) {
      return this->operator()(std::get<costgraph_t::move_t>(x));
    } else {
      throw std::runtime_error("invalid alternative in worker_map");
    }
  }
  T& operator()(costgraph_t::compute_t const& c) {
    return this->operator()(c.loc);
  }
  T& operator()(costgraph_t::move_t const& m) {
    return this->operator()(m.src, m.dst);
  }

  vector<T> const& get() const { return info; }
  vector<T>&       get()       { return info; }

private:
  cluster_t const& cluster;

  vector<T> info;
};

struct cost_state_t {
  cost_state_t(cluster_t const& c, costgraph_t const& cg)
    : cluster(c), costgraph(cg), time(0.0), pending(c), capacity(c)
  {
    // initialize capacity
    for(int loc = 0; loc != cluster.devices.size(); ++loc) {
      capacity(loc) = cluster.devices[loc].capacity;
    }
    for(auto const& [_, src, dst]: cluster.connections) {
      capacity(src, dst) = 1;
    }

    // initialize pending and num_remaining
    num_remaining.reserve(costgraph.nodes.size());
    for(int id = 0; id != costgraph.nodes.size(); ++id) {
      auto const& node = costgraph.nodes[id];
      auto const& [_, inns, outs, op] = node;

      if(node.is_move()) {
        if(!in_range(node.move_src(), 0, cluster.num_device())) {
          throw std::runtime_error("invalid src");
        }
        if(!in_range(node.move_dst(), 0, cluster.num_device())) {
          throw std::runtime_error("invalid dst");
        }
      } else if(node.is_compute()) {
        if(!in_range(node.compute_loc(), 0, cluster.num_device())) {
          throw std::runtime_error("invalid compute loc");
        }
      }

      num_remaining.push_back(inns.size());

      if(inns.size() == 0) {
        if(node.is_barrier()) {
          throw std::runtime_error("a barrier must have dependencies");
        } else {
          pending(op).insert(id);
        }
      }
    }
  }

  void decrement_nodes(set<int> outs) {
    // this function is recursive; if outs is empty,
    // this is the base case
    if(outs.size() == 0) {
      return;
    }

    set<int> barrier_outs;
    for(auto const& out_id: outs) {
      num_remaining[out_id] -= 1;
      if(num_remaining[out_id] == 0) {
        auto const& out_node = costgraph.nodes[out_id];
        if(out_node.is_barrier()) {
          barrier_outs.insert(out_node.outs.begin(), out_node.outs.end());
        } else {
          pending(out_node.op).insert(out_id);
        }
      }
    }

    // now tail recurse on the collected barrier outs
    decrement_nodes(std::move(barrier_outs));
  }

  // returns true if something was taken from in_progress
  bool step() {
    if(in_progress.size() == 0) {
      return false;
    }

    // take off of in_progress
    auto [_time, op_id] = in_progress.top();
    in_progress.pop();

    // increment time
    time = _time;

    // alias the node
    auto const& node = costgraph.nodes[op_id];

    // these resources are now freed
    capacity(node.op) += node.worker_utilization();

    // tell the out nodes there is one less dependency
    // and possibly add to pending
    decrement_nodes(node.outs);

    return true;
  }

  void fill_with_work(int worker_id) {
    auto& capacity_vec = capacity.get();
    auto& pending_vec = pending.get();

    int& capacity_here          = capacity_vec[worker_id];
    std::set<int>& pending_here = pending_vec[worker_id];

    vector<int> possible;
    possible.reserve(pending_here.size());
    for(int const& op_id: pending_here) {
      auto const& node = costgraph.nodes[op_id];
      if(node.worker_utilization() <= capacity_here) {
        possible.push_back(op_id);
      }
    }

    while(possible.size() > 0) {
      int op_id = vector_random_pop(possible);
      auto const& node = costgraph.nodes[op_id];

      // move from pending
      pending_here.erase(op_id);
      // utilize the workers
      capacity_here -= node.worker_utilization();
      // and start working
      double time_when_finished = time + compute_cost(node.op);
      in_progress.push({time_when_finished, op_id});

      auto it = std::copy_if(
        possible.begin(),
        possible.end(),
        possible.begin(),
        [&](int const& id) {
          return costgraph.nodes[id].worker_utilization() < capacity_here;
        });
      possible.resize(it - possible.begin());
    }
  }

  void fill_with_work() {
    int num_workers = capacity.get().size();
    for(int i = 0; i != num_workers; ++i) {
      fill_with_work(i);
    }
  }

  double compute_cost(costgraph_t::op_t const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)){
      auto const& compute = std::get<costgraph_t::compute_t>(x);
      return cluster.compute(compute.loc, compute.flops);
    } else if(std::holds_alternative<costgraph_t::move_t>(x)) {
      auto const& move = std::get<costgraph_t::move_t>(x);
      return cluster.move(move.src, move.dst, move.bytes);
    } else {
      throw std::runtime_error("invalid alternative in compute_cost");
    }
  }

  // returns true if pending and in progress is empty
  bool is_done() {
    if(in_progress.size() > 0) {
      return false;
    }
    for(auto const& pending_at: pending.get()) {
      if(pending_at.size() != 0) {
        return false;
      }
    }
    return true;
  }

  cluster_t const& cluster;
  costgraph_t const& costgraph;

  double time;

  // for each worker, store the ids of all ops that can be
  // started
  worker_map_t<set<int>> pending;

  // store when the id will be ready, and the id
  priority_queue_least<tuple<double, int>> in_progress;

  worker_map_t<int> capacity;

  vector<int> num_remaining; // for each id, the num remaining
};

double costgraph_t::operator()(cluster_t const& cluster) const {
  cost_state_t state(cluster, *this);

  do {
    state.fill_with_work();
  } while(state.step());

  if(!state.is_done()) {
    throw std::runtime_error("costgraph comput time: state should be done");
  }

  return state.time;
}

std::ostream& operator<<(std::ostream& out, costgraph_t::node_t const& node) {
  using compute_t = costgraph_t::compute_t;
  using move_t    = costgraph_t::move_t;
  using barrier_t = costgraph_t::barrier_t;

  out << "costgraph" << node.id << "|inns:" <<
    vector(node.inns.begin(), node.inns.end()) <<
    "|outs:" <<
    vector(node.outs.begin(), node.outs.end()) <<
    "|";
  if(std::holds_alternative<compute_t>(node.op)) {
    auto const& compute = std::get<compute_t>(node.op);
    out << "compute[" << compute.flops << "]@" << compute.loc;
  } else if(std::holds_alternative<move_t>(node.op)) {
    auto const& move = std::get<move_t>(node.op);
    out << "move[" << move.bytes << "]" << move.src << "->" << move.dst;
  } else if(std::holds_alternative<barrier_t>(node.op)) {
    out << "barrier";
  }

  return out;
}

