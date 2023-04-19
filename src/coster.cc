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

float cluster_t::move(int src, int dst, uint64_t bytes) const {
  connection_t const& c = connections[to_connection.at({src,dst})];
  return (1.0 / c.bandwidth) * bytes;
}

float cluster_t::compute(int loc, uint64_t flops) const {
  device_t const& d = devices[loc];
  return (1.0 / d.compute) * flops;
}

twolayergraph_t twolayergraph_t::make(graph_t const& graph) {
  twolayergraph_t ret(graph);

  for(auto const& graph_id: graph.get_order()) {
    // TODO
    // for every out block,
    //   set up a join_t object

    // collect the usage objects and the refinemet multiple partition

    // set up the refinement nodes
  }

  return ret;
}

costgraph_t costgraph_t::make(twolayergraph_t const& twolayer) {

  // TODO
  // maintain a map from (rid_t, loc) to costgraph id (as a tensor)
  // maintain a map from jid_t to costgraph id        (as a vector)
  // for every twolayergraph id in order,
  //   if it is a refinement:
  //     collect the usage locations
  //     do the moves
  //     save into the map
  //   if it is a join:
  //     get the location
  //     get the costgraph dependencies
  //     add a costgraph comptue ndoe
  //     save into map

  return costgraph_t();
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

int costgraph_t::insert_move(
  int src, int dst, uint64_t bytes,
  vector<int> const& deps)
{
  return insert(
    move_t {
      .src = src,
      .dst = dst,
      .bytes = bytes
    },
    deps);
}

int costgraph_t::insert(
  std::variant<compute_t, move_t> op,
  vector<int> const& deps)
{
  int ret = nodes.size();

  nodes.push_back(node_t {
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
  T& operator()(
    std::variant<
      costgraph_t::compute_t,
      costgraph_t::move_t> const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)) {
      return this->operator()(std::get<costgraph_t::compute_t>(x));
    } else {
      return this->operator()(std::get<costgraph_t::move_t>(x));
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
      auto const& [inns, outs, op] = costgraph.nodes[id];
      num_remaining.push_back(inns.size());

      if(inns.size() == 0) {
        pending(op).insert(id);
      }
    }
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
    // and possibly add to ready
    for(auto const& out_id: node.outs) {
      num_remaining[out_id] -= 1;
      if(num_remaining[out_id] == 0) {
        auto const& out_node = costgraph.nodes[out_id];
        pending(out_node.op).insert(out_id);
      }
    }

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
      float time_when_finished = time + compute_cost(node.op);
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

  float compute_cost(
    std::variant<
      costgraph_t::compute_t,
      costgraph_t::move_t> const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)){
      auto const& compute = std::get<costgraph_t::compute_t>(x);
      return cluster.compute(compute.loc, compute.flops);
    } else {
      auto const& move = std::get<costgraph_t::move_t>(x);
      return cluster.move(move.src, move.dst, move.bytes);
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

  float time;

  // for each worker, store the ids of all ops that can be
  // started
  worker_map_t<set<int>> pending;

  // store when the id will be ready, and the id
  priority_queue_least<tuple<float, int>> in_progress;

  worker_map_t<int> capacity;

  vector<int> num_remaining; // for each id, the num remaining
};

float costgraph_t::operator()(cluster_t const& cluster) const {
  cost_state_t state(cluster, *this);

  do {
    state.fill_with_work();
  } while(state.step());

  if(!state.is_done()) {
    throw std::runtime_error("costgraph comput time: state should be done");
  }

  return state.time;
}

