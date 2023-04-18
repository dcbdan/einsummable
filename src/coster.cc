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
  int loc, uint64_t flops, int capacity,
  vector<int> const& deps)
{
  return insert(
    compute_t {
      .loc = loc,
      .flops = flops,
      .capacity = capacity
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

//template <typename T>
//using priority_queue_least = std::priority_queue<T, vector<T>, std::greater<T>>;
// For priority_queue_least, the top most element is the smallest,
// which is the opposite behaviour of priority_queue which puts the
// largest element at the top.

float costgraph_t::operator()(cluster_t const& cluster) const {

  // TODO
  // maintain a priority queue of in progress operations
  // maintain a worker to set of pending ops
  // maintain available worker resources
  //
  // while in_progress is not empty:
  //   pop from in progress
  //   update the time
  //   remove resources from the workers
  //   add to pending
  //
  //   for each worker,
  //     randomly pick something within the capacity available
  //     and add to pending
  //     (if no capacity, don't do anything)
  //
  // return the time

  return 0.0;
}

