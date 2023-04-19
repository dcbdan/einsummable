#pragma once
#include "setup.h"

#include "graph.h"

struct cluster_t {
  struct device_t {
    uint64_t compute; // flop per second per capacity
    int capacity;     // the number of workers or streams
  };

  struct connection_t {
    uint64_t bandwidth; // bytes per second
    int src;
    int dst;

    tuple<int,int> key() const { return {src,dst}; }
  };

  static cluster_t make(vector<device_t> const& ds, vector<connection_t> const& cs);

  connection_t const& get_connection(int src, int dst) const {
    return connections[to_connection.at({src,dst})];
  }

  float move(int src, int dst, uint64_t bytes) const;

  float compute(int loc, uint64_t flops) const;

  vector<device_t> const devices;
  vector<connection_t> const connections;
  map<tuple<int, int>, int> const to_connection;
};

struct twolayergraph_t {
  static twolayergraph_t make(graph_t const& graph);

  struct gid_t {
    int id;
    int index;
  };
  using rid_t = int; // refinement ids
  using jid_t = int; // join ids

  struct agg_unit_t {
    uint64_t bytes;
    vector<jid_t> deps;
  };

  struct refinement_t {
    vector<agg_unit_t> units;
    vector<jid_t> outs;
  };

  struct join_t {
    uint64_t flops;
    gid_t gid;
    vector<rid_t> deps;
    vector<rid_t> outs;
  };

  int const& join_location(gid_t const& gid) const {
    auto const& [id, index] = gid;
    return graph.nodes[id].placement.locations.get()[index];
  }

  vector<join_t> joins;
  vector<refinement_t> refinements;

  graph_t const& graph;

private:
  twolayergraph_t(graph_t const& graph)
    : graph(graph)
  {}
};

struct costgraph_t {
  static costgraph_t make(twolayergraph_t const& twolayer);
  float operator()(cluster_t const& cluster) const;

  struct compute_t {
    int loc;
    uint64_t flops;
    int util; // worker utilization
  };

  struct move_t {
    int src;
    int dst;
    uint64_t bytes;
  };

  struct node_t {
    set<int> inns;
    set<int> outs;
    std::variant<compute_t, move_t> op;

    int worker_utilization() const {
      if(std::holds_alternative<compute_t>(op)) {
        return std::get<compute_t>(op).util;
      } else {
        return 1;
      }
    }
  };

  vector<node_t> nodes;

  int insert_compute(int loc, uint64_t flops, int util, vector<int> const& deps);
  int insert_move(int src, int dst, uint64_t bytes, vector<int> const& deps);
  int insert(std::variant<compute_t, move_t> op, vector<int> const& deps);
};

// This object takes a graph with a fixed partition
// and efficiently recosts it. It tracks the graph object
// by reference.
//
// Example usage:
//   graph_t graph = create graph with initial partition
//   coster_t coster(graph, cluster);
//   until stop:
//     modify graph's placements
//     current_cost = coster()
struct coster_t {
  coster_t(graph_t const& g, cluster_t const& cluster)
    : cluster(cluster), twolayer(twolayergraph_t::make(g))
  {}

  float operator()() const {
    return this->operator()(cluster);
  }

  float operator()(cluster_t const& cluster) const {
    auto costgraph = costgraph_t::make(twolayer);
    return costgraph(cluster);
  }

private:
  cluster_t const cluster;
  twolayergraph_t const twolayer;
};
