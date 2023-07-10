#pragma once
#include "../base/setup.h"

#include "twolayer.h"

// Some assumptions:
// * compute cost is the same on all nodes
// * move cost is the same across all pairs of nodes

struct relationwise_t {
  struct ginfo_t {
    partition_t partition;
    vector<join_t> joins;
    vector<int> locations;

    optional<partition_t> refinement_partition;
    optional<vector<refinement_t>> refis;

    vector<int64_t> compute_cost;
    vector<int64_t> move_cost;

    bool has_refinement() const { return bool(refinement_partition); }
  };
  // Note: all partition and refinement partitions are with respect to
  //       real dtypes

  relationwise_t(
    int nlocs,
    graph_t const& graph,
    vector<placement_t> const& pls);

  tuple<int64_t, int64_t> operator()(jid_t jid, int loc);

  // TODO(maybe just create a placement with chosen locs and dispatch to new_placement)
  tuple<int64_t, int64_t> operator()(int gid, partition_t const& new_partition);

  tuple<int64_t, int64_t> operator()(int gid, placement_t const& new_placement);

  vector<placement_t> get_placements() const;

  vector<int64_t> move_cost_at(rid_t rid) const;

  tuple<int64_t, int64_t> total_cost() const;

  std::function<partition_t const&(int)> f_get_partition() const;
  std::function<partition_t const&(int)> f_get_refinement_partition() const;
  std::function<vector<refinement_t>&(int)> f_get_mutable_refis();

  void reset_compute_cost(int gid);
  void reset_move_cost(int gid);

  int const nlocs;
  graph_t const& graph;
  vector<ginfo_t> ginfos;
};
