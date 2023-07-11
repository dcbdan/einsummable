#pragma once
#include "../base/setup.h"

#include "twolayer.h"

// Inside relationwise_t, each compute node is given some number
// of threads that do computation. Whenever a compute location is
// changed, work is removed from the previous location
// and added to the new location.
//
// It is a challenge to implement this mechanism accurately and
// efficiently for the purposes of relationwise_t.
//
// One implementation would be to assign every block compute both the
// location and the thread at that location. However, storing the thread for
// every block and then reassigning it is at lot of work to do in the
// simulation--you would also have to choose which thread at every iteration,
// so you'd have to keep track of which thread having the least work.
// All that is too costly.
//
// Instead, threads_costs_t provides an interface to (1) add work (2)
// pop work and (3) get the total cost for executing all threads.
//
// Internally, it just stores the maximum cost encountered and the total
// number of things to compute so that the total cost is
//   ceiling(cnt / n_threads) * max_cost
//
// This approximation works well when the costs are all the same. If the
// costs were to very a lot (say costs = 1,2,3,4,5,6,7,8) and (n_threads = 4),
// the best implementation for this would deduce (total cost = 2 + 4 + 6 + 8).
struct threads_costs_t {
  threads_costs_t(int n_threads);

  void add(int64_t cost);
  void pop(int64_t cost);
  int64_t cost() const;

private:
  int64_t max_cost;
  int cnt;
  int const n_threads;
};

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

    vector<threads_costs_t> compute_cost;
    vector<int64_t> move_cost;

    bool has_refinement() const { return bool(refinement_partition); }
  };
  // Note: all partition and refinement partitions are with respect to
  //       real dtypes

  relationwise_t(
    int nlocs,
    int n_threads_per_loc,
    graph_t const& graph,
    vector<placement_t> const& pls);

  tuple<int64_t, int64_t> operator()(jid_t jid, int loc);

  tuple<int64_t, int64_t> operator()(int gid, partition_t const& new_partition);

  tuple<int64_t, int64_t> operator()(int gid, placement_t const& new_placement);

  vector<placement_t> get_placements() const;

  placement_t get_placement_at(int gid) const;

  vector<int64_t> move_cost_at(rid_t rid) const;

  tuple<int64_t, int64_t> total_cost() const;

  std::function<partition_t const&(int)> f_get_partition() const;
  std::function<partition_t const&(int)> f_get_refinement_partition() const;
  std::function<vector<refinement_t>&(int)> f_get_mutable_refis();

  void reset_compute_cost(int gid);
  void reset_move_cost(int gid);

  int const nlocs;
  int const n_threads_per_loc;
  graph_t const& graph;
  vector<ginfo_t> ginfos;
};
