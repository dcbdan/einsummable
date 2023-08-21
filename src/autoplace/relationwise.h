#pragma once
#include "../base/setup.h"

#include "twolayer.h"

struct kernel_coster_t {
  static kernel_coster_t for_cpu_cluster(int nlocs);

  double compute(einsummable_t const& e) const;
  double move(uint64_t n_bytes, int src, int dst) const;
  double touch(uint64_t n_bytes) const;

  vector<vector<double>> bandwidths; // bytes per second

  double flops; // floating points per second

  double rw;

  double compute_start;
  double touch_start;
  double move_start;
};
// TODO: maybe put kernel_coster in it's own file

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

  void add(double cost);
  void pop(double cost);
  double cost() const;
  void clear();

private:
  double max_cost;
  int cnt;
  int const n_threads;
};

struct relationwise_stat_t {
  int num_no_touch;
  int num_can_no_touch;
  double total_join;
  double total_touch;
  double total_move;
  struct blocks_t {
    int num;
    int num_blocks;
    int num_locs;
  };
  blocks_t non_contraction;
  blocks_t contraction;

  void print_line(std::ostream& out) const;
  void print(std::ostream& out) const;
};

struct relationwise_t {
  struct ginfo_t {
    partition_t partition;
    vector<join_t> joins;
    vector<int> locations;

    optional<partition_t> refinement_partition;
    optional<vector<refinement_t>> refis;

    // The computation of a single node proceeds as follows:
    //   1. do the computation           <- compute    cost
    //   2. compute the refinement       <- touch src  cost
    //   3. broadcast the data           <- move       cost
    //   4. materialize the next inputs  <- touch dst  cost

    vector<threads_costs_t> join_cost;
    vector<threads_costs_t> touch_src_cost;
    vector<double> move_cost;
    vector<threads_costs_t> touch_dst_cost;

    bool has_refinement() const { return bool(refinement_partition); }
    double total_join_cost() const;
    double total_refi_cost() const;
    double total_cost() const { return total_join_cost() + total_refi_cost(); }
  };
  // Note: all partition and refinement partitions are with respect to
  //       real dtypes

  relationwise_t(
    int nlocs,
    int n_threads_per_loc,
    graph_t const& graph,
    kernel_coster_t const& kernel_coster,
    vector<placement_t> const& pls);

  // NOTE: These return the _approximate_ change in total cost.
  double operator()(jid_t jid, int loc);
  double operator()(int gid, partition_t const& new_partition);
  double operator()(int gid, placement_t const& new_placement);

  vector<placement_t> get_placements() const;

  relationwise_stat_t make_stat() const;

  placement_t get_placement_at(int gid) const;

  bool has_no_touch(int gid) const;

  optional<partition_t> notouch_partition(int gid) const;

  // NOTE: to get a better costing, call reset_cost first
  double total_cost() const;

  std::function<partition_t const&(int)> f_get_partition() const;
  std::function<partition_t const&(int)> f_get_refinement_partition() const;
  std::function<vector<refinement_t>&(int)> f_get_mutable_refis();

  void reset_cost();
  void reset_join_cost(int gid);
  void reset_refi_cost(int gid);

  void add_refi_cost_at(rid_t rid) { _change_refi_cost_at(rid, true ); }
  void sub_refi_cost_at(rid_t rid) { _change_refi_cost_at(rid, false); }
  void _change_refi_cost_at(rid_t rid, bool add);

  bool has_join_cost(int gid) const { return graph.nodes[gid].op.is_einsummable(); }

  int const nlocs;
  int const n_threads_per_loc;
  graph_t const& graph;
  kernel_coster_t const kernel_coster;
  vector<ginfo_t> ginfos;
};
