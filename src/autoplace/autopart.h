#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

// If the smallest block is smaller than min sizing:
//   Make coarser until it can't be made coarser or that is
//   not the case.
//   Then return.
// if max_blocking:
//   Make the block finer until and if it becomes too fine,
//     return the last valid block.
//     A block is too fine when the smallest block is smaller
//     than min sizing or has more blocks than max number of blockings
partition_t make_correctly_sized_partition(
  partition_t const& init,
  uint64_t min_sizing,
  optional<int> max_blocking = std::nullopt,
  bool via_doubling = false);

// For each node in order:
//   get the default partition from the inputs and call
//   make_correctly_sized_partition
vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t min_sizing,
  int max_blocking,
  equal_items_t<int> equal_constraints = {},
  bool via_doubling = false);

optional<partition_t> make_finer_via_doubling(partition_t const& p);
optional<partition_t> make_coarser_via_halving(partition_t const& p);

optional<partition_t> make_finer_via_increment(partition_t const& p);
optional<partition_t> make_coarser_via_decrement(partition_t const& p);

struct autopart_t {
  autopart_t(graph_t const& graph, int nworkers);

  struct ginfo_t {
    vector<int> partition;
    int64_t cost;
    vector<int64_t> inn_costs;
  };

  int64_t operator()(int gid, vector<int> const& new_partition);

  int64_t get_current_cost() const;

  vector<placement_t> get_placements() const;

  void print_graphviz(std::ostream& out) const;

  vector<ginfo_t> ginfos;
  graph_t const& graph;
  int const nworkers;

private:
  int64_t update_inn_cost(int gid, int which_inn);
};

struct autopart_mcmc_t {
  autopart_mcmc_t(
    graph_t const& graph,
    int nworkers,
    int max_blocks);

  bool step(double beta);

  vector<placement_t> const& get_best_placements() const {
    return best_placements;
  }

  int64_t const& get_best_cost() const {
    return best_cost;
  }

  int64_t const& get_current_cost() const {
    return current_cost;
  }

  autopart_t autopart;

private:
  int max_blocks;

  int64_t current_cost;

  int64_t best_cost;
  vector<placement_t> best_placements;

private:
  //
};
