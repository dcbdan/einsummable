#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

enum class parts_space_t {
  contraction,  // contractions nodes at 1x, other nodes at <=1x
  all,          // all nodes at 1x
  all_range     // all nodes at 1x,2x or 4x
};

// 1. Flip the edges of the graph so that output nodes come first and then
//    form a dag of trees
// 2. For every tree, in dag order, apply dynamic programming
//    across the space of possible partitions. If the tree has a branching
//    more than max_branching, split it into multiple trees
//
// The partition space:
// * Given a space of rank r, all combinations of (2^i1, ..., 2^ir) where
//   2^(i1 + ... + i4) op<??> n_compute
//   When parts_space is contraction,
//     num blocks on contraction nodes must be n_compute, other nodes can
//     be less than n_compute.
//   When parts_space is all, all partitions must have n_compute blocks
//   When parts_space is all_ragne, all partitions must be either >= 1x and <= 4x
//     n_compute
// * Must be the case that n_compute is a power of 2.
//
// The cost:
// * Einsummable ops: The cost to move every input to a new location
// * Repartitions: The cost of all touches; the cost of a touch is the
//     cost to move the input and output to a new location
// * Non einsummable and non repartitions: no cost
vector<partition_t> apart01(
  graph_t const& graph,
  int n_compute,
  int max_branching = 2,
  parts_space_t search_space = parts_space_t::contraction);

uint64_t apart01_cost(
  graph_t const& graph,
  vector<partition_t> const& partitions);

