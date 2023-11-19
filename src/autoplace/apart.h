#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

enum class parts_space_t {
  contraction,  // contractions nodes at 1x, other nodes at <=1x
  all,          // all nodes at 1x
  all_range     // all nodes at 1x,2x or 4x
};

vector<partition_t> autopartition_for_bytes(
  graph_t const& graph,
  int n_compute,
  int max_branching = 2,
  parts_space_t search_space = parts_space_t::contraction);

uint64_t autopartition_for_bytes_cost(
  graph_t const& graph,
  vector<partition_t> const& partitions);

