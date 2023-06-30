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
  optional<int> max_blocking = std::nullopt);

// For each node in order:
//   get the default partition from the inputs and call
//   make_correctly_sized_partition
vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t min_sizing,
  int max_blocking,
  equal_items_t<int> equal_constraints = {});

