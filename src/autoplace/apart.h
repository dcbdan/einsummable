#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

vector<partition_t> autopartition_for_bytes(
  graph_t const& graph,
  int n_compute);

uint64_t autopartition_for_bytes_cost(
  graph_t const& graph,
  vector<partition_t> const& partitions);
