#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

vector<partition_t> autopartition_for_repart(
  graph_t const& graph,
  int n_compute,
  double time_per_write,
  double time_per_elem);

vector<partition_t> autopartition_for_bytes(
  graph_t const& graph,
  int n_compute);

