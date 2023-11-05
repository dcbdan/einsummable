#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

vector<placement_t> autolocate(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs);
