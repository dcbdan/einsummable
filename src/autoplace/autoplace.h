#pragma once
#include "../base/setup.h"

#include "apart.h"
#include "alocate.h"

struct autoplace_config_t {
  int n_compute;
  int max_branching;
  uint64_t discount_input_factor;
  parts_space_t search_space;
  uint64_t flops_per_byte_moved;

  static autoplace_config_t make_default(int n_compute) {
    return autoplace_config_t {
      .n_compute = n_compute,
      .max_branching = 2,
      .discount_input_factor = 1,
      .search_space = parts_space_t::contraction,
      .flops_per_byte_moved = 1000
    };
  }
};

vector<placement_t> autoplace01(
  graph_t const& graph,
  autoplace_config_t const& config);

// fixed_pls: these nodes have this placement
// equal_pls: these nodes have the same placement
vector<placement_t> autoplace02(
  graph_t const& graph,
  autoplace_config_t const& config,
  map<int, placement_t> const& fixed_pls,
  vector<tuple<int,int>> const& equal_pls);
