#pragma once
#include "../base/setup.h"

#include "apart.h"
#include "alocate.h"

struct autoplace_config_t {
  int n_compute() const { return _n_locs * _n_compute_per_loc; }
  int n_locs() const { return _n_locs; }
  uint64_t flops_per_byte_moved() const { return _flops_per_byte_moved; }
  int max_branching() const { return _max_branching.value(); }
  uint64_t discount_input_factor() const { return _discount_input_factor.value(); }
  parts_space_t search_space() const { return _search_space.value(); }

  autoplace_config_t(int __n_locs, int __n_compute_per_loc, uint64_t __fps_per_byte)
    : _n_locs(__n_locs), _n_compute_per_loc(__n_compute_per_loc),
      _flops_per_byte_moved(__fps_per_byte)
  {}

  static autoplace_config_t make_default01(
    int n_locs, 
    int n_compute_per_loc,
    uint64_t discount_input_factor_ = 1)
  {
    autoplace_config_t ret(n_locs, n_compute_per_loc, 1000);
    ret._max_branching = 2;
    ret._discount_input_factor = discount_input_factor_;
    // all range seem to work well with GPU llama setting
    ret._search_space = parts_space_t::all_range;
    return ret;
  }

  static autoplace_config_t make_default02(int n_locs, int n_compute_per_loc) {
    autoplace_config_t ret(n_locs, n_compute_per_loc, 1000);
    return ret;
  }

private:
  int _n_locs;
  int _n_compute_per_loc;
  uint64_t _flops_per_byte_moved;
  optional<int>           _max_branching;
  optional<uint64_t>      _discount_input_factor;
  optional<parts_space_t> _search_space;
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

struct equal_holder_t {
  equal_holder_t() {}

  equal_holder_t(vector<tuple<int, int>> const& eqs);

  bool has(int i) const;

  set<int> const& operator[](int i) const;

  void insert(int i, int j);

private:
  vector<set<int>> sets;
  map<int, int> lookup;

  friend std::ostream& operator<<(std::ostream& out, equal_holder_t const& e);
};


