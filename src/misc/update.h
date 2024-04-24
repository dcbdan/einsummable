#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

struct updater_desc_t {
  struct vanilla_t {};

  struct adamw_t {
    dtype_t min_precision;
  };

  struct momentum_t {};

  dtype_t dtype;

  bool is_vanilla()  const { return std::holds_alternative<vanilla_t>(t); }
  bool is_adamw()    const { return std::holds_alternative<adamw_t>(t); }
  bool is_momentum() const { return std::holds_alternative<momentum_t>(t); }

  adamw_t const& get_adamw() const { return std::get<adamw_t>(t); }

  std::variant<vanilla_t, adamw_t, momentum_t> t;
};

// Return all inputs added to the graph and
// how to initialize those inputs
vector<tuple<int, fill_t>>
update_weights(
  updater_desc_t const& desc,
  graph_t& graph,
  vector<tuple<int, int>>& old_news,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids);

void update_vars(
  updater_desc_t const& desc,
  int iter,
  map<string, scalar_t>& vars);

