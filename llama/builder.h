#pragma once
#include "../src/base/setup.h"

#include "modules.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/relation.h"

struct builder_t {
  static builder_t
  make_first_token(
    model_args_t const& args,
    uint64_t seqlen,
    std::function<vector<placement_t>(graph_t const&)> build_pls);

  static builder_t
  make_next_token(builder_t const& prev, bool make_last = false);

  bool is_first() const { return start_pos == 0; }
  bool is_last()  const { return !bool(next_kv); }

  void print_info() const;

  model_args_t args;
  graph_t graph;
  taskgraph_t taskgraph;
  uint64_t start_pos;

  map<string, relation_t> weights;
  relation_t freqs_cis;

  relation_t embeddings;

  // if not first
  optional<vector<tuple<relation_t, relation_t>>> prev_kv;

  // if not last
  optional<vector<tuple<relation_t, relation_t>>> next_kv;

  // if first & seqlen > 1
  optional<relation_t> mask;

  relation_t scores;

  // if not first
  optional<remap_relations_t> remap;

  std::function<vector<placement_t>(graph_t const&)> build_placements;

private:
  static builder_t _make(
    model_args_t const& args,
    uint64_t start_pos,
    uint64_t seqlen,
    std::function<vector<placement_t>(graph_t const&)> build_pls,
    bool make_last);
};

