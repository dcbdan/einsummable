#pragma once
#include "../src/base/setup.h"

#include "modules.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/relation.h"

struct builder_t {
  static builder_t
  make_first_token(
    model_args_t const& args,
    uint64_t seqlen);

  static builder_t
  make_next_token(builder_t const& prev, bool make_last = false);

  bool is_first() const { return start_pos == 0; }
  bool is_last()  const { return !bool(next_kv); }

  void print_info() const;

  model_args_t args;
  graph_t graph;
  uint64_t start_pos;

  map<string, int> weights;
  int freqs_cis;

  int embeddings;

  // if not first
  optional<vector<tuple<int, int>>> prev_kv;

  // if not last
  optional<vector<tuple<int, int>>> next_kv;

  // if first & seqlen > 1
  optional<int> mask;

  int scores;

  // if not first
  optional<vector<tuple<int,int>>> remap;

  dtype_t input_dtype(int gid) const;
  vector<uint64_t> input_shape(int gid) const;

private:
  static builder_t _make(
    model_args_t const& args,
    uint64_t start_pos,
    uint64_t seqlen,
    bool make_last);
};
