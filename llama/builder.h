#pragma once
#include "../src/base/setup.h"

#include "modules.h"
#include "../src/einsummable/taskgraph.h"

struct builder_t {
  static builder_t
  make_first_token(model_args_t const& args, uint64_t seqlen);

  static builder_t
  make_next_token(builder_t const& prev, bool make_last = false);

  bool is_first() const { return start_pos == 0; }
  bool is_last()  const { return !bool(next_kv); }

  void transform_from_prev(map<int, buffer_t>& data) const;

  void print_info() const;

  struct tinfo_t {
    dtype_t dtype;
    placement_t placement;
    vtensor_t<int> tids;
  };

  model_args_t args;
  graph_t graph;
  taskgraph_t taskgraph;
  uint64_t start_pos;

  map<string, tinfo_t> weights;
  tinfo_t freqs_cis;

  tinfo_t embeddings;

  // if not first
  optional<vector<tuple<tinfo_t, tinfo_t>>> prev_kv;

  // if not last
  optional<vector<tuple<tinfo_t, tinfo_t>>> next_kv;

  // if first & seqlen > 1
  optional<tinfo_t> mask;

  tinfo_t scores;

  // if not first
  optional<map<int, int>> prev_tid_to_input_tids;
private:
  static builder_t _make(
    model_args_t const& args,
    uint64_t start_pos,
    uint64_t seqlen,
    bool make_last);

  static void same_placement_convert(
    map<int, int>& prev_to_new,
    tinfo_t const& prev_info,
    tinfo_t const& new_info);
};

void repartition_into_map_single_loc(
  map<int, buffer_t>& tid_to_buffer,
  builder_t::tinfo_t const& tinfo,
  buffer_t inn_relation);

dbuffer_t unpartitioned_from_map_single_loc(
  map<int, buffer_t>& tid_to_buffer,
  builder_t::tinfo_t const& tinfo);
