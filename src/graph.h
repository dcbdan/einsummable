#pragma once
#include "setup.h"

#include "placement.h"
#include "einsummable.h"

struct input_t {
  vector<uint64_t> shape;
};

struct output_t {
  vector<uint64_t> shape;
};

// TODO: make node a class proper
using node_t = std::variant<input_t, output_t, einsummable_t>;

struct graph_t {
  // Methods to construct a graph object
  // {{{
  int insert_input(
    placement_t placement);
  int insert_input(
    partition_t partition);
  int insert_input(
    vector<uint64_t> shape);

  int insert_einsummable(
    placement_t placement,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    partition_t partition,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    einsummable_t e,
    vector<int> inns);

  int insert_output(
    placement_t placement,
    int inn);
  int insert_output(
    partition_t partition,
    int inn);
  int insert_output(
    vector<uint64_t> shape,
    int inn);

  // for each input and einsummable ops O, make sure O
  // is used by another op. If not, add an output op that
  // uses O.
  void set_outputs();
  // }}}

private:
  struct info_t {
    node_t node;
    vector<int> inns;
    vector<int> outs;
    partition_t partition;
  };

  vector<info_t> infos;
};

