#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

vector<partition_t> autopartition(
  graph_t const& graph,
  int nloc,
  int nmax,
  equal_items_t<int> equal_constraints = {});

struct autopartition_state_t {
  autopartition_state_t(
    graph_t const& g,
    int nloc,
    int nmax,
    equal_items_t<int> const& equal_parts,
    std::function<partition_t(partition_t const&)> make_finer,
    std::function<partition_t(partition_t const&)> make_coarser);

  // Set nodes from the input partitions and
  // return whether or not this node was set.
  // Does not set if input node, mmlike or
  // not all inputs are available.
  bool set_from_inputs_and_recurse(int id);

  // Set the partition of any non-mmlike node.
  // Decide the partition based on (1) the
  // usage partitions and (2) the input partitions,
  // if available.
  //
  // There is one caveat: formation
  // nodes that do not have an ouput are not set.
  // Consider
  //   node 0: A/Input
  //   node 1: C = A + B
  //   node 2: Form(C)
  // Where A does not have a partition.
  // Then, with output-formation nodes considered,
  // 1. set node 2 to singleton
  // 2. set C to intersection of node 2 and node B
  //    which is just node B
  // 3. Set node 0 to the partition of node 1
  //
  // How it'd work now, not setting output-formation
  // nodes:
  // 1. set node 1 to partition of node B
  // 2. set node 0 to partition of C
  // Now node 2 does not have a partition, so go
  // back and set it to the partition of C.
  void set_from_outputs_and_recurse(int id);

  void set_mmlike(int id);

  void set_partition(int id, partition_t const& p);

  bool is_mmlike(int id) const;
  bool is_output_formation(int id) const;

  // If the partition has blocks finer
  // than min_sizing, then either return
  // choice if not none, or make it coarser
  // by iteratively making dimensions coarser
  partition_t construct_minsized_partition(
    partition_t const& maybe_too_fine_partition,
    optional<partition_t> const& choice) const;

  bool is_too_fine(partition_t const& p) const;

  graph_t const& graph;
  int nloc;
  int nmax;

  set<int> remaining;
  vector<optional<partition_t>> ret;

  equal_items_t<int> equals;

  std::function<partition_t(partition_t const&)> make_finer;
  std::function<partition_t(partition_t const&)> make_coarser;
};


