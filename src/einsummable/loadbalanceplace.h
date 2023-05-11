#pragma once
#include "setup.h"

#include "twolayergraph.h"

// In graph order, for each node:
//   If this is an elementwise node with the
//   same partition-through-permutation as
//   the input, copy directly.
//
//   Otherwise, order placements in terms of minimum
//   number of bytes moved, and assign with
//   the corresponding locations until
//   a location has run out of its load-balanced
//   allotment. Then redo the ordering without
//   that location and repeat until all assignments
//   have been given.
//
// Note: This placement will work best when
//       all compute blocks on a node are about
//       the same size.
//
// Input nodes are either round robin placed 0,1,2...
// or randomly assigned (but still load balanced)
vector<tensor_t<int>>
load_balanced_placement(
  graph_t const& graph,
  int nlocs,
  bool random_input);

