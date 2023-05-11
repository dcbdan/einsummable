#pragma once
#include "setup.h"

#include "graph.h"

struct twolayergraph_t {
  // Return
  //   1. The mapping from graph id to twolayer join ids
  //   2. A suggestion for which join ids should have the same placement
  //   3. The two layer graph constructed from the graph and it's partitions
  static
  tuple<
    vector<tensor_t<int>>,
    equal_items_t<int>,
    twolayergraph_t>
  make(graph_t const& graph);

  using rid_t = int; // refinement ids
  using jid_t = int; // join ids

  // An agg unit is something that will get summed.
  // So if Y = X1 + X2 + X3 + X4 at locations
  //       0    0   1    1    2
  // then X1 is not moved,
  //      X2 and X3 are summed at location 1 and moved
  //      X4 is moved.
  // An agg unit depends on a set of join ids (X1,X2,X3,X4)
  // but neccessarily on all bytes of the join ids.
  // The bytes variable how much of each input it takes.
  struct agg_unit_t {
    uint64_t bytes;
    vector<jid_t> deps;
  };

  // A refinement is called such because when constructing from
  // a graph object, this is one block of the refinement partition
  // of all usages of some graph node.
  //
  // Consider Y = UnaryElementwiseOp(X) where X is partitioned into 2x2 blocks
  // and Y is a formation and the only usage, partitioned into 3x3 blocks.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   just Unary(X[0,0]) and the size of bytes is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit has one of the
  //   Unary(X[i,j]) blocks for i,j in [0,1]x[0,1]. The size of each agg unit block
  //   is roughly 1/4 the size of the Y[1,1] block.
  // Since this is an elementwise op, each agg unit is just a copy and does not
  //   actual summation.
  //
  // Consider instead Y = (ijk->ij, X) where X is partition into 2x2x3 blocks
  // and Y is partitioned again into 3x3 blocks. Everything holds the same
  // as before except each agg unit has 3 inputs.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   (ijk->ij, X[0,0,k]) for k=0,1,2 and the size of bytes is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit represents some i,j
  //   and that agg unit has blocks (ijk->ij, X[i,j,k]) for k=0,1,2.
  //   The size of each agg unit block is roughly 1/4 the size of the Y[1,1] block.
  struct refinement_t {
    vector<agg_unit_t> units;
    set<jid_t> outs;
  };

  // A join cannot complete until each agg in each dependent refinement
  // is completed.
  struct join_t {
    uint64_t flops;
    vector<rid_t> deps;
    set<rid_t> outs;
  };

  vector<join_t> joins;
  vector<refinement_t> refinements;

  struct twolayerid_t {
    int id;       // either join or refinemet id
    bool is_join; // depending on is_join
  };
  vector<twolayerid_t> order;

private:
  jid_t insert_join(uint64_t flops, vector<rid_t> const& deps);
  rid_t insert_empty_refinement();
  void add_agg_unit(rid_t rid, uint64_t bytes, vector<jid_t> deps);
};


