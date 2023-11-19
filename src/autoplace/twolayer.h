#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"

struct jid_t {
  int gid;
  int bid;
};

struct rid_t {
  int gid;
  int bid;
};

struct agg_unit_t {
  uint64_t size;
  vector<int> deps; // these are the join bids of the graph node that
                    // this agg unit belongs to
  // TODO: why is this not a set?
};

struct refinement_t {
  vector<agg_unit_t> units;
  set<jid_t> outs;
};

struct join_t {
  optional<einsummable_t> einsummable;
  set<rid_t> deps;
  set<int> outs; // get all refinement bids that depend on this join
};

// Note:
// * join partitions are with resepect to the graph node dtype
// * refin partitions are with resepect to real

// Construct the join objects for this partition,
// without setting up deps or outs
vector<join_t> twolayer_construct_joins(
  graph_t const& graph,
  int gid,
  partition_t const& partition);

void twolayer_insert_join_deps(
  graph_t const& graph,
  int gid,
  vector<join_t>& join_infos,
  partition_t const& join_partition,
  std::function<partition_t const&(int)> get_refinement_partition);

void twolayer_insert_refi_outs_from_join_deps(
  graph_t const& graph,
  int join_gid,
  vector<join_t> const& join_infos,
  std::function<vector<refinement_t>&(int)> get_refis);

// Construct the refinement partition, which is how this
// join object gets used. Will call get_partition for each
// outgoing gid
// Note: refinement partitions are always with respect to real dtypes
partition_t twolayer_construct_refinement_partition(
  graph_t const& graph,
  int gid,
  std::function<partition_t const&(int)> get_partition);

// Construct the refinement partition and connect it with
// the join of the same gid
vector<refinement_t> twolayer_construct_refis_and_connect_joins(
  graph_t const& graph,
  int gid,
  vector<join_t>& joins,
  partition_t const& join_partition,
  partition_t const& refinement_partition);

void twolayer_connect_join_to_refi(
  graph_t const& graph,
  int gid,
  vector<join_t>& joins,
  partition_t const& join_partition,
  vector<refinement_t>& refis,
  partition_t const& refinement_partition);

void twolayer_erase_refi_deps(vector<refinement_t>& refis);
void twolayer_erase_join_outs(vector<join_t>&       joins);
void twolayer_erase_join_deps(vector<join_t>&       joins);

bool operator==(jid_t const& lhs, jid_t const& rhs);
bool operator!=(jid_t const& lhs, jid_t const& rhs);
bool operator< (jid_t const& lhs, jid_t const& rhs);
bool operator==(rid_t const& lhs, rid_t const& rhs);
bool operator!=(rid_t const& lhs, rid_t const& rhs);
bool operator< (rid_t const& lhs, rid_t const& rhs);

std::ostream& operator<<(std::ostream&, jid_t const&);
std::ostream& operator<<(std::ostream&, rid_t const&);

std::ostream& operator<<(std::ostream&, join_t const&);
std::ostream& operator<<(std::ostream&, refinement_t const&);

// AGG UNIT--
// An agg unit is something that will get summed.
// So if Y = X1 + X2 + X3 + X4 at locations
//       0    0   1    1    2
// then X1 is not moved,
//      X2 and X3 are summed at location 1 and moved
//      X4 is moved.
// An agg unit depends on a set of join ids (X1,X2,X3,X4)
// but not neccessarily on all elements of the join ids.
// The size variable how much of each input it takes.

// REFINEMENT--
// A refinement is called such because when constructing from
// a graph object, this is one block of the refinement partition
// of all usages of some graph node.
//
// Consider Y = UnaryElementwiseOp(X) where X is partitioned into 2x2 blocks
// and Y is a formation and the only usage, partitioned into 3x3 blocks.
// The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
//   just Unary(X[0,0]) and the size is the size of Y[0,0] block.
// The refinement of Y[1,1] has four agg units. Each agg unit has one of the
//   Unary(X[i,j]) blocks for i,j in [0,1]x[0,1]. The size of each agg unit block
//   is roughly 1/4 the size of the Y[1,1] block.
// Since this is an elementwise op, each agg unit is just a copy and does no
//   actual summation.
//
// Consider instead Y = (ijk->ij, X) where X is partition into 2x2x3 blocks
// and Y is partitioned again into 3x3 blocks. Everything holds the same
// as before except each agg unit has 3 inputs.
// The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
//   (ijk->ij, X[0,0,k]) for k=0,1,2 and the size is the size of Y[0,0] block.
// The refinement of Y[1,1] has four agg units. Each agg unit represents some i,j
//   and that agg unit has blocks (ijk->ij, X[i,j,k]) for k=0,1,2.
//   The size of each agg unit block is roughly 1/4 the size of the Y[1,1] block.
