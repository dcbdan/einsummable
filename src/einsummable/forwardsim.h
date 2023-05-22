#pragma once
#include "setup.h"

#include "graph.h"
#include "cluster.h"

// Actions:
//   partition
//   enqueue worker
//   assign_jid_loc
//   pop_work
// Events:
//   assign partition (gid)
//   setup joins (gid)
//   assign location (jid)
//   completed move (rid,uid,dst)
//   completed agg unit (rid,uid,dst)
//   completed refinement (rid,dst)
//   completed join (jid)
struct forward_state_t {
  struct jid_t {
    int gid;
    int bid;
  };
  struct rid_t {
    int gid;
    int bid;
  };

  struct completed_t {
    completed_t(double b, double e, int src, int dst, int gid, int bid, int uid, uint64_t size)
      : start(b), finish(e), c(done_move_t{ src, dst, gid, bid, uid, size })
    {}
    completed_t(double b, double e, int loc, int gid, int bid, uint64_t flops)
      : start(b), finish(e), c(done_apply_t{ loc, gid, bid, flops })
    {}

    struct done_move_t {
      int src;
      int dst;
      int gid;
      int bid;
      int unit;
      uint64_t size;
    };
    struct done_apply_t {
      int loc;
      int gid;
      int bid;
      uint64_t flops;
    };

    bool did_move()  const { return std::holds_alternative<done_move_t>(c);  }
    bool did_apply() const { return std::holds_alternative<done_apply_t>(c); }

    done_move_t  const& get_move_info()  const { return std::get<done_move_t>(c);  }
    done_apply_t const& get_apply_info() const { return std::get<done_apply_t>(c); }

    double start;
    double finish;

  private:
    std::variant<done_move_t, done_apply_t> c;
  };

  forward_state_t(graph_t const& g);

  // get all gids that can currently be given a partition
  set<int> const& can_assign_partition() const;

  // get all bids at graph node gid that can currently be given a location
  vector<int> can_assign_location(int gid);
  // TODO

  void assign_partition(int gid, partition_t const& part);

  void assign_location(jid_t jid, int loc);
  // TODO

  void enqueue_apply_worker(int loc, int which);
  // TODO

  void enqueue_move_worker(int src, int dst, int which);
  // TODO

  completed_t pop_work();
  // TODO

private:
  // ec = Event Completed

  // Once a partition is assigned, it may be the case that
  // a refinement partition of one of the inputs can be setup.
  // Once a partition is assigned, those locations can also
  // be assigned
  void ec_assign_partition(int gid);

  // Once a refinement is setup, it may the case that an output
  // graph node can have the the joins setup
  void ec_setup_refinement(int gid);

  // Setup these joins once the usage partition of gid can be
  // established
  void ec_setup_joins(int gid);
  // TODO

  // Once a location has been assigned, it may
  // be the case that an agg unit is now available
  void ec_assign_location(jid_t jid);
  // TODO

  // Once a move is completed, an agg unit at dst is that
  // move closer to being complete.
  // (This should be called even when the src location
  //  is dst and thus a physical move didn't actually happen)
  void ec_move(rid_t rid, int uid, int dst);
  // TODO

  // Once an agg unit at some dst has completed,
  // the corresponding refinement has one less dependent
  void ec_agg_unit(rid_t rid, int uid, int dst);
  // TODO

  // Once a refinment at dst has completed, the outgoing joins at
  // dst have one less dependency to wait for
  void ec_refinement(rid_t rid, int dst);
  // TODO

  // Once a join completes, the outgoing agg units at
  // the computed location have one less dependent
  void ec_join(int jid);
  // TODO

private:

  // An agg unit is something that will get summed.
  // So if Y = X1 + X2 + X3 + X4 at locations
  //       0    0   1    1    2
  // then X1 is not moved,
  //      X2 and X3 are summed at location 1 and moved
  //      X4 is moved.
  // An agg unit depends on a set of join ids (X1,X2,X3,X4)
  // but not neccessarily on all elements of the join ids.
  // The size variable how much of each input it takes.
  struct agg_unit_t {
    uint64_t size;
    vector<int> deps; // these are the bids of the graph node that
                      // this agg unit belongs to
  };

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
  struct refinement_t {
    vector<agg_unit_t> units;
    set<jid_t> outs;
  };

  struct join_t {
    uint64_t flops;
    vector<rid_t> deps;
    set<int> outs; // get all refinement bids that depend on this join
  };

private:
  graph_t const& graph;

  struct graph_node_info_t {
    graph_node_info_t();

    optional<partition_t> partition;
    optional<tensor_t<int>> locs;

    optional<tensor_t<join_t>> joins;

    optional<partition_t> refinement_partition;
    optional<tensor_t<refinement_t>> refis;
  };

  vector<graph_node_info_t> ginfos;

  set<int> can_partition;

private:
  bool can_setup_joins(int gid) const;
  void setup_joins(int gid);

  bool can_setup_refinement_partition(int gid) const;
  void setup_refinement_partition(int gid);

  bool can_setup_refis(int gid) const;
  void setup_refis(int gid);
};

bool operator==(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator< (forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator==(forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs);
bool operator< (forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs);

