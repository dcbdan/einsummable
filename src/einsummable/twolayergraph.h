#pragma once
#include "setup.h"

#include "graph.h"

struct twolayergraph_t {
  static twolayergraph_t make(graph_t const& graph);

  struct gid_t {
    int id;
    int index;
  };
  using rid_t = int; // refinement ids
  using jid_t = int; // join ids

  struct agg_unit_t {
    uint64_t bytes;
    vector<jid_t> deps;
  };

  struct refinement_t {
    vector<agg_unit_t> units;
    set<jid_t> outs;
  };

  struct join_t {
    uint64_t flops;
    int util;
    gid_t gid;
    vector<rid_t> deps;
    set<rid_t> outs;
  };

  int const& join_location(gid_t const& gid) const {
    auto const& [id, index] = gid;
    return graph.nodes[id].placement.locations.get()[index];
  }
  int const& join_location(join_t const& join) const {
    return join_location(join.gid);
  }
  int const& join_location(jid_t jid) const {
    return join_location(joins[jid]);
  }

  vector<join_t> joins;
  vector<refinement_t> refinements;

  struct twolayerid_t {
    int id;
    bool is_join;
  };
  vector<twolayerid_t> order;

  graph_t const& graph;

private:
  twolayergraph_t(graph_t const& graph)
    : graph(graph)
  {}

  jid_t insert_join(uint64_t flops, int util, gid_t gid, vector<rid_t> deps);
  rid_t insert_empty_refinement();
  void add_agg_unit(rid_t rid, uint64_t bytes, vector<jid_t> deps);
};

