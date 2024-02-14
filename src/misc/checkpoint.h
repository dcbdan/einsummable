#pragma once
#include "../base/setup.h"

#include "../einsummable/graph.h"
#include "../einsummable/taskgraph.h"
#include "../einsummable/relation.h"

struct graph_id_manager_t {
  struct context_t {
    context_t(graph_id_manager_t* s, int w)
      : self(s), which(w)
    {}

    optional<int> get(int fid) const {
      return self->get(which, fid);
    }

    void insert(int fid, int nid) {
      return self->insert(which, fid, nid);
    }
  private:
    graph_id_manager_t* self;
    int which;
  };

  context_t make_context(int which) {
    return context_t(this, which);
  }

  void insert(int which, int fid, int sid);

  optional<int> get(int which, int fid) const {
    return get_sid(which, fid);
  }

  optional<int> get_sid(int which, int fid) const;

  optional<int> get_fid(int which, int sid) const;

  optional<int> get_sid_from_sid(int which_src, int src_sid, int which_dst) const;

  void print() const;

private:
  // which -> (fid -> sid, sid -> fid)
  map<int, tuple<map<int, int>, map<int, int>>> data;
};

struct checkpoint_graphs_t {
  checkpoint_graphs_t(
    graph_t const& full_graph,
    vector<int> const& checkpoints,
    set<int> const& forward_ids);

  // There are n graphs and n+1 remaps.
  // So to execute full_graph, you do
  //   remap[0], graphs[0], ..., remap[n-1], graphs[n-1], remap[n]

  graph_t const& full_graph;
  graph_id_manager_t manager;
  vector<graph_t> graphs;
  vector<vector<tuple<int, int>>> remaps;
};

struct checkpoint_taskgraphs_t {
  checkpoint_taskgraphs_t(
    checkpoint_graphs_t const& gs,
    vector<placement_t> const& pls);

  checkpoint_graphs_t const& gs;

  struct info_t {
    map<int, relation_t> init_rel;
    taskgraph_t taskgraph;
    map<int, relation_t> save_rel;
  };
  vector<info_t> infos;

  // To execute:
  //   0  : gs.remaps[0],   info[0]:   init_rel, taskgraph, save_rel
  //        ...
  //   n-1: gs.remaps[n-1], info[n-1]: init_rel, taskgraph, save_rel
  //   n  : gs.remaps[n]
};
