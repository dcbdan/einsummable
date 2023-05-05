#pragma once
#include "setup.h"

#include "taskgraph.h"

struct mem_t {
  int loc;
  uint64_t offset;
  uint64_t size;
};

// A memgraph is a taskgraph that has dependencies spanning
// memory constraints. In addition, for every compute node,
// there is a cache node. Cache nodes may not do computation
// and have infinite space.
struct memgraph_t {
  memgraph_t(
    int num_compute_locs,
    int num_cache_locs,
    vector<int> const& cache_locs,
    vector<uint64_t> const& memory_per_compute_loc);

  // Besides the memgraph, return for each save node
  // in the input taskgraph where the tensor is in memory
  // after execution.
  static
  tuple<
    map<int, mem_t>,
    memgraph_t >
  make(
    taskgraph_t const& graph,
    vector<uint64_t> const& memory_per_compute_loc,
    vector<int> const& cache_locs_for_each_compute_loc
  );

  struct input_t {
    int loc;
    uint64_t offset;
    uint64_t size;
  };

  struct apply_t {
    int loc;
    uint64_t offset;
    einsummable_t einsummable;
    vector<tuple<int, uint64_t>> inns; // for each input the inn id and the offset
  };

  // TODO: what does nvidia scheduling primitives
  //       allow you to do to make partial expressible
  struct partial_t {
    int inn; // the start partial
    int loc;
    touch_t touch;
    uint64_t inn_offset;
    uint64_t out_offset;
    int key; // ??????????!!!!!!!!!!
             // There is a dependency
  };

  struct move_t {
    tuple<int, uint64_t> src; // src loc, offset
    tuple<int, uint64_t> dst; // dst loc, offset
    uint64_t size;
  };

  struct evict_t {
    int src;         // compute loc
    int inn;         // the input id
    uint64_t offset; // input offset
    uint64_t size;   // input size
  };

  struct load_t {
    int dst; // compute loc
    int inn; // the input evict
    uint64_t offset; // output offset
    uint64_t size;   // output size
  }

  struct del_t {
    int inn;
    uint64_t offset;
    uint64_t size;
  };

  struct node_t {
    variant<input_t, apply_t, partial_t, move_t, evict_t, load_t, del_t>;
    set<int> memdeps; // all dependencies required for memory to work but
                      // not for the actual computation
    set<int> outs;    // where this node leads to a dependency
  };

  // compute locs are 0,...num_locs-1
  // cache locs are num_compute_locs, ... num_locs+num_cache_locs-1
  int const num_compute_locs;
  int const num_cache_locs;

  // Example: Four gpu node, with ram as the cache, then
  //          cache_locs = {4,4,4,4} and compute locs are 0,1,2,3.
  vector<int> const cache_locs;
  vector<uint64_t> const mem_sizes;

private:
  // int insert TODO
};
