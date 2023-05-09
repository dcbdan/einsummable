#pragma once
#include "setup.h"

#include "taskgraph.h"

struct memloc_t;

struct mem_t {
  uint64_t offset;
  uint64_t size;

  memloc_t as_memloc(int loc) const;
};

struct memloc_t {
  uint64_t offset;
  uint64_t size;
  int loc;

  mem_t as_mem() const;
};

std::ostream& operator<<(std::ostream&, mem_t const&);
std::ostream& operator<<(std::ostream&, memloc_t const&);

struct memgraph_t {
  memgraph_t(
    int num_compute_locs,
    int num_cache_locs,
    vector<int> const& cache_locs);

  // Create a memgraph without any memory-size constraints.
  // Return also mappings
  //   input taskgraph node ids -> memory
  //   save taskgraph node ids  -> memory.
  //
  // The algorithm is to
  //   (1) place an ordering on all task group ops,
  //   (2) walk throught the ordering one node at a time,
  //       allocating memory as necessary, constructing the
  //       op, and deleting memory as necessary
  // Note that the ordering is important because the
  // deletes will create a dependency between ops.
  static
  tuple<
    map<int, mem_t>, // input -> mem
    map<int, mem_t>, // save -> mem
    memgraph_t>
  make_without_evict(
    taskgraph_t const& graph,
    vector<int> const& which_cache);

  void print_graphviz(std::ostream& out) const;

  // Get the amount of memory used by each location
  vector<uint64_t> mem_sizes() const;

  // An ordering of 0,1,2... works if memgraph_t::insert
  // was used to construct *this
  vector<int> get_order() const {
    vector<int> ret(nodes.size());
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
  }

  // at time zero, this input is here with this memory
  struct input_t {
    int loc;
    uint64_t offset;
    uint64_t size;
  };

  // An apply needs these memories to do the computation
  // at hand. (for einsummable, output then inn memories)
  // (for touch, write memory then read memories)
  struct apply_t {
    int loc;
    vector<mem_t> mems;
    std::variant<einsummable_t, touch_t> op;
    int group;
  };
  // Consider an aggregation Y = X1 + X2 + X3 + X4 + X5
  // where the order that X1, ..., X5 comes available is
  // unknown and may be very different. We don't want to
  // constrain these opereations to happen in a particular
  // order and we don't want multiple operations to be
  // happening at the same time.
  //
  // To do this, touch(X1,Y), ..., touch(X5,Y) should
  // all be given the same group parameter so the execution
  // engine can tell that these ops require a lock before
  // proceeding.
  //
  // If group < 0, there is no grouping.
  //
  // Straight elementwise ops may also have the same output
  // and input memory.
  // For example: in relu(matmul(A,B)),
  //              the relu node may have the same input
  //              and output memory
  // Similarly for touch ops of the form
  //   A min= B
  //   A min= C
  //   ...
  //   Then A,B,C may have the same memory.

  struct move_t {
    tuple<int, uint64_t> src; // src loc, offset
    tuple<int, uint64_t> dst; // dst loc, offset
    uint64_t size;
  };

  // Note: every location has one cache, but a
  //       cache may have multiple locations

  // Move this memory off of location loc and into
  // the corresponding cache
  struct evict_t {
    int loc;
    int cache_id;
    uint64_t offset;
    uint64_t size;
  };

  // Load from cache this id into loc
  // with this offset and size
  struct load_t {
    int cache_id;
    int loc;
    uint64_t offset;
    uint64_t size;
  };

  struct del_t {
    int loc;
    uint64_t offset;
    uint64_t size;
  };

  struct op_t {
  private:
    using _op_t = std::variant<
      input_t, apply_t, move_t,
      evict_t, load_t, del_t>;
  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t  x): op_t(_op_t(x)) {}
    op_t(apply_t  x): op_t(_op_t(x)) {}
    op_t(move_t   x): op_t(_op_t(x)) {}
    op_t(evict_t  x): op_t(_op_t(x)) {}
    op_t(load_t   x): op_t(_op_t(x)) {}
    op_t(del_t    x): op_t(_op_t(x)) {}

    bool is_input() const { return std::holds_alternative<input_t>(op); }
    bool is_apply() const { return std::holds_alternative<apply_t>(op); }
    bool is_move()  const { return std::holds_alternative<move_t>(op);  }
    bool is_evict() const { return std::holds_alternative<evict_t>(op); }
    bool is_load()  const { return std::holds_alternative<load_t>(op);  }
    bool is_del()   const { return std::holds_alternative<del_t>(op);   }

    input_t const& get_input() const { return std::get<input_t>(op); }
    apply_t const& get_apply() const { return std::get<apply_t>(op); }
    move_t  const& get_move()  const { return std::get<move_t>(op);  }
    evict_t const& get_evict() const { return std::get<evict_t>(op); }
    load_t  const& get_load()  const { return std::get<load_t>(op);  }
    del_t   const& get_del()   const { return std::get<del_t>(op);   }

    // get all the memlocs touched by
    // this operation
    vector<memloc_t> get_memlocs() const;

  private:
    _op_t op;
  };

  struct node_t {
    op_t op;
    set<int> inns; // This op can be started when these nodes
                   // have completed
    set<int> outs; // These nodes can't be started until this node
                   // is completed
  };
  vector<node_t> nodes;

  int const num_compute_locs;
  int const num_cache_locs;

  // Example: Four gpu node, with ram as the cache, then
  //          cache_locs = {0,0,0,0} and compute locs are 0,1,2,3.
  vector<int> const cache_locs;

private:
  int insert(op_t op, set<int> const& deps);

  // Get whether or not there is a directed path from
  // bot to top
  bool depends_on(int top, int bot) const;
  // For every node, store a vector of 1s and 0s for all nodes
  // that will execute before this node executes.
  // Note also that all_deps[i] has length i--that is,
  // 0,1,2,3,4,.. is a valid order of the graph.
  vector<vector<char>> all_deps;
};
