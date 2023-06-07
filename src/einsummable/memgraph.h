#pragma once
#include "../base/setup.h"

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

struct memgraph_make_state_t;

enum class allocator_strat_t { lowest_dependency, first };

struct allocator_settings_t {
  allocator_strat_t strat;
  uint8_t alignment; // 2^alignment

  static allocator_settings_t default_settings();
};

struct memgraph_t {
  memgraph_t(
    int num_compute_locs,
    int num_cache_locs,
    vector<int> const& cache_locs);

  memgraph_t();

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
    vector<int> const& which_cache,
    vector<uint64_t> mem_sizes = {},
    allocator_settings_t settings = allocator_settings_t::default_settings());

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

  int const num_compute_locs;
  int const num_cache_locs;

  // Example: Four gpu node, with ram as the cache, then
  //          cache_locs = {0,0,0,0} and compute locs are 0,1,2,3.
  vector<int> const cache_locs;

public:
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

  struct partialize_t {
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
      evict_t, load_t,
      partialize_t, del_t>;
  public:
    op_t(_op_t op): op(op) { check_op(); }

    op_t(input_t      x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(evict_t      x): op_t(_op_t(x)) {}
    op_t(load_t       x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}
    op_t(del_t        x): op_t(_op_t(x)) {}

    bool is_input()      const { return std::holds_alternative<input_t>(op);      }
    bool is_apply()      const { return std::holds_alternative<apply_t>(op);      }
    bool is_move()       const { return std::holds_alternative<move_t>(op);       }
    bool is_evict()      const { return std::holds_alternative<evict_t>(op);      }
    bool is_load()       const { return std::holds_alternative<load_t>(op);       }
    bool is_partialize() const { return std::holds_alternative<partialize_t>(op); }
    bool is_del()        const { return std::holds_alternative<del_t>(op);        }

    input_t      const& get_input()      const { return std::get<input_t>(op);      }
    apply_t      const& get_apply()      const { return std::get<apply_t>(op);      }
    move_t       const& get_move()       const { return std::get<move_t>(op);       }
    evict_t      const& get_evict()      const { return std::get<evict_t>(op);      }
    load_t       const& get_load()       const { return std::get<load_t>(op);       }
    partialize_t const& get_partialize() const { return std::get<partialize_t>(op); }
    del_t        const& get_del()        const { return std::get<del_t>(op);        }

    // get all the memlocs touched by
    // this operation
    vector<memloc_t> get_memlocs() const;

    memloc_t get_output_memloc() const;
    mem_t    get_output_mem() const;

  private:
    _op_t op;

    void check_op()         const;

    void check_input()      const;
    void check_apply()      const;
    void check_move()       const;
    void check_evict()      const;
    void check_load()       const;
    void check_partialize() const;
    void check_del()        const;
  };

  struct node_t {
    op_t op;
    set<int> inns; // This op can be started when these nodes
                   // have completed
    set<int> outs; // These nodes can't be started until this node
                   // is completed
  };
  vector<node_t> nodes;

private:
  friend class memgraph_make_state_t;

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

// allocator_t contains a vector of blocks that either
// have been (1) deleted, or (2) are currently occupied
struct allocator_t {
  allocator_t() = delete;

  allocator_t(
    uint64_t memsize_t,
    allocator_settings_t settings = allocator_settings_t::default_settings());

  // Allocate this much memory if possible and return
  // the offset and all dependents. If there is not
  // free memory of this size, none is returned.
  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate(uint64_t size);

  tuple<uint64_t, vector<int>>
  allocate(uint64_t size);

  void set_strategy(allocator_strat_t s) { strat = s; };
  allocator_strat_t get_strategy() const { return strat; }

  // delete this memory, storing the delete dependent
  // for future use of this memory block
  void free(uint64_t offset, int del);

  void print() const;

private:
  struct block_t {
    uint64_t beg;
    uint64_t end;

    // dep is none:
    //   this memory is occupied
    // dep is < 0:
    //   this memory is free and can be used without
    //   adding a dependency
    // dep is >= 0:
    //   this memory is free and can only be used
    //   after dep id has been deleted
    optional<int> dep;

    uint64_t size() const { return end - beg; }
    bool occupied() const  { return !dep.has_value(); }
    bool available() const { return !occupied(); }
    void free(int dep);
  };

  vector<block_t> blocks;
  allocator_strat_t strat;
  uint64_t alignment;

  using iter_t = vector<block_t>::iterator;

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_lowest_dependency_available(uint64_t size);

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_first_available(uint64_t size);
};

struct _which_node_t {
  int task_id;
};
struct _which_touch_t {
  int task_id;
  int unit_id;
  int touch_id;
};

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs);
bool operator< (_which_touch_t const& lhs, _which_touch_t const& rhs);

struct memgraph_make_state_t {
  memgraph_make_state_t(
    taskgraph_t const& taskgraph,
    vector<int> const& which_cache,
    vector<allocator_t> const& as,
    int num_compute,
    int num_cache);

  using op_t         = memgraph_t::op_t;
  using input_t      = memgraph_t::input_t;
  using apply_t      = memgraph_t::apply_t;
  using move_t       = memgraph_t::move_t;
  using partialize_t = memgraph_t::partialize_t;
  using del_t        = memgraph_t::del_t;

  void allocate_inputs();

  void add_to_memgraph(
    std::variant<_which_node_t, _which_touch_t> const& which_op);

  // This function will insert dummy partialize nodes
  // if necessary.
  int task_to_mem(int task_id);

  int get_group_at(int task_id, int unit_id);

  void try_to_delete(int task_id);

  // Allocate the output memory if neccessary.
  // For partials, the memory may have already been allocated.
  // For some einsummables, an input tensor may get donated
  uint64_t get_output_alloc_if_necc(
    int task_id,
    set<int>& deps);

  taskgraph_t const& taskgraph;

  memgraph_t memgraph;

  vector<allocator_t> allocators;

  // taskgraph ids to offsets
  map<int, uint64_t> current_tensors;

  int _group;
  map<tuple<int,int>, int> to_group;

  map<int, int> task_node_to_mem;
  map<_which_touch_t, int> task_touch_to_mem;

  vector<int> remaining_usage_counts;

  set<int> donated;
};



