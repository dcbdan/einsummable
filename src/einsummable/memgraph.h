#pragma once
#include "../base/setup.h"

#include "taskgraph.h"

struct memloc_t;

struct mem_t {
  uint64_t offset;
  uint64_t size;

  memloc_t as_memloc(int loc) const;
};

// This does not use std::variant so that it can be sent over the wire.
struct memsto_t {
  memsto_t() {}
  memsto_t(mem_t const& m): _is_mem(true), info { .mem = m } {}
  memsto_t(int sto_id): _is_mem(false), info { .sto_id = sto_id } {}

  bool is_mem() const { return _is_mem; }

  mem_t const& get_mem() const;
  int const& get_sto() const;

  bool _is_mem;
  union {
    mem_t mem;
    int sto_id;
  } info;
};

struct memloc_t {
  uint64_t offset;
  uint64_t size;
  int loc;

  memsto_t as_memsto() const { return memsto_t(as_mem()); }

  mem_t as_mem() const;
};

struct stoloc_t {
  int loc; // this storage location
  int id;  // with this id

  memsto_t as_memsto() const { return memsto_t(id); }
};

struct memstoloc_t {
  memstoloc_t() {}

  memstoloc_t(memloc_t const& m): data(m) {}
  memstoloc_t(stoloc_t const& s): data(s) {}

  bool is_memloc() const { return std::holds_alternative<memloc_t>(data); }
  bool is_stoloc() const { return std::holds_alternative<stoloc_t>(data); }

  memloc_t const& get_memloc() const { return std::get<memloc_t>(data); }
  stoloc_t const& get_stoloc() const { return std::get<stoloc_t>(data); }

  memsto_t as_memsto() const {
    return is_memloc() ? get_memloc().as_memsto() : get_stoloc().as_memsto() ;
  }

  std::variant<memloc_t, stoloc_t> data;
};

std::ostream& operator<<(std::ostream&, mem_t const&);
std::ostream& operator<<(std::ostream&, memloc_t const&);

struct memgraph_make_state_t;

enum class allocator_strat_t { lowest_dependency, first };

struct allocator_settings_t {
  allocator_strat_t strat;
  uint8_t alignment_power; // 2^alignment_power

  static allocator_settings_t default_settings();
};

struct memgraph_t {
  memgraph_t(
    int num_compute_locs,
    int num_storage_locs,
    vector<int> const& storage_locs);

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
    vector<uint64_t> mem_sizes = {},
    allocator_settings_t settings = allocator_settings_t::default_settings());

  static
  tuple<
    map<int, memstoloc_t>,
    map<int, memstoloc_t>,
    memgraph_t>
  make(
    taskgraph_t const& graph,
    vector<int> const& which_storage,
    vector<uint64_t> mem_sizes = {},
    map<int, memstoloc_t> init_input_tid_to_data = {},
    allocator_settings_t settings = allocator_settings_t::default_settings(),
    bool use_storage = true);

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

  string to_wire() const;
  static memgraph_t from_wire(string const& str);

  int const num_compute_locs;
  int const num_storage_locs;

  // Example: Four gpu node, with ram as the storage, then
  //          storage_locs = {0,0,0,0} and compute locs are 0,1,2,3.
  vector<int> const storage_locs;

public:
  struct inputmem_t {
    int loc;
    uint64_t offset;
    uint64_t size;

    memloc_t as_memloc() const { return memloc_t{offset, size, loc}; }
    mem_t as_mem() const { return as_memloc().as_mem(); }
    static inputmem_t from_memloc(memloc_t const& m);
  };

  struct inputsto_t {
    int loc;
    int storage_loc;
    int storage_id;
    stoloc_t as_stoloc() const { return stoloc_t { storage_loc, storage_id }; }
  };

  // An apply needs these memories to do the computation
  // at hand. (for einsummable, output then inn memories)
  // (for touch, write memory then read memories)
  struct apply_t {
    int loc;
    vector<mem_t> mems;
    std::variant<einsummable_t, touch_t> op;
    int group;

    bool is_einsummable() const;
    bool is_touch() const;
    einsummable_t const& get_einsummable() const;
    touch_t const& get_touch() const;
    dtype_t out_dtype() const;
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

    int const& get_src_loc() const { return std::get<0>(src); }
    int const& get_dst_loc() const { return std::get<0>(dst); }
  };

  // Note: every location has one storage, but a
  //       storage may have multiple locations

  // Move this memory off of location loc and into
  // the corresponding storage; every evict should produce
  // a new tensor in the storage and storage_id must be unique
  // across all evicts.
  struct evict_t {
    memloc_t src;
    stoloc_t dst;
  }; // inside the graph, src.loc's storage location must be dst.loc

  // Load from storage this id into loc with this offset and size;
  // This deletes the tensor from the storage.
  // TODO: perhaps it should be possible to
  //       gpu1->storage->gpu2--v
  //                    ->gpu1->delete_from_storage
  //       Now it is only possible to
  //       gpu1->storage->gpu2->delete_from_storage
  struct load_t {
    stoloc_t src;
    memloc_t dst;
  }; // inside the graph dst.loc's storage location must src.loc

  struct partialize_t {
    int loc;
    uint64_t offset;
    uint64_t size;

    memloc_t as_memloc() const { return memloc_t{offset, size, loc}; }
    mem_t as_mem() const { return as_memloc().as_mem(); }
    static partialize_t from_memloc(memloc_t const& m);
  };

  struct alloc_t {
    int loc;
    uint64_t offset;
    uint64_t size;

    memloc_t as_memloc() const { return memloc_t{offset, size, loc}; }
    mem_t as_mem() const { return as_memloc().as_mem(); }
    static alloc_t from_memloc(memloc_t const& m);
  };

  struct del_t {
    int loc;
    uint64_t offset;
    uint64_t size;

    memloc_t as_memloc() const { return memloc_t{offset, size, loc}; }
    mem_t as_mem() const { return as_memloc().as_mem(); }
    static del_t from_memloc(memloc_t const& m);
  };

  struct op_t {
  private:
    using _op_t = std::variant<
      inputmem_t, inputsto_t,
      apply_t, move_t,
      evict_t, load_t, partialize_t,
      alloc_t, del_t>;
  public:
    op_t(_op_t op): op(op) { check_op(); }

    op_t(inputmem_t   x): op_t(_op_t(x)) {}
    op_t(inputsto_t   x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(evict_t      x): op_t(_op_t(x)) {}
    op_t(load_t       x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}
    op_t(alloc_t      x): op_t(_op_t(x)) {}
    op_t(del_t        x): op_t(_op_t(x)) {}

    bool is_inputmem()   const { return std::holds_alternative<inputmem_t>(op);   }
    bool is_inputsto()   const { return std::holds_alternative<inputsto_t>(op);   }
    bool is_apply()      const { return std::holds_alternative<apply_t>(op);      }
    bool is_move()       const { return std::holds_alternative<move_t>(op);       }
    bool is_evict()      const { return std::holds_alternative<evict_t>(op);      }
    bool is_load()       const { return std::holds_alternative<load_t>(op);       }
    bool is_partialize() const { return std::holds_alternative<partialize_t>(op); }
    bool is_alloc()      const { return std::holds_alternative<alloc_t>(op);      }
    bool is_del()        const { return std::holds_alternative<del_t>(op);        }

    inputmem_t   const& get_inputmem()   const { return std::get<inputmem_t>(op);   }
    inputsto_t   const& get_inputsto()   const { return std::get<inputsto_t>(op);   }
    apply_t      const& get_apply()      const { return std::get<apply_t>(op);      }
    move_t       const& get_move()       const { return std::get<move_t>(op);       }
    evict_t      const& get_evict()      const { return std::get<evict_t>(op);      }
    load_t       const& get_load()       const { return std::get<load_t>(op);       }
    partialize_t const& get_partialize() const { return std::get<partialize_t>(op); }
    alloc_t      const& get_alloc()      const { return std::get<alloc_t>(op);      }
    del_t        const& get_del()        const { return std::get<del_t>(op);        }

    // get all the memlocs touched by
    // this operation
    vector<memloc_t> get_memlocs() const;

    memstoloc_t get_output_memstoloc() const;
    memloc_t    get_output_memloc() const;
    mem_t       get_output_mem() const;

    bool is_local_to(int loc) const;
  private:
    _op_t op;

    void check_op()         const;

    void check_inputmem()   const;
    void check_inputsto()   const;
    void check_apply()      const;
    void check_move()       const;
    void check_evict()      const;
    void check_load()       const;
    void check_partialize() const;
    void check_alloc()      const;
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

  // Allocate this much memory if possible and return the offset and all dependents.
  // If there is not free memory of this size, none is returned.
  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate(uint64_t size);

  tuple<uint64_t, vector<int>>
  allocate(uint64_t size);

  // This function is specifically for allocating without any dependencies.
  // It will try to allocate a block without any deps and on failure returns none.
  optional<uint64_t>
  try_to_allocate_without_deps(uint64_t size);

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
  uint64_t alignment_power;

  using iter_t = vector<block_t>::iterator;

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_lowest_dependency_available(uint64_t size);

  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate_impl(uint64_t size_without_rem, bool no_deps);

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

bool operator==(_which_node_t const& lhs, _which_node_t const& rhs);
bool operator< (_which_node_t const& lhs, _which_node_t const& rhs);

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs);
bool operator< (_which_touch_t const& lhs, _which_touch_t const& rhs);

struct memgraph_make_state_t {
  memgraph_make_state_t(
    taskgraph_t const& taskgraph,
    vector<int> const& which_storage,
    vector<allocator_t> const& as,
    map<int, memstoloc_t>& input_tid_to_data,
    int num_compute,
    int num_storage,
    bool use_storage);

  using op_t         = memgraph_t::op_t;
  using inputmem_t   = memgraph_t::inputmem_t;
  using inputsto_t   = memgraph_t::inputsto_t;
  using apply_t      = memgraph_t::apply_t;
  using move_t       = memgraph_t::move_t;
  using partialize_t = memgraph_t::partialize_t;
  using alloc_t      = memgraph_t::alloc_t;
  using del_t        = memgraph_t::del_t;

  void initialize_input(int inn);

  bool input_has_been_initialized(int inn);

  void add_to_memgraph(
    std::variant<_which_node_t, _which_touch_t> const& which_op);

  int get_group_at(int task_id, int unit_id);

  // At the end of this call, these tensors should be in memory. If they can't
  // all be in memory, then an error is thrown. If the tensor isn't yet created,
  // a tensor of the correct size is allocated. It is up to the caller to make
  // sure that newly created tensors make it into task_tensor_to_mem_node
  vector<tuple<int, mem_t>> get_tensors_in_memory(vector<int> const& task_ids);

  // TODO: where should tensor donation occur?

  // this tensor was used, see if you can free the memory
  void register_usage(int task_id);

  taskgraph_t const& taskgraph;

  memgraph_t memgraph;

  vector<allocator_t> allocators;

  int _group;
  map<tuple<int,int>, int> to_group;

  bool const use_storage;

  // A mapping from partialize id to all apply memids doing a touch
  map<int, vector<int>> partializes_in_progress;

  // Mappings from the taskgraph tensor to the corresponding
  // mem graph node. This gets updated as tensors get evicted,
  // and fully computed.
  map<int, int> task_tensor_to_mem_node;

  // A mapping from (apply || move) taskgraph node to the corresponding
  // apply or move
  map<_which_node_t, int> task_node_to_mem_node;
  // A mapping form a taskgraph touch to the corresponding apply
  map<_which_touch_t, int> task_touch_to_mem_node;

  vector<int> remaining_usage_counts;

  // This contains tensors who have been donated
  // to another node
  set<int> donated;

  int _sto_id;

  // A mapping from input tid to where it's stored initially
  map<int, memstoloc_t>& input_tid_to_data;
};
// Some notes about nodes in the taskgraph vs nodes in the memgraph
// and how that relates to tensors.
//
// Every node in the taskgraph produces a tensor:
//   tg input:      produces an input tensor
//   tg apply:      produces a tensor by an einsummable
//   tg move:       produces a tensor by moving from another location
//   tg partialize: have lots of touches that fill out an output tensor
//
// In the memgraph, every node
// 1. is a barrier: this node happens when those nodes have finished
// 2. says something about where a tensor is:
//      this tensor now lives at this memory
//      or at this storage (evict)
//      or it now  to live at this memory (del)
//
// The computation components of the memgraph lives in the apply_t op.
// Every
//   tg apply maps to a single mg apply node and
//   every touch in a tg partialize maps to single mg apply node.
//
// When some touches in a tg partialize occur, the tensor is
// partially computed.
//
// In the taskgraph, a node is a computation _and_ a tensor.
// So we use taskgraph ids to refer to the tensor produced
// by the corresponding taskgraph node. However, in the memgraph,
// every tensor can be moved (evicted, put into different memory, ect).


