#pragma once
#include "../base/setup.h"

#include "einsummable.h"
#include "taskgraph.h"
#include <variant>

struct memloc_t;

struct mem_t {
  uint64_t offset;  // |________||xxxxxxxx|
  uint64_t size;

  memloc_t as_memloc(int loc) const;

  static mem_t from_proto(es_proto::Mem const& m);
  void to_proto(es_proto::Mem& m) const;
};

// This does not use std::variant so that it can be sent over the wire.
struct memsto_t {
  memsto_t() {}
  memsto_t(mem_t const& m): _is_mem(true), info { .mem = m } {}
  memsto_t(int sto_id): _is_mem(false), info { .sto_id = sto_id } {}

  bool is_mem() const { return  _is_mem; }
  bool is_sto() const { return !_is_mem; }

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

  static memloc_t from_proto(es_proto::MemLoc const& m);
  void to_proto(es_proto::MemLoc& m) const;
};

struct stoloc_t {
  int loc;    // this storage location
  int id;     // with this id

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
std::ostream& operator<<(std::ostream&, stoloc_t const&);

struct memgraph_make_state_t2;

enum class allocator_strat_t { lowest_dependency, first };

struct allocator_settings_t {
  allocator_strat_t strat;
  uint8_t alignment_power; // 2^alignment_power

  static allocator_settings_t default_settings();

  static allocator_settings_t gpu_alignment_settings();
};

// TODO: refactor memgraph so that a every compute loc has one storage loc
//       and every storage loc has one compute loc. maybe?
struct memgraph_t {
  memgraph_t():
    memgraph_t(1, 1, vector<int>(1, 0))
  {}

  memgraph_t(
    int num_compute_locs,
    int num_storage_locs,
    vector<int> const& storage_locs,
    bool prune_edges = false);

  memgraph_t(
    memgraph_t const& other);

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
    optional<memgraph_t>,
    memgraph_t>
  make_(
    taskgraph_t const& graph,
    vector<int> which_storage = {},
    vector<uint64_t> mem_sizes = {},
    map<int, memstoloc_t> init_input_tid_to_data = {},
    allocator_settings_t settings = allocator_settings_t::default_settings(),
    bool use_storage = true,
    bool split_off_inputs = false);

  static
  tuple<
    map<int, memstoloc_t>,
    map<int, memstoloc_t>,
    memgraph_t>
  make(
    taskgraph_t const& graph,
    vector<int> which_storage = {},
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
  void to_proto(es_proto::MemGraph& mg) const;

  static memgraph_t from_proto(es_proto::MemGraph const& mg);
  static memgraph_t from_wire(string const& str);

  int num_compute_locs;
  int num_storage_locs;

  // Example: Four gpu node, with ram as the storage, then
  //          storage_locs = {0,0,0,0} and compute locs are 0,1,2,3.
  vector<int> storage_locs;

  // A single sto_loc may map into multiple locations,
  // so this function returns all of them.
  vector<int> get_locs_from_storage_loc(int sto_loc) const;

  // Given a node and a device, return whether or not
  // that node "occurs" at that device.
  bool is_local_to(int id, int loc) const;

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
    int storage_loc;
    int storage_id;
    uint64_t size;
    stoloc_t as_stoloc() const { return stoloc_t { storage_loc, storage_id }; }
  };

  struct constant_t {
    int loc;
    uint64_t offset;
    fill_t fill;

    uint64_t get_size() const { return dtype_size(fill.value.dtype) * product(fill.shape); }
    memloc_t as_memloc() const { return memloc_t{offset, get_size(), loc}; }
    mem_t as_mem() const { return as_memloc().as_mem(); }
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
      inputmem_t, inputsto_t, constant_t,
      apply_t, move_t,
      evict_t, load_t, partialize_t,
      alloc_t, del_t>;
  public:
    op_t(_op_t op): op(op) { check_op(); }

    op_t(inputmem_t   x): op_t(_op_t(x)) {}
    op_t(inputsto_t   x): op_t(_op_t(x)) {}
    op_t(constant_t   x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(evict_t      x): op_t(_op_t(x)) {}
    op_t(load_t       x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}
    op_t(alloc_t      x): op_t(_op_t(x)) {}
    op_t(del_t        x): op_t(_op_t(x)) {}

    bool is_inputmem()   const { return std::holds_alternative<inputmem_t>(op);   }
    bool is_inputsto()   const { return std::holds_alternative<inputsto_t>(op);   }
    bool is_constant()   const { return std::holds_alternative<constant_t>(op);   }
    bool is_apply()      const { return std::holds_alternative<apply_t>(op);      }
    bool is_move()       const { return std::holds_alternative<move_t>(op);       }
    bool is_evict()      const { return std::holds_alternative<evict_t>(op);      }
    bool is_load()       const { return std::holds_alternative<load_t>(op);       }
    bool is_partialize() const { return std::holds_alternative<partialize_t>(op); }
    bool is_alloc()      const { return std::holds_alternative<alloc_t>(op);      }
    bool is_del()        const { return std::holds_alternative<del_t>(op);        }

    inputmem_t   const& get_inputmem()   const { return std::get<inputmem_t>(op);   }
    inputsto_t   const& get_inputsto()   const { return std::get<inputsto_t>(op);   }
    constant_t   const& get_constant()   const { return std::get<constant_t>(op);   }
    apply_t      const& get_apply()      const { return std::get<apply_t>(op);      }
    move_t       const& get_move()       const { return std::get<move_t>(op);       }
    evict_t      const& get_evict()      const { return std::get<evict_t>(op);      }
    load_t       const& get_load()       const { return std::get<load_t>(op);       }
    partialize_t const& get_partialize() const { return std::get<partialize_t>(op); }
    alloc_t      const& get_alloc()      const { return std::get<alloc_t>(op);      }
    del_t        const& get_del()        const { return std::get<del_t>(op);        }

    // check and get einsummable
    bool is_einsummable() const { return is_apply() && std::holds_alternative<einsummable_t>(get_apply().op); }
    bool is_touch()       const { return is_apply() && std::holds_alternative<touch_t>(get_apply().op); }
    einsummable_t get_einsummable() const {
      if (!is_einsummable()) throw std::runtime_error("trying to get einsummable for an non-einsummable op");
      return std::get<einsummable_t>(get_apply().op);
    }
    touch_t get_touch() const {
      if (!is_touch()) throw std::runtime_error("trying to get touch for an non-touch op");
      return std::get<touch_t>(get_apply().op);
    }
    bool is_contraction() const { return is_einsummable() && get_einsummable().is_contraction(); }

    void print_type() {
      if (is_inputmem() || is_inputsto())      std::cout << "input";
      if (is_constant())   std::cout << "constant";
      if (is_move())       std::cout << "move";
      if (is_evict())      std::cout << "evict";
      if (is_load())       std::cout << "load";
      if (is_partialize()) std::cout << "partialize";
      if (is_del())        std::cout << "del";
      if (is_alloc())      std::cout << "alloc";

      if (is_einsummable()) {
        if (is_contraction()) std::cout << "contraction";
        else                  std::cout << "other einsum";
      }
      if (is_touch())       std::cout << "touch";
    }

    int get_loc() const{
      if (is_inputmem())   return get_inputmem().loc;
      if (is_constant())   return get_constant().loc;
      if (is_apply())      return get_apply_loc();
      if (is_move())       return get_move().get_dst_loc();
      if (is_evict())      return get_evict().src.loc;
      if (is_load())       return get_load().dst.loc;
      if (is_partialize()) return get_partialize().loc;
      if (is_alloc())      return get_alloc().loc;
      if (is_del())        return get_del().loc;

      if (is_inputsto()) {
        throw std::runtime_error("input sto can have multiple locs");
      };

      throw std::runtime_error("trying to get loc for an unknown op");
    }

    int get_apply_loc() const {
      if (!is_apply()) throw std::runtime_error("trying to get apply loc for a non-apply op");
      return get_apply().loc; }

    // get all the memlocs touched by this operation
    vector<memloc_t> get_memlocs() const;

    memstoloc_t get_output_memstoloc() const;
    memloc_t    get_output_memloc() const;
    mem_t       get_output_mem() const;

    // get the single stoloc touched by this operation
    stoloc_t    get_stoloc() const;

    bool is_local_to(int loc) const;
    string get_name() const;

  private:
    _op_t op;

    void check_op()         const;

    void check_inputmem()   const;
    void check_inputsto()   const;
    void check_constant()   const;
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

  int insert(op_t op, set<int> const& deps);

private:
  friend class memgraph_make_state_t2;

  // Get whether or not there is a directed path from
  // bot to top
  bool depends_on(int top, int bot) const;
  // For every node, store a vector of 1s and 0s for all nodes
  // that will execute before this node executes.
  // Note also that all_deps[i] has length i--that is,
  // 0,1,2,3,4,.. is a valid order of the graph.
  vector<vector<char>> all_deps;
  bool prune_edges;
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

  // tuple<uint64_t, vector<int>>
  // allocate(uint64_t size);

  optional<tuple<uint64_t, set<int>>> allocate(uint64_t sz);

  optional<tuple<vector<uint64_t>, vector<set<int>>>> allocate_multiple(vector<uint64_t> sizes);
  // This function is specifically for allocating without any dependencies.
  // It will try to allocate a block without any deps and on failure returns none.
  optional<uint64_t>
  try_to_allocate_without_deps(uint64_t size);

  void allocate_at_without_deps(uint64_t offset, uint64_t size);

  void set_strategy(allocator_strat_t s) { strat = s; };
  allocator_strat_t get_strategy() const { return strat; }

  // delete this memory, storing the delete dependent
  // for future use of this memory block
  void free(uint64_t offset, int del);

  void print() const;

  // chcek that no memory is being taken up
  bool is_empty() const;

  // If the memory at offset is occupied, return the corresponding
  // occupied interval. Else return None.
  optional<tuple<uint64_t, uint64_t>>
  get_allocated_region(uint64_t offset) const;

  // Remove all dependencies from available memory
  void clear_dependencies();

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


struct memgraph_make_state_t2 {
  memgraph_make_state_t2(
    taskgraph_t const& taskgraph,
    vector<int> const& which_storage,
    vector<allocator_t> const& empty_allocators,
    map<int, memstoloc_t>& input_tid_to_data,
    int num_compute,
    int num_storage,
    bool use_storage);

  using op_t         = memgraph_t::op_t;
  using inputmem_t   = memgraph_t::inputmem_t;
  using inputsto_t   = memgraph_t::inputsto_t;
  using constant_t   = memgraph_t::constant_t;
  using apply_t      = memgraph_t::apply_t;
  using move_t       = memgraph_t::move_t;
  using partialize_t = memgraph_t::partialize_t;
  using alloc_t      = memgraph_t::alloc_t;
  using del_t        = memgraph_t::del_t;
  using evict_t      = memgraph_t::evict_t;
  using load_t       = memgraph_t::load_t;

  void initialize_input(int inn);

  bool input_has_been_initialized(int inn);

  // void add_to_memgraph(
  //   std::variant<_which_node_t, _which_touch_t> const& which_op);

  // This calls add to memgraph for every op, but also sets up all metadata
  // for eviction and loading
  void process(
    vector<std::variant<_which_node_t, _which_touch_t>> const& all_ops);

  bool allocate_op(std::variant<_which_node_t, _which_touch_t> const& which_op, bool force=false);
  bool add_op(std::variant<_which_node_t, _which_touch_t> const& which_op);
  vector<tuple<int, mem_t>> get_tensors_in_memory_without_alloc(vector<int> const& task_ids);


  int get_group_at(int task_id, int unit_id);

  // At the end of this call, these tensors should be in memory. If they can't
  // all be in memory, then an error is thrown. If the tensor isn't yet created,
  // a tensor of the correct size is allocated.
  // vector<tuple<int, mem_t>> get_tensors_in_memory(vector<int> const& task_ids);

  // Load as many tensors as possible, with a maximum number of bytes
  // loaded at hint.
  // The algorihtm is:
  //   1. find all tensors in storage less than size hint and will
  //      be used again
  //   2. load the tensor that is used earliest
  //   3. decrement hint and recurse
  // If allocation fails or there are no tensors smaller than
  // hint, stop.
  void load_tensors_until(int loc, uint64_t hint);

  // find the tid that
  // 1. is bigger than size and
  // 2. not in `cannot_evict` and
  // 3. will be used latest into the future among tids that
  //    satisfy 1 and 2
  optional<int> find_victim(int loc, uint64_t size, vector<int> cannot_evict = {});
  // If not tensors satisfy 1 and 2, return None.

  // Insert an allocate node and return the alloc_t mem id
  int allocate_with_evict(
    int loc, uint64_t size,
    vector<int> cannot_evict = {});

  optional<int> allocate_without_evict(int loc, uint64_t size);

  optional<vector<int>> allocate_multiple_without_evict(
  int loc, vector<uint64_t> sizes);

  // push this tensor onto memory
  void evict_tensor(int tid);

  // load tid into memory, possibly evicting tensors.
  // Don't evict any items in cannot_evict
  void load_tensor_with_evict(int tid, vector<int> cannot_evict = {});

  // if this cannot allocate memory, will return false
  bool load_tensor_without_evict(int tid);

  bool load_multiple_without_evict(vector<int> tids, bool has_output_in_tids);

  void _load_tensor_helper(int tid, int alloc_mid);

  // TODO: where should tensor donation occur?

  void print_task_node_to_mem_node(map<_which_node_t, int> task_node_to_mem_node);
  void print_task_touch_to_mem_node(map<_which_touch_t, int> task_touch_to_mem_node);

  // this tensor was used, see if you can free the memory
  bool register_usage(int task_id);

  memgraph_t pop_memgraph();

  // A bunch of helper methods to modify
  //   task_tensor_to_mem_node,
  //   tensors_on_memory,
  //   tensors_on_storage
  void task_tensor_to_mem_node_insert_on_storage(int tid, int mid);
  void task_tensor_to_mem_node_insert_on_memory(int tid, int mid);
  void _task_tensor_to_mem_node_insert(int tid, int mid);

  void task_tensor_to_mem_node_update_on_storage(int tid, int mid);
  void task_tensor_to_mem_node_update_on_memory(int tid, int mid);
  void _task_tensor_to_mem_node_update(int tid, int mid);

  void task_tensor_to_mem_node_erase_on_storage(int tid);
  void task_tensor_to_mem_node_erase_on_memory(int tid);
  void _task_tensor_to_mem_node_erase(int tid);

  taskgraph_t const& taskgraph;

  memgraph_t memgraph;

  vector<allocator_t> allocators;

  int _group;
  map<tuple<int,int>, int> to_group;

  bool const use_storage;

  // A mapping from partialize id to all apply memids doing a touch
  map<int, vector<int>> partializes_in_progress;

  // These objects should tend to be updated together {{{
  // Mappings from the taskgraph tensor to the corresponding
  // mem graph node. This gets updated as tensors get evicted,
  // and fully computed.
  map<int, int> task_tensor_to_mem_node;

  // these are all the tensors (represented as tids) that are in storage
  set<int> tensors_on_storage;

  // tensors_on_memory is a map from tid to all of the mids that have used
  // that tensor while it has been in memory.
  map<int, set<int>> tensors_on_memory;
  // }}}

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

  struct order_state_t {
    // For each tid, when is it used
    vector<set<int>> when_used;

    // any usage less than this may get removed
    int threshold;

    // Return the next time a tensor is "used".
    // An tid is "used" at each t in when_used[tid] provided t >= threshold.
    int get(int tid);
    // This method may update when_used; it is assumed that threshold is only
    // increased.
  };
  optional<order_state_t> order_state;
};
