#pragma once
#include "../base/setup.h"

#include "memgraph.h"
#include "mgallocator.h"

struct _which_node_t {
  int task_id;
};
struct _which_touch_t {
  int task_id;
  int unit_id;
  int touch_id;
};

using _which_op_t = std::variant<_which_node_t, _which_touch_t>;

bool operator==(_which_node_t const& lhs, _which_node_t const& rhs);
bool operator< (_which_node_t const& lhs, _which_node_t const& rhs);

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs);
bool operator< (_which_touch_t const& lhs, _which_touch_t const& rhs);

// Get all (inn, which_touch_t) from partialize node out
vector<tuple<int, _which_touch_t>> get_which_touches_from(
  taskgraph_t const &taskgraph,
  int out);

vector<_which_touch_t> get_which_touches_from_to(
  taskgraph_t const &tg,
  int out,
  int inn);

vector<_which_op_t>
order_taskgraph(taskgraph_t const &taskgraph);
// ^ TODO: This ordering should be "wide" for parallelism,
//         but not too wide for full breadth-first search.
//         At the moment, this guys is built off of
//         taskgraph.get_order()

tuple<vector<_which_op_t>, vector<_which_op_t>>
order_split_taskgraph(taskgraph_t const &taskgraph);

vector<_which_op_t>
build_tg_ops(
  taskgraph_t const& taskgraph,
  vector<int> const& tids_in_order);

struct memgraph_make_state_t {
  memgraph_make_state_t(
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

  // This calls add to memgraph for every op, but also sets up all metadata
  // for eviction and loading
  void process(vector<_which_op_t> const& all_ops);

  // Allocate memory for the provided op. If memory wasn't allocated,
  // return false. If force is true, memory will be allocated and
  // may use evict. If force is false, memory may not be allocated and
  // and evict will not be used.
  bool allocate_op(
    _which_op_t const& which_op,
    bool force = false);
  // > get's the required tensors in memory, calling load if necc
  //   > if force: evict tensors as necc
  //   > if not force: don't evict anything
  // > return whether or not the op at oid could be allocated for
  //   > if force, then this must return true or throw an error

  // Do the provided op. Return whether or not any tensors
  // are delteted
  bool add_op(
    _which_op_t const& which_op);
  // > assumption: allocate_op(oid) was called and successful for oid
  // > insert memgraph-op into the memgraph
  // > register usage for all input tensors
  // > return whether or not a delete occurred in one of the register usages

  // return whether or not we can bring all of these tids into memory
  bool allocate_tids_without_evict(vector<int> const& tids);
  // TODO: implement

  // make sure that all of these tids are on memory
  void force_allocate_tids(vector<int> const& tids);

  // Insert an allocate node and return the alloc_t mem id
  int allocate_with_evict(
    int loc, uint64_t size,
    vector<int> cannot_evict = {});

  // Try to insert an allocate note and return the alloc_t mem id
  optional<int> 
  allocate_without_evict(int loc, uint64_t size);

  // find the tid that
  // 1. is bigger than size and
  // 2. not in `cannot_evict` and
  // 3. will be used latest into the future among tids that
  //    satisfy 1 and 2
  optional<int> find_victim(
    int loc, 
    uint64_t size, 
    vector<int> cannot_evict = {});
  // If not tensors satisfy 1 and 2, return None.

  // load tid on storage into memory, possibly evicting tensors.
  // Don't evict any items in cannot_evict
  void load_tensor_with_evict(
    int tid, 
    vector<int> cannot_evict = {});

  void _load_tensor_helper(int tid, int alloc_mid);

  ////////////////////

  vector<tuple<int, mem_t>> 
  get_tensors_in_memory_without_alloc(vector<int> const& task_ids);

  int get_group_at(int task_id, int unit_id);

  // TODO
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

  optional<vector<int>> allocate_multiple_without_evict(
    int loc, 
    vector<uint64_t> sizes);

  // push this tensor onto memory
  void evict_tensor(int tid);

  // if this cannot allocate memory, will return false
  bool load_tensor_without_evict(int tid);

  // TODO: why has_output_in_tids needed?
  bool load_multiple_without_evict(vector<int> const& tids);

  // TODO: where should tensor donation occur?

  void print_task_node_to_mem_node(
    map<_which_node_t, int> task_node_to_mem_node);
  void print_task_touch_to_mem_node(
    map<_which_touch_t, int> task_touch_to_mem_node);

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
