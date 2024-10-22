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

// #define USE_LOCATIONWISE_APPLY_ORDERING

struct _which_op_t {
  _which_op_t(_which_node_t const& x): _op(x) {}
  _which_op_t(_which_touch_t const& x): _op(x) {}

  int get_tid() const {
    if(is_which_node())  { return get_which_node().task_id; }
    if(is_which_touch()) { return get_which_touch().task_id; }
    throw std::runtime_error("get_tid _which_op_t: missing case, should not reach");
  }

  bool is_which_node() const {
    return std::holds_alternative<_which_node_t>(_op);
  }
  bool is_which_touch() const {
    return std::holds_alternative<_which_touch_t>(_op);
  }

  _which_node_t const& get_which_node() const {
    return std::get<_which_node_t>(_op);
  }
  _which_touch_t const& get_which_touch() const {
    return std::get<_which_touch_t>(_op);
  }

  std::variant<_which_node_t, _which_touch_t> _op;
};

std::ostream& operator<<(std::ostream&, _which_op_t const&);

bool operator==(_which_node_t const& lhs, _which_node_t const& rhs);
bool operator<(_which_node_t const& lhs, _which_node_t const& rhs);

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs);
bool operator< (_which_touch_t const& lhs, _which_touch_t const& rhs);

std::ostream& operator<<(std::ostream& out, _which_op_t const& x);

// Get all (inn, which_touch_t) from partialize node out
vector<tuple<int, _which_touch_t>> get_which_touches_from(taskgraph_t const& taskgraph, int out);

vector<_which_touch_t> get_which_touches_from_to(taskgraph_t const& tg, int out, int inn);

vector<_which_op_t> order_taskgraph(taskgraph_t const& taskgraph);
// ^ TODO: This ordering should be "wide" for parallelism,
//         but not too wide for full breadth-first search.
//         At the moment, this guys is built off of
//         taskgraph.get_order()

tuple<vector<_which_op_t>, vector<_which_op_t>> order_split_taskgraph(taskgraph_t const& taskgraph);

vector<_which_op_t>
order_taskgraph_priority_min_delta(taskgraph_t const& taskgraph);

vector<uint64_t>
order_taskgraph_memory_usage(
  taskgraph_t const& taskgraph,
  vector<_which_op_t> const& ops);

vector<_which_op_t>
build_tg_ops(
  taskgraph_t const& taskgraph,
  vector<int> const& tids_in_order);

struct memgraph_make_state_t {
    memgraph_make_state_t(taskgraph_t const&         taskgraph,
                          vector<int> const&         which_storage,
                          vector<allocator_t> const& empty_allocators,
                          map<int, memstoloc_t>&     input_tid_to_data,
                          int                        num_compute,
                          int                        num_storage,
                          bool                       use_storage);

    using op_t = memgraph_t::op_t;
    using inputmem_t = memgraph_t::inputmem_t;
    using inputsto_t = memgraph_t::inputsto_t;
    using constant_t = memgraph_t::constant_t;
    using apply_t = memgraph_t::apply_t;
    using move_t = memgraph_t::move_t;
    using copy_t = memgraph_t::copy_t;
    using safe_copy_t = memgraph_t::safe_copy_t;
    using partialize_t = memgraph_t::partialize_t;
    using alloc_t = memgraph_t::alloc_t;
    using del_t = memgraph_t::del_t;
    using evict_t = memgraph_t::evict_t;
    using load_t = memgraph_t::load_t;

    void initialize_input(int inn);

  bool input_has_been_initialized(int inn) const;

  bool is_unused_input(int inn) const;

  //a helper function that finds the used tids for a given op. (inns)
  vector<int> find_used_tids(_which_op_t const& which_op) const;

    // This calls add to memgraph for every op, but also sets up all metadata
    // for eviction and loading
    void process(vector<_which_op_t> const& all_ops);

  bool allocate_op_output(_which_op_t const& op, bool force);

  bool allocate_tensor_without_evict(int tid);

  void allocate_tensor_force(
    int tid,
    vector<int> const& keep_tids);

  void add_op(_which_op_t const& which_op);

  // Insert an allocate node and return the alloc_t mem id
  int allocate_with_evict(int loc, uint64_t size, vector<int> cannot_evict = {});

  // When we cannot allocate even after evict, but actually the size is enough for the op to happen
  // it's just that the used tensors are being put in an inappropriete position.
  // return false if size of used tensor > size of buffer, then wee have to something else than simply rearranging
  bool rearrange_allocator(vector<int> cannot_evict, int loc, uint64_t out_size);


  // Try to insert an allocate node and return the alloc_t mem id
  optional<int>
  allocate_without_evict(int loc, uint64_t size);

  //specifically used when we rearrange the allocator. 
  // Will try to allocate with first available 
  //  (should not return nullopt because we have already evicted everything in the front)
  optional<tuple<int, uint64_t>>
  allocate_without_evict_first_available(int loc, uint64_t size);

  optional<vector<int>> find_victims(
    int loc,
    uint64_t size,
    vector<int> cannot_evict = {});

  // load tid on storage into memory, possibly evicting tensors.
  // Don't evict any items in cannot_evict
  void load_tensor_with_evict(
    int tid,
    vector<int> cannot_evict = {});

    void _load_tensor_helper(int tid, int alloc_mid);

  vector<tuple<int, mem_t>>
  get_tensors_in_memory_without_alloc(vector<int> const& task_ids);

    // this tensor was used, see if you can free the memory
    bool register_usage(int task_id);

    int get_group_at(int task_id, int unit_id);

    // push this tensor onto memory
    void evict_tensor(int tid);

    // if this cannot allocate memory, will return false
    bool load_tensor_without_evict(int tid);

    // TODO: where should tensor donation occur?

    void print_task_node_to_mem_node(map<_which_node_t, int> task_node_to_mem_node);
    void print_task_touch_to_mem_node(map<_which_touch_t, int> task_touch_to_mem_node);

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
  void _int_map_print(map<int, int> map);

    taskgraph_t const& taskgraph;

    memgraph_t memgraph;

    vector<allocator_t> allocators;

    int                       _group;
    map<tuple<int, int>, int> to_group;

    bool const use_storage;

    // A mapping from partialize id to all apply memids doing a touch
    map<int, vector<int>> partializes_in_progress;

    // These objects should tend to be updated together {{{
    // Mappings from the taskgraph tensor to the corresponding
    // mem graph node. This gets updated as tensors get evicted,
    // loaded and fully computed.
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

    /* The following fields are used to keep track of how many percentage of the einsu
    m ops can actually happen on cpu, and how many of them can opentially eliminate some
    use of evict and load */
    int at_least_one_on_storage_count; //the number of ops where at least one of the ops is on memory
    int total_apply_nodes; // the number of apply nodes in total
    /* This is used to specify the case where we have A+B->C, and we have one of them on cpu. 
    Then we want to see if C is being evicted in the future, and exactly how many time it has been evicted 
    in the entire graph*/
    map<int, int> accum_tid_to_occurance; 


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

  /*For performance debugging*/
  //mapping from memid to doneoid (add to this map when we add a node to memgraph)
  map<int, int> mem_to_done; 

  //mapping from each memory dependency edge (memid -> memid) to the distance of it
  map<tuple<int, int>, int> mem_deps; 

  //mapping from each edge to the distance of it.
  //  Added everytime we add a node, no matter what it is. 
  map<tuple<int, int>, int> all_deps;

  void mem_to_done_insert(int memid);
  void mem_deps_insert(set<int> deps, int out_mid);
  void all_deps_insert(set<int> deps, int out_mid);
  void print_performance_debugging();
  void print_out_node_info(int out_tid);
};