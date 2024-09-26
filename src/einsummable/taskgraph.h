#pragma once
#include "../base/setup.h"

#include "einsummable.h"
#include "graph.h"
#include "touch.h"

struct regiondim_t {
  uint64_t dim;
  uint64_t offset;
  uint64_t size;
};

// Every tensor is either repliacted across locations or split at a dimension
struct model_parallel_placement_t {
  static model_parallel_placement_t make_replicate() {
    return model_parallel_placement_t { .which = -1 }; }

  static model_parallel_placement_t make_split(int which) {
    if(which < 0) { throw std::runtime_error("this is not split"); }
    return model_parallel_placement_t { .which = which };
  }

  bool replicate() const { return which <  0; }

  bool partition() const { return which >= 0; }
  bool split() const { return which >= 0; }

  int split_dim() const {
    if(partition()) {
      return which;
    } else {
      throw std::runtime_error("no split dim, this tensor is replicated");
    }
  }

  partition_t make_partition(vector<uint64_t> const& shape, int nlocs) const {
    partition_t p = partition_t::singleton(shape);
    if(split()) {
      p.partdims[which] = partdim_t::split(shape[which], nlocs);
    } else {
      // correct
    }
    return p;
  }

  int which; // for <0, replicate; for >= 0, split that dim
};

struct taskgraph_t {
  static
  tuple<
    map<int, vtensor_t<int> >, // for each input, the taskgraph ids of the blocks
    map<int, vtensor_t<int> >, // for each save graph id, the taskgraph ids of the blocks
    taskgraph_t>              // the actual taskgraph
  make(graph_t const& graph, vector<placement_t> const& placements);

  static
  tuple<
    map<int, vector<int>>, // for each input, the tid for each location
    map<int, vector<int>>, // for each save,  the tid for each location
    taskgraph_t>
  make_model_parallel(
    graph_t const& graph,
    int nlocs,
    map<int, model_parallel_placement_t> const& pls // must specify for every input
  );


  // Methods to construct a task graph object
  // {{{

  // TODO: loc and src is always deducible because an id belongs to a loc
  //       so either remove the loc from the api or assert the
  //       loc is correct

  int insert_input(
    int loc,
    dtype_t dtype,
    vector<uint64_t> shape,
    bool is_save = false);

  int insert_input(
    int loc,
    uint64_t size,
    bool is_save = false);

  int insert_constant(
    int loc,
    fill_t const& fill,
    bool is_save = false);

  int insert_einsummable(
    int loc,
    einsummable_t e,
    vector<int> inns,
    bool is_save = false);

  int insert_move(
    int src,
    int dst,
    int inn,
    bool is_save = false);

  int insert_consumed_aggregate(
    int loc,
    dtype_t dtype,
    castable_t castable,
    vector<int> inns,
    bool is_save = false);

  int insert_select_subset(
    int loc,
    vector<regiondim_t> selection,
    int inn,
    dtype_t dtype,
    bool is_save = false);

  int new_partial(
    int loc,
    dtype_t dtype,
    vector<uint64_t> write_shape,
    bool is_save = false);

  void add_to_partial(
    int id_out,
    int id_inn,
    touch_t touch,
    bool consume = false);

  void add_to_partial_the_full_aggregate(
    int id_out,
    int id_inn,
    castable_t castable,
    bool consume);
  // }}}

  uint64_t get_size_at(int id) const;

  // a passthrough partial is a partialize node that
  // (1) is not a save
  // (2) does no aggregation and
  // (3) is only used by other partializes (with the same shape)
  bool is_passthrough_partial(int id) const;

  set<int> collect_passthrough_partials() const;

private:
  optional<
    tuple<
      map<int, int>,
      taskgraph_t > >
  remove_passthrough_partials() const;

  tuple<
    map<int, int>,
    taskgraph_t >
  prune() const;
  // These simplifications maintain that all inn nodes are still there and all
  // save nodes are still there, but everything inbetween may be different.
  // A map which is guaranteed to have all source save and inn tids is returned

public:
  // for debugging
  void print() const;

  void print_graphviz(std::ostream& out) const;
  void print_graphviz(
    std::ostream& out,
    vector<string> const& colors) const;

  // Get a compute order for the graph.
  // (It is not neccesarily the case that get_order = 0,1,2,...
  //  because things can get added to partialize ops)
  vector<int> get_order() const;
  vector<int> get_reverse_order() const;

  tuple<vector<int>, vector<int>> get_input_core_order() const;
  set<int> get_input_everywhere_ids() const;

  // If a taskgraph node has zero outputs, it better be a save.
  // Return whether or not this holds for all nodes
  bool all_zero_outs_is_save() const;

  bool all_valid_partialize() const;

  int num_locs() const;

  uint64_t total_bytes_moved() const;
  uint64_t total_flops() const;

  // Get a nodes x num_locs memory usage
  vtensor_t<uint64_t> possible_memory_usage() const;

  string to_wire() const;
  static taskgraph_t from_wire(string const& str);

  uint64_t out_size(int id) const { return nodes[id].op.out_size(); }
  int out_loc(int id) const { return nodes[id].op.out_loc(); }

  bool is_save(int id) const { return nodes[id].is_save; }

  bool is_local_to(int id, int loc) const { return nodes[id].op.is_local_to(loc); }

//private:
public:
  struct input_t {
    int loc;
    uint64_t size;
  };
  struct apply_t {
    int loc;
    vector<int> inns;
    einsummable_t einsummable;
  };
  struct move_t {
    int src;
    int dst;
    int inn;
    uint64_t size;
  };
  struct constant_t {
    int loc;
    fill_t fill;
  };

  // Some words are neccessary to describe what a partialize is.
  // A partialize operation produces a single tid.
  // Partialize represent a generalization of "multiple" partial reduces.
  // Partialize operations may have concurrent
  // write operations and may have multiple order-agnostic increments.
  //
  // Consider A = (B1; B2) + (C1; C2) + D + E
  // Here, A D and E are contiguous, but B and C are split in the same way.
  // And we want to compute
  //   A1 = B1 + C1 + D1 + E1
  //   A2 = B2 + C2 + D2 + E2
  // This is represented as a partialize:
  // 1. Have two separate concurrent writes, one for A1 and another for A2.
  //    When both complete, the tid for A is available.
  // 2. Each A1 and A2 consists of an aggregation of some inputs. Take A1:
  //    B1,C1,D and E may be available at different times, so write into A1
  //    one at a time and order is not important. On the first call, initialize.
  //    So if D is avilable first, copy the corresponding range of D1 into A1 and
  //    for each input therafter, sum into the output.
  // Note that here, we are not forming A1, D1 or E1, but instead only a portion of
  // A, D and E are being used at a time. These are specified with an
  // offset for each dimenison. Consider
  //   struct touchdim_t {
  //     uint64_t d_inn;
  //     uint64_t d_out;
  //     uint64_t offset_inn;
  //     uint64_t offset_out;
  //     uint64_t size;
  //   };
  // and vectors x,y with x[:u] := y[v:v+u]. That has
  //   d_inn = |y|
  //   d_out = |x|
  //   offset_inn = v
  //   offset_out = 0
  //   size = u
  // For arbitrary ranked tensors, life gets more complicated. In this case,
  // a offset and size have to be specified for each dimension and the product
  // over the dimensions specifies the given regions.
  //
  // A partial_unit is the Ax = Bx + Cx + Dx + Ex
  // and a partialize is a list of partialize units.
  // It must be the case that each partial_unit is representing
  // a disjoint portion of the output.
  //
  // Since all units are writing to different output memories, they can
  // all be done in parallel. However, all writes within a unit cannot
  // be done at the same time because they are all writing into the
  // same output memory. A unit can be split into many units by
  // partitioning the unit's output region.
  //
  // Another tidbit: every input that has the same shape as the output may
  // be "consumed"--that is, if a consumable input is the first input ready, the
  // memory referred to by the input becomes consumed and used as the output
  // memory.
  struct partialize_t {
    struct out_regiondim_t {
      uint64_t offset;
      uint64_t size;
    };

    struct inn_regiondim_t {
      uint64_t dim;
      uint64_t offset;
    };                             // necc info for where in the input region

    struct input_op_t {
      int id;                      // the input
      bool consumable;             // can the input be consumsed
      vector<inn_regiondim_t> region;
    };
    // TODO: Where does consumable get used? How to use it?
    //       We should be able to determine if it is a consumable without
    //       this flag so it should be removed, maybe

    struct partial_unit_t {
      // Each partial unit can have a different castable.
      // If the number of inputs is one, this value does not get used.
      optional<castable_t> castable;

      // For each dimension in the write shape,
      //   contain the offset and the size for the write region
      vector<out_regiondim_t> out_region;

      // For each input, the relevant info to figure out the region
      // and the initial behaviour (consumable)
      vector<input_op_t> inputs;
    };

    vector<vector<tuple<int, touch_t> > > as_touches_from() const;
    tuple<int, touch_t> get_touch(int which_unit, int which_touch_in_unit) const;

    vector<tuple<int, touch_t>> as_touches_from_flat() const {
      return vector_flatten(as_touches_from());
    }

    int get_num_touches() const {
      return as_touches_from_flat().size();
    }

    static partialize_t make_from_touches(
      int loc, vector<tuple<int, touch_t>> const&);

    // For each unit with n inputs, try to split that unit into
    // n sub-units.
    void make_parallel();

    // determine if the entire write shape has been touched
    // by exactly one unit
    bool valid() const;

    // determine if this op is a direct copy from one
    // buffer to another of the same size
    bool is_straight_copy() const;

    bool does_agg() const;

    // If this partialize has multiple different possible
    // castables, throw an error
    optional<castable_t> get_agg_castable() const;

    // collect all the shapes that id is used as an input
    // (this should usally be of length 1, but you never know)
    vector<vector<uint64_t>> inn_shapes_of(int id) const;

    int loc;
    dtype_t dtype;
    vector<uint64_t> write_shape;
    vector<partial_unit_t> units;
  };

  friend
  bool operator==(
    partialize_t::out_regiondim_t const& lhs,
    partialize_t::out_regiondim_t const& rhs);
  friend
  bool operator!=(
    partialize_t::out_regiondim_t const& lhs,
    partialize_t::out_regiondim_t const& rhs);

public:
  int insert_partialize(partialize_t const& p, bool is_save = false);

  struct op_t {
  private:
    using _op_t = std::variant<input_t, apply_t, move_t, constant_t, partialize_t>;
  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t      x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(constant_t   x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}

    _op_t op;

    int out_loc() const;

    bool is_local_to(int loc) const;

    uint64_t out_size() const;

    set<int> inputs() const;

    bool is_input() const {
      return std::holds_alternative<input_t>(op);
    }
    bool is_apply() const {
      return std::holds_alternative<apply_t>(op);
    }
    bool is_move() const {
      return std::holds_alternative<move_t>(op);
    }
    bool is_constant() const {
      return std::holds_alternative<constant_t>(op);
    }
    bool is_partialize() const {
      return std::holds_alternative<partialize_t>(op);
    }
    bool is_valid_if_partialize() const {
      return !is_partialize() || get_partialize().valid();
    }
    // TODO: check in taskgraph make that these don't occur
    bool is_no_op_partialize() const {
      return is_partialize() && get_partialize().is_straight_copy();
    }

    input_t const& get_input() const { return std::get<input_t>(op); }
    input_t&       get_input()       { return std::get<input_t>(op); }

    apply_t const& get_apply() const { return std::get<apply_t>(op); }
    apply_t&       get_apply()       { return std::get<apply_t>(op); }

    move_t const& get_move() const { return std::get<move_t>(op); }
    move_t&       get_move()       { return std::get<move_t>(op); }

    constant_t const& get_constant() const { return std::get<constant_t>(op); }
    constant_t&       get_constant()       { return std::get<constant_t>(op); }

    partialize_t&       get_partialize()       { return std::get<partialize_t>(op); }
    partialize_t const& get_partialize() const { return std::get<partialize_t>(op); }

    op_t remap(std::function<int(int)> to_new_tid) const;

    vector<vector<tuple<int, touch_t>>> get_touches() const {
      return get_partialize().as_touches_from();
    }
  };

  struct node_t {
    node_t(op_t op, bool is_save): op(op), is_save(is_save), barrier(0) {}

    op_t op;
    set<int> outs;
    bool is_save;
    int barrier; // not meaningful for input_t ops
  };
  vector<node_t> nodes;

private:
  int insert(op_t op, bool is_save);

  // simplify as many nodes as possible;
  // return the number simplified in some way
  int simplify();

  void _replace_with_new_node(int tid, op_t const& op);
  void _replace_with_fill(int tid, fill_t const& fill);
  bool _replace_apply(int tid);
  bool _replace_partialize(int tid);

  // A helper class for _replace_partialize
  struct _unit_simplify_t {
    using input_op_t     = taskgraph_t::partialize_t::input_op_t;
    using partial_unit_t = taskgraph_t::partialize_t::partial_unit_t;

    _unit_simplify_t(
      taskgraph_t const& tg,
      dtype_t dtype,
      partial_unit_t const& unit);

    bool is_changed() const;
    bool is_constant() const;
    scalar_t get_constant() const;
    vector<input_op_t> get_inputs() const;

  private:
    // 1. was already a constant               ; unchanged
    // 2. was not simplified                   ; unchanged
    // 3. was simplified to a constant         ; changed
    // 4. was simplified but not to a constant ; changed
    // 2. evaluates to a constant, changed

    taskgraph_t const& tg;

    // Case input_op_t         : is a constant
    // Case vector<input_op_t> : is not a constant
    std::variant<input_op_t, vector<input_op_t>> op;

    bool changed;

    optional<scalar_t> get_constant_at(input_op_t const& input) const;

    //   These are annihilator functions:
    //     (+ inf) (+ -inf) (max inf) (min -inf) (mul 0)
    static bool is_annihilator(dtype_t dtype, castable_t castable, scalar_t val);

    //   These are identity functions:
    //     (+ 0), (max -inf) (min inf) (mul 1)
    static bool is_identity(dtype_t dtype, castable_t castable, scalar_t val);
  };
};

partition_t double_last_dim(partition_t const& p);
placement_t double_last_dim(placement_t const& p);

void double_last_dim_inplace(partition_t& p);
void double_last_dim_inplace(placement_t& p);

partition_t halve_last_dim(partition_t const& p);
placement_t halve_last_dim(placement_t const& p);

void halve_last_dim_inplace(partition_t& p);
void halve_last_dim_inplace(placement_t& p);

partition_t convert_squeezer_partition(
  vector<uint64_t> new_shape,
  partition_t const& part);

bool operator==(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs);
bool operator!=(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs);

std::ostream& operator<<(std::ostream& out, touchdim_t const&);
std::ostream& operator<<(std::ostream& out, touch_t const&);

bool tg_do_simplify();
void set_tg_do_simplify(bool b);

/////////////////////////////////////

struct multiple_placement_t {
  multiple_placement_t(
    partition_t const& pa,
    vtensor_t<set<int>> const& ls);

  static multiple_placement_t from_single_placement(placement_t const& p);

  static multiple_placement_t make_refinement(vector<placement_t> const& ps);

  static multiple_placement_t make_refinement(vector<multiple_placement_t> const& ps);

  // deduce the required multiple placement of an einsummable's
  // input at which_input given that the einsummable is placed with
  // with join_placement
  static multiple_placement_t make_einsummable_input(
    placement_t const& join_placement,
    einsummable_t const& einsummable,
    int which_input);
  static multiple_placement_t make_select_input(
    placement_t const& join_placement,
    select_t const& select,
    int which_input);

  partition_t partition;
  vtensor_t<set<int>> const locations;
  // Note: it is possible to have empty location sets
  //       (from a subset operation, for example)
};

multiple_placement_t construct_refinement_placement(
  graph_t const& graph,
  int gid,
  std::function<placement_t const&(int)> get_placement);

std::ostream& operator<<(std::ostream& out, model_parallel_placement_t const&);
bool operator==(model_parallel_placement_t const&, model_parallel_placement_t const&);
bool operator!=(model_parallel_placement_t const&, model_parallel_placement_t const&);
