#pragma once
#include "setup.h"

#include "einsummable.h"
#include "graph.h"

struct touchdim_t {
  uint64_t d_inn;
  uint64_t d_out;
  uint64_t offset_inn;
  uint64_t offset_out;
  uint64_t size;
};

struct touch_t {
  vector<touchdim_t> selection;
  optional<castable_t> castable;
};

struct regiondim_t {
  uint64_t dim;
  uint64_t offset;
  uint64_t size;
};

struct taskgraph_t {
  static
  tuple<
    map<int, tensor_t<int> >, // for each input, the taskgraph ids of the blocks
    map<int, tensor_t<int> >, // for each save graph id, the taskgraph ids of the blocks
    taskgraph_t>              // the actual taskgraph
  make(graph_t const& graph);
  // TODO: The conversion between tensors is a bit tricky.
  //       See what to optimize.

  // Methods to construct a task graph object
  // {{{

  // TODO: loc and src is always deducible because an id belongs to a loc
  //       so either remove the loc from the api or assert the
  //       loc is correct

  int insert_input(
    int loc,
    vector<uint64_t> shape,
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
    castable_t castable,
    vector<int> inns,
    bool is_save = false);

  int insert_select_subset(
    int loc,
    vector<regiondim_t> selection,
    int inn,
    bool is_save = false);

  int new_partial(
    int loc,
    vector<uint64_t> write_shape,
    bool is_save = false);

  void add_to_partial(
    int id_out,
    int id_inn,
    touch_t touch,
    bool consume = false);

  void add_to_partial_the_full_input(
    int id_out,
    int id_inn,
    vector<tuple<uint64_t, uint64_t>> hrect_out,
    bool consume = false);

  void add_to_partial_the_full_aggregate(
    int id_out,
    int id_inn,
    castable_t castable,
    bool consume);
  // }}}

  uint64_t get_size_at(int id) const;

  // find all partial inputs that can be consumed and
  // consume them
  void insert_possible_consumables() { /* TODO */ }

  // for debugging
  void print() const;

  // Get a compute order for the graph.
  // (It is not neccesarily the case that get_order = 0,1,2,...
  //  because things can get added to partialize ops)
  vector<int> get_order() const;

  // If a taskgraph node has zero outputs, it better be a save.
  // Return whether or not this holds for all nodes
  bool all_zero_outs_is_save() const;

  bool all_valid_partialize() const;

private:
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
  //   d_out = |x]
  //   offset_inn = 0
  //   offset_out = v
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

    // determine if the entire write shape has been touched
    // by exactly one unit
    bool valid() const;

    int loc;
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
  struct op_t {
  private:
    using _op_t = std::variant<input_t, apply_t, move_t, partialize_t>;
  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t      x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}

    _op_t op;

    uint64_t tensor_size() const;

    set<int> inputs() const;

    partialize_t&       get_partialize()       { return std::get<partialize_t>(op); }
    partialize_t const& get_partialize() const { return std::get<partialize_t>(op); }

    bool is_input() const {
      return std::holds_alternative<input_t>(op);
    }
    bool is_apply() const {
      return std::holds_alternative<apply_t>(op);
    }
    bool is_move() const {
      return std::holds_alternative<move_t>(op);
    }
    bool is_partialize() const {
      return std::holds_alternative<partialize_t>(op);
    }
    bool is_valid_if_partialize() const {
      return !is_partialize() || get_partialize().valid();
    }
    int output_loc() const {
      if(is_input()) {
        return std::get<input_t>(op).loc;
      }
      if(is_apply()) {
        return std::get<apply_t>(op).loc;
      }
      if(is_move()) {
        return std::get<move_t>(op).dst;
      }
      if(is_partialize()) {
        return std::get<partialize_t>(op).loc;
      }
      throw std::runtime_error("should not reach: output_loc");
    }
    input_t const& get_input() const {
      return std::get<input_t>(op);
    }
    apply_t const& get_apply() const {
      return std::get<apply_t>(op);
    }
    move_t const& get_move() const {
      return std::get<move_t>(op);
    }
    vector<vector<tuple<int, touch_t>>> get_touches() const {
      return std::get<partialize_t>(op).as_touches_from();
    }

  };

  struct node_t {
    node_t(op_t op, bool is_save): op(op), is_save(false) {}

    op_t op;
    set<int> outs;
    bool is_save;
  };
  vector<node_t> nodes;

private:
  int insert(op_t op, bool is_save);
};

bool operator==(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs);
bool operator!=(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs);

std::ostream& operator<<(std::ostream& out, touchdim_t const&);
std::ostream& operator<<(std::ostream& out, touch_t const&);