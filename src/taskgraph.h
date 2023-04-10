#pragma once
#include "setup.h"

#include "einsummable.h"
#include "graph.h"

// TODO: note that inputs refer to nodes in the task graph,
//       not to wtvr tids are

struct taskgraph_t {
  static
  tuple<
    map<int, tensor_t<int> >, // for each output id, the tids of the blocks
    taskgraph_t>              // the actual taskgraph
  make(graph_t const& graph);

  // Methods to construct a task graph object
  // {{{
  int insert_input(
    int loc,
    vector<uint64_t> shape);
  int insert_einsummable(
    int loc,
    einsummable_t e,
    vector<int> inns);
  int insert_move(
    int src,
    int dst,
    int tid);
  // TODO: what does the partialize interface look like?


  // }}}

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
    struct partial_unit_t {
      struct out_regiondim_t {
        uint64_t offset;
        uint64_t size;
      };
      struct input_op_t {
        int id;                      // the input
        bool consumable;             // can the input be consumsed
        struct inn_regiondim_t {
          uint64_t dim;
          uint64_t offset;
        };                           // necc info for where in the input region
        vector<inn_regiondim_t> region;
      };

      // Each partial unit can have a different castable.
      // If the number of inputs is one, this value does not get used.
      castable_t castable;

      // For each dimension in the write shape,
      //   contain the offset and the size for the write region
      vector<out_regiondim_t> out_region;

      // For each input, the relevant info to figure out the region
      // and the initial behaviour (consumable)
      vector<input_op_t> inputs;
    };

    vector<uint64_t> write_shape;
    vector<partial_unit_t> units;
  };

  struct op_t {
  private:
    using _op_t = std::variant<input_t, apply_t, move_t, partialize_t>;
  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t      x): op_t(_op_t(x)) {}
    op_t(apply_t      x): op_t(_op_t(x)) {}
    op_t(move_t       x): op_t(_op_t(x)) {}
    op_t(partialize_t x): op_t(_op_t(x)) {}
  private:
    _op_t op;
  };

  struct node_t {
    op_t op;
    set<int> outs;
  };
  vector<node_t> nodes;
};

