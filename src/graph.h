#pragma once
#include "setup.h"

#include "placement.h"
#include "einsummable.h"

struct graph_t {
  // Methods to construct a graph object
  // {{{
  int insert_input(
    placement_t placement);
  int insert_input(
    partition_t partition);
  int insert_input(
    vector<uint64_t> shape);

  int insert_einsummable(
    placement_t placement,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    partition_t partition,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    einsummable_t e,
    vector<int> inns);

  int insert_output(
    placement_t placement,
    int inn);
  int insert_output(
    partition_t partition,
    int inn);
  int insert_output(
    vector<uint64_t> shape,
    int inn);

  // for each input and einsummable ops O, make sure O
  // is used by another op. If not, add an output op that
  // uses O.
  // TODO: better name for this method
  void set_outputs();
  // }}}

  vector<uint64_t> out_shape(int id);

  vector<int> get_order() const;

  // TODO: implement a fuse:
  //   void optimize_fuse();
  //
  // Given
  //   B = relu(A); D = BC
  // compile it to a single einsummable of
  //   D = relu(A)C
  // if the intermediate B is not used anywhere.
  //
  // Perhaps similarly for ops like (A+B)C

public:

  struct input_t {
    vector<uint64_t> shape;

    vector<uint64_t> out_shape() const { return shape; }
  };

  struct output_t {
    vector<uint64_t> shape;

    vector<uint64_t> out_shape() const { return shape; }
  };


  struct op_t {
  private:
    using _op_t = std::variant<input_t, output_t, einsummable_t>;

  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t       x): op_t(_op_t(x)) {}
    op_t(output_t      x): op_t(_op_t(x)) {}
    op_t(einsummable_t x): op_t(_op_t(x)) {}

    vector<uint64_t> out_shape() const {
      return std::visit([](auto x){ return x.out_shape(); }, op);
    }

    int out_rank() const {
      return this->out_shape().size();
    }

    bool is_output() const {
      return std::holds_alternative<output_t>(op);
    }
    bool is_input() const {
      return std::holds_alternative<input_t>(op);
    }
    bool is_einsummable() const {
      return std::holds_alternative<einsummable_t>(op);
    }

    einsummable_t const& get_einsummable() const {
      return std::get<einsummable_t>(op);
    }

    _op_t op;
  };

  struct node_t {
    op_t op;
    vector<int> inns;
    set<int> outs;
    placement_t placement;

    set<int> get_inns_set() {
      return set<int>(inns.begin(), inns.end());
    }
    int num_distinct_inputs() {
      return get_inns_set().size();
    }
  };

  vector<node_t> nodes;

private:
  int insert(op_t const& op, vector<int> inns, placement_t placement);
};

