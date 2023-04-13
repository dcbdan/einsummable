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

  int insert_formation(
    placement_t placement,
    int inn,
    bool is_save = true);
  int insert_formation(
    partition_t partition,
    int inn,
    bool is_save = true);
  int insert_formation(
    vector<uint64_t> shape,
    int inn,
    bool is_save = true);

  // For each non-save node, make sure it gets marked as save if it isn't used
  // elsewhere.
  // For non-formation nodes, if the node is not used elsewhere, insert an outgoing
  // formation node with is_save = true.
  // For formation nodes with is_save = false, if there are no outgoing edges,
  // flip is_save to true.
  void set_saves();
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

  struct formation_t {
    vector<uint64_t> shape;
    bool is_save; // if this is false, it is a temporary

    vector<uint64_t> out_shape() const { return shape; }
  };


  struct op_t {
  private:
    using _op_t = std::variant<input_t, formation_t, einsummable_t>;

  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t       x): op_t(_op_t(x)) {}
    op_t(formation_t   x): op_t(_op_t(x)) {}
    op_t(einsummable_t x): op_t(_op_t(x)) {}

    vector<uint64_t> out_shape() const {
      return std::visit([](auto x){ return x.out_shape(); }, op);
    }
    int out_rank() const {
      return this->out_shape().size();
    }

    vector<uint64_t> shape() const {
      if(std::holds_alternative<input_t>(op)) {
        return std::get<input_t>(op).shape;
      }
      if(std::holds_alternative<formation_t>(op)) {
        return std::get<formation_t>(op).shape;
      }
      if(std::holds_alternative<einsummable_t>(op)) {
        return std::get<einsummable_t>(op).join_shape;
      }
      throw std::runtime_error("graph::op_t should not reach");
      return {};
    }

    int rank() const {
      return this->shape().size();
    }

    bool is_save() const {
      return is_formation() && get_formation().is_save;
    }
    bool is_formation() const {
      return std::holds_alternative<formation_t>(op);
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
    einsummable_t& get_einsummable() {
      return std::get<einsummable_t>(op);
    }

    formation_t const& get_formation() const {
      return std::get<formation_t>(op);
    }
    formation_t& get_formation() {
      return std::get<formation_t>(op);
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

