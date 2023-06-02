#pragma once
#include "../base/setup.h"

#include "../base/placement.h"
#include "einsummable.h"

struct graph_t {
  // Methods to construct a graph object
  // {{{
  int insert_input(
    vector<uint64_t> shape);

  int insert_einsummable(
    einsummable_t e,
    vector<int> inns);

  int insert_formation(
    int inn,
    bool is_save = true);

  // For each non-save node, make sure it gets marked as save if it isn't used
  // elsewhere.
  // For non-formation nodes, if the node is not used elsewhere, insert an outgoing
  // formation node with is_save = true.
  // For formation nodes with is_save = false, if there are no outgoing edges,
  // flip is_save to true.
  void set_saves();

  // TODO: Implement a way to intelligently insert formation nodes.
  //       (any non elementwise einsummable should have one output of
  //        a formation node)
  // Why would you want a formation node that is not also an output node?
  // The reason is to avoid broadcasting partial aggregation results.
  // For instance
  //   X -> A
  //     -> B
  //     -> C
  // If X is partition alot and A,B,C all use blocks at varying locations,
  // each partial will get sent to all the locations the partial gets used
  // at. Doing this instead will leaf to less communication
  //   X -> Formation Node -> A
  //                       -> B
  //                       -> C
  // as the formation node will aggregate partials of X into a
  // single location

  // }}}

  vector<uint64_t> out_shape(int id) const;

  vector<int> get_order() const;

  void print() const;

  vector<int> get_inputs() const;

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
    bool has_aggregation() const {
      return is_einsummable() && get_einsummable().has_aggregation();
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

    set<int> get_inns_set() const {
      return set<int>(inns.begin(), inns.end());
    }
    int num_distinct_inputs() const {
      return get_inns_set().size();
    }
  };

  vector<node_t> nodes;

private:
  int insert(op_t const& op, vector<int> inns);
};

// graph_constructor_t is for building a graph
// object with associated placements.
// graph_writer_t is for building a graph object
// but returning a virtual tensor object

struct graph_constructor_t {
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
    int inn,
    bool is_save = true);

  vector<placement_t> get_placements() const;

  graph_t graph;
  map<int, placement_t> placements;
};

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_constructor_t three_dimensional_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk,
  int num_processors);

// Same tensor dimensions as 3d matmul
// Does not explicitly set locations
graph_constructor_t straight_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk);

struct graph_writer_t {
  struct tensor_t {
    tensor_t transpose(int i, int j) const;
    tensor_t view(vector<uint64_t> shape) const;
    vector<uint64_t> const& get_shape() const { return shape; }
    void save();

    tensor_t& operator=(tensor_t const&);
  private:
    friend class graph_writer_t;

    tensor_t(
      vector<uint64_t> const& shape,
      vector<uint64_t> const& full_shape,
      int id,
      graph_writer_t& self);

    vector<uint64_t> shape;

    vector<uint64_t> full_shape;
    vector<int> modes;

    int id;

    graph_writer_t& self;

  private:
    // after this, the modes are 0,1,...,full_shape.size()-1
    void physically_permute();
    vector<tuple<int,int>> get_breaks() const;
    vector<vector<uint64_t>> _full_shape() const;

    static vector<tuple<int,int>> get_breaks_(
      vector<uint64_t> const& shape,
      vector<uint64_t> const& full_shape);
  };

  struct to_einsummable_info_t {
    vector<uint64_t> full_join_shape;
    vector<uint64_t> join_shape;
    vector<vector<int>> full_inns;
    int full_out_rank;
    int out_rank;

    vector<uint64_t> get_out_full_shape() const;

    vector<uint64_t> get_out_shape() const;

    einsummable_t build_einsummable(
      scalarop_t scalarop,
      optional<castable_t> castable = std::nullopt) const;
  };

  graph_t const& get_graph() const { return graph; }

  // the core ops
  tensor_t input(
    vector<uint64_t> shape);
  tensor_t contraction(
    string str,
    tensor_t const& lhs,
    tensor_t const& rhs);
  tensor_t reduction(
    string str,
    castable_t castable,
    tensor_t const& inn);
  tensor_t ew( // ew = elementwise
    string str,
    scalarop_t op,
    tensor_t const& inn);
  tensor_t ew(
    string str,
    scalarop_t op,
    tensor_t const& lhs,
    tensor_t const& rhs);
  tensor_t ew(
    string str,
    scalarop_t op,
    vector<tensor_t> const& inns);

  // helper ops that dispatch to the core ops

  // add two tensors with the same shape
  tensor_t add(
    tensor_t const& lhs,
    tensor_t const& rhs);

  // straight elementwise scale
  tensor_t scale(
    float val,
    tensor_t const& inn);

  // supports ij,jk->ik and
  //         ...ij,...jk->...ik and
  //         ...ij,jk->ik       and
  //         ij,...jk->ik
  tensor_t matmul(
    tensor_t const& lhs,
    tensor_t const& rhs);
private:
  graph_t graph;

  int _insert_elementwise(
    string str,
    scalarop_t op,
    int id);

  optional<to_einsummable_info_t>
  make_einsummable_info(string str, vector<tensor_t> const& inns);
};


