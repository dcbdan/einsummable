#pragma once
#include "../einsummable/setup.h"

#include "../einsummable/graph.h"

// A matgraph is a graph where every tensor is a matrix.
// It supports backpropagation to compute gradients.
struct matgraph_t {
  // insert elementwise and elementwise binary ops
  int insert_ew(scalarop_t op, int inn);
  int insert_ewb(scalarop_t op, int lhs, int rhs);

  // insert matrixm multiply ops. Here,
  // t stands for transpose and s stands for not-transpose.
  int insert_matmul_ss(int lhs, int rhs); // ij,jk->ik
  int insert_matmul_ts(int lhs, int rhs); // ji,jk->ik
  int insert_matmul_st(int lhs, int rhs); // ij,kj->ik
  int insert_matmul_tt(int lhs, int rhs); // ji,kj->ik

  // insert inputs or ones of a given shape
  int insert_input(uint64_t d0, uint64_t d1);
  int insert_ones(uint64_t d0, uint64_t d1);

  // For a binary tree of additions. If zero or one inputs
  // are given, throw an error: this method must create
  // a new operation.
  int insert_adds(vector<int> inns);

  // Compute the gradients of each identifier in weights
  // with respect to the sum of all elements in
  // the out matrix.
  //
  // As output, return the identifier containing
  // each gradient.
  vector<int> backprop(int out, vector<int> weights);

  // Build a graph object from the matgraph.
  // All output nodes are save nodes.
  // Return 1. thre resulting graph and 2. a map
  // from (included matgraph ids) to (graph ids).
  tuple<graph_t, map<int, int>>
  compile() const;
  // Every id in save_ids is saved.
  // This will automatically prune parts of the
  // graph that doesn't live in a path from input
  // to save_id.
  tuple<graph_t, map<int, int>>
  compile(vector<int> const& save_ids) const;

  void print() const;

private:
  struct matmul_t {
    bool t_lhs;
    int lhs;
    bool t_rhs;
    int rhs;
  };

  struct ew_t {
    scalarop_t op;
    int inn;
  };

  struct ewb_t {
    scalarop_t op;
    int lhs;
    int rhs;
  };

  struct input_t {
  };

  struct ones_t {
  };


  using op_t = std::variant<matmul_t, ew_t, ewb_t, input_t, ones_t>;

  struct node_t {
    tuple<uint64_t, uint64_t> out_shape;
    op_t op;
    set<int> outs;

    optional<int> inn0() const;
    optional<int> inn1() const;
    vector<int> inns() const;
    set<int> inns_set() const;
    void print() const;

    inline bool is_einsummable() const;

    inline bool is_matmul()      const;
    inline bool is_ew()          const;
    inline bool is_ewb()         const;
    inline bool is_input()       const;
    inline bool is_ones()        const;
  };
  vector<node_t> nodes;

  int insert(op_t op, tuple<uint64_t, uint64_t> out_shape);

  // Compute all nodes that exist on a path from some input to some output,
  // For inn -> x0 -> x1 -> x2 -> out,
  //   put x0, x1, x2 into the output.
  // For inn0 -> inn1 -> x1 -> x2 -> out1 -> out0,
  //   put inn1, x1, x2, out1 into the output.
  // If true, include all inns and outs
  set<int> compute_nodeset(
    vector<int> const& outs,
    vector<int> const& inns,
    bool include_inns_outs) const;

  struct backprop_state_t {
    // get the gradient of this id
    int operator[](int id);

    // Add the initial ones at out_id
    void start(int out_id);

    map<int, int> grads;
    matgraph_t& self;
    set<int> nodeset;

    struct out_edge_t {
      int out;
      int which_inn;
    };
    vector<out_edge_t> get_out_edges(int id) const;

  };

  // Gets the gradient term of this node and edge
  // (TODO: this is called something--jacobian? outer jacobian?)
  //
  // Note: this may insert a node into the graph, but it also may not
  int build_grad_term(int node, int which_inn, int node_grad);
  // node_grad has the shape of node,
  // the output grad will have the shape of node's which_inn input

  int build_grad_term_matmul_lhs(matmul_t const& matmul, int node_grad);
  int build_grad_term_matmul_rhs(matmul_t const& matmul, int node_grad);
  int build_grad_term_ewb_lhs(ewb_t const& ewb, int node_grad);
  int build_grad_term_ewb_rhs(ewb_t const& ewb, int node_grad);
  int build_grad_term_ew_inn(ew_t const& ew, int node_grad);

  // Return the einsummable and the inputs. Note that the inputs
  // returned may be shorter than the node inputs if ones are absorbed.
  //
  // Node must be einsummable
  tuple<einsummable_t, vector<int>> translate_node(node_t const& node) const;
};

