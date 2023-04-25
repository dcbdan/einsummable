#pragma once
#include "../einsummable/setup.h"

#include "../einsummable/graph.h"

// A matgraph is a graph where every tensor is a matrix.
// It supports backpropagation to compute gradients.
struct matgraph_t {
  // insert elementwise and elementwise binary ops
  int insert_ew(scalar_join_t op, int inn);
  int insert_ewb(scalar_join_t op, int lhs, int rhs);

  // insert matrixm multiply ops. Here,
  // t stands for transpose and s stands for not-transpose.
  int insert_matmul_ss(int lhs, int rhs); // ij,jk->ik
  int insert_matmul_ts(int lhs, int rhs); // ji,jk->ik
  int insert_matmul_st(int lhs, int rhs); // ij,kj->ik
  int insert_matmul_tt(int lhs, int rhs); // ji,kj->ik

  // insert inputs or ones of a given shape
  int insert_input(uint64_t d0, uint64_t d1);
  int insert_ones(uint64_t d0, uint64_t d1);

  // Compute the gradients of each identifier in weights
  // with respect to the sum of all elements in
  // the out matrix.
  //
  // As output, return the identifier containing
  // each gradient.
  vector<int> backprop(int out, vector<int> weights);

private:
  struct matmul_t {
    bool t_lhs;
    int lhs;
    bool t_rhs;
    int rhs;
  };

  struct ew_t {
    scalar_join_t op;
    int inn;
  };

  struct ewb_t {
    scalar_join_t op;
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

    std::optional<int> inn0() const;
    std::optional<int> inn1() const;
    vector<int> inns() const;
  };
  vector<node_t> nodes;

  int insert(op_t op, tuple<uint64_t, uint64_t> out_shape);
};

// Build a graph object from the matgraph
graph_t compile(matgraph_t const& matgraph);



