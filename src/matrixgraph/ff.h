#pragma once
#include "matrixgraph.h"

struct ff_sqdiff_t {
  matrixgraph_t mgraph;

  uint64_t dn;
  uint64_t dp;
  uint64_t dd;
  vector<uint64_t> dws;
  float learning_rate;

  int x;
  int y;
  int yhat;
  int sqdiff;
  vector<int> wsinn;
  vector<int> wsout;
  vector<int> grads;

  using shape_t = tuple<uint64_t, uint64_t>;

  shape_t shape_x()       const { return {dn,dp}; }
  shape_t shape_y()       const { return {dn,dd}; }
  shape_t shape_yhat()    const { return {dn,dd}; }
  shape_t shape_sqdiff()  const { return {dn,dd}; }
  shape_t shape_wi(int i) const;
};

// Given data matrix x: dn,dp
// and output matrix y: dn,dd,
// create a ff gradient update where
//   W0: dp,dws[0]
//   W1: dws[0],dws[1]
//   ...
//   Wn: dws[-1],dd
// Yhat = relu( ... relu(relu(X W0) W1) ... Wn-1) Wn
// Then take the gradient with respect to
//   sum(square(Yhat-Y))
// and update each W0, ..., Wn.
//
// Note: if dws = {}, the model is linear regression.
ff_sqdiff_t
ff_sqdiff_update(
  uint64_t dn, uint64_t dp, uint64_t dd,
  vector<uint64_t> dws,
  float learning_rate);
