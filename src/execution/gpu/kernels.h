#pragma once

#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h"

#include <cuda_runtime.h>
#include <cutensor.h>

using touch_kernel_t = std::function<
    void(cudaStream_t, float*, float const*)
  >;
using cutensor_kernel_t = std::function<
    void(cudaStream_t, cutensorHandle_t const*, float*, vector<float const*>)
  >;

touch_kernel_t build_touch(touch_t const& touch);

// return a function to execute every einsummable; throw an error
// if (1) the einsummable is a contraction or (2) no implementation
// exists.
cutensor_kernel_t
build_einsummable(einsummable_t const& einsummable);

// build a cutensor contraction description; throw an error if
// the einsummable is not a contraction
void build_contraction(
  cutensorContractionDescriptor_t* desc,
  einsummable_t const& einsummable);

void execute_contraction(
  cudaStream_t,
  cutensorHandle_t const*,
  cutensorContractionDescriptor_t const*,
  float* out,
  float const* lhs,
  float const* rhs);

bool is_contraction(einsummable_t const& e);

cutensor_kernel_t
build_reduction(
  castable_t castable,
  vector<uint64_t> shape,
  int out_rank);

struct cutensor_elementwise_op_t {
  struct arg_t {
    float scale;
    cutensorOperator_t op;
    vector<int> modes;
  };
  struct unary_t {
    arg_t arg;
  };
  struct binary_t {
    cutensorOperator_t op;
    arg_t lhs;
    arg_t rhs;
  };
  struct ternary_t {
    cutensorOperator_t op_01_2;
    cutensorOperator_t op_0_1;
    arg_t a0;
    arg_t a1;
    arg_t a2;
  };
  std::variant<unary_t, binary_t, ternary_t> op;
};

cutensor_kernel_t
build_cutensor_elementwise(cutensor_elementwise_op_t op);

// Attempt to construct a cutensor elementwise op
// from an einsummable. If the einsummable can't be
// converted, return None
optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e);

cutensor_kernel_t
build_straight_elementwise(einsummable_t const& e);
