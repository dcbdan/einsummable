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

cutensor_kernel_t
build_cutensor_reduction(
  vector<int> inn_modes, vector<uint64_t> inn_shape,
  vector<int> out_modes, vector<uint64_t> out_shape);

// create a canned ij->i kernel that does not use
// cutensor. (cutensor reduction can only do this
// for the add castable)
cutensor_kernel_t
build_simple_reduction(
  uint64_t ni, uint64_t nj,
  castable_t castable);

struct cutensor_elementwise_op_t {
  struct arg_t {
    float scale;
    cutensorOperator_t op;
    vector<int> modes;
    // ^ modes is analagous to einsummable_t::inns[i]
  };
  struct unary_t {
    // Out{0,1,..,rank-1} = arg.scale * arg.op( Inn{arg.modes} )
    arg_t arg;
  };
  struct binary_t {
    // Out{0,1,...,rank-1} = op(
    //   lhs.scale * lhs.op( Lhs{lhs.modes} ),
    //   rhs.scale * rhs.op( Rhs{rhs.modes} ),
    cutensorOperator_t op;
    arg_t lhs;
    arg_t rhs;
  };
  struct ternary_t {
    // Out{0,1,...,rank-1} = op_01_2(
    //   op_0_1(
    //     a0.scale * a0.op( A0{a0.modes} ),
    //     a1.scale * a1.op( A1{a1.modes} )),
    //   a2.scale * a2.op( A2{a2.modes} )
    // );
    cutensorOperator_t op_01_2;
    cutensorOperator_t op_0_1;
    arg_t a0;
    arg_t a1;
    arg_t a2;
  };

  vector<uint64_t> join_shape;
  std::variant<unary_t, binary_t, ternary_t> op;
};

cutensor_kernel_t
build_cutensor_elementwise(cutensor_elementwise_op_t op);

// Attempt to construct a cutensor elementwise op
// from an einsummable. If the einsummable can't be
// converted, return none
optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e);

// Straight elementwise means:
//   for(int i = 0; i != size; ++i) {
//     out[i] = op(inn0[i], ..., innN[i]);
//   }
cutensor_kernel_t
build_straight_elementwise(
  scalarop_t op,
  uint64_t size);