#pragma once

#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h"

#include <cuda_runtime.h>
#include <cutensor.h>
#include <library_types.h>

#include "utility.h"

using cutensor_kernel_t = std::function<
    void(cudaStream_t, cutensorHandle_t, void*, vector<void const*>, void*, uint64_t)
  >;

using cutensor_elementwise_kernel_t = std::function<
  void(cudaStream_t, cutensorHandle_t, void*, vector<void const*>)
>;

using cuda_kernel_t = std::function<
  void(cudaStream_t, float*, float const*)
>;

using void_cuda_kernel_t = std::function<
  void(cudaStream_t, void*, void const*)
>;

//using cutensor_scalarop_t = scalar_ns::cutensor_scalarop_t;

void launch_touch_kernel(
  touch_t const& touch,
  cudaStream_t stream,
  void* out,
  void const* inn);

// return a function to execute every einsummable; throw an error
// if (1) the einsummable is a contraction or (2) no implementation
// exists.
//cutensor_kernel_t
//build_einsummable(einsummable_t const& einsummable);

// cutensor_kernel_t
// build_cutensor_reduction(
//   vector<int> inn_modes, vector<uint64_t> inn_shape,
//   vector<int> out_modes, vector<uint64_t> out_shape,
//   castable_t castable,dtype_t type);

uint64_t reduction_worksize(
  einsummable_t einsummable, void* out,
  vector<void const*> inns,
  cutensorHandle_t const* handle);

struct cutensor_elementwise_op_t {
  struct arg_t {
    scalar_t scale;
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

cutensor_elementwise_kernel_t
cutensor_silu_elementwise(uint64_t size);

cutensor_elementwise_kernel_t
build_cutensor_type_conversion(einsummable_t const& e);

cutensor_kernel_t
build_elementwise_and_pow(cutensor_elementwise_op_t op, uint64_t a_size);

cutensor_elementwise_op_t make_mul_op(
  einsummable_t const& e);

cutensorDataType_t dtype_to_cudatype(dtype_t type);

cutensorComputeDescriptor_t dtype_to_computetype(dtype_t type);

cudaDataType_t dtype_to_elementwise_computetype(dtype_t type);

void increment_in_place(
  scalar_t const& scale, 
  cudaStream_t stream, 
  void* out_mem, 
  uint64_t out_elem);

