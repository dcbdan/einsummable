#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include <thread>

using kernel_t = std::function<void(void*, vector<void const*>)>;
using touch_kernel_t = std::function<void(void*, void const*)>;

kernel_t
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t unary_op);

kernel_t
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t binary_op);

kernel_t
build_binary_212_elementwise_kernel(
  int num_threads,
  uint64_t na,
  uint64_t nb,
  scalarop_t binary_op);

kernel_t
build_ab_a_reduction(
  int num_thread,
  uint64_t na,
  uint64_t nb,
  dtype_t dtype,
  castable_t castable);

// Note: This could be multithreaded.
//       That should be straightforward to do:
//       Partition the touch operation into the
//       number of threads and call each of
//       those touches in parallel.
touch_kernel_t
build_touch(touch_t const& touch);

kernel_t
build_einsummable(
  int num_threads, // passed to build_*_elementwise_kernel
  einsummable_t const& einsummable);

// trans lhs   trans rhs
// F           F          ij,jk->ik
// T           F          ji,jk->ik
// F           T          ji,jk->ik
// T           T          ji,kj->ik
void matrix_multiply_update(
  dtype_t const& dtype,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs,
  bool is_zero_else_one);

void matrix_multiply(
  dtype_t const& dtype,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs);

// b<ij> , b<jk> -> b<ik>
//
// This kernel includes things like
//   bij,jk->ik
//   ji,bjk->bik
//   ij,jk->bik
//   bij,bjk->ik
// by just looping over the batched dimension
void batch_matrix_multiply(
  dtype_t const& dtype,
  uint64_t const& nb,
  bool const& batched_out,
  bool const& batched_lhs,
  bool const& batched_rhs,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs);

void c64_mul_abcd_bd_to_abcd(
  uint64_t na,
  uint64_t nb,
  uint64_t nc,
  uint64_t nd,
  std::complex<float>* out,
  std::complex<float> const* lhs,
  std::complex<float> const* rhs);
