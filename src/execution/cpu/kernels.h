#pragma once
#include "../../einsummable/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include <thread>

void print_elementwise_function(scalarop_t op);

std::function<void(float*,vector<float const*>)>
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t unary_op);

std::function<void(float*,vector<float const*>)>
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t binary_op);

// TODO: Does this guy need to be multithreaded?
//       That should be straightforward to do:
//       Partition the touch operation into the
//       number of threads and call each of
//       those touches in parallel.
std::function<void(float*, float const*)>
build_touch(touch_t const& touch);

std::function<void(float*, vector<float const*>)>
build_einsummable(
  int num_threads, // passed to build_*_elementwise_kernel
  einsummable_t const& einsummable);

// trans lhs   trans rhs
// F           F          ij,jk->ik
// T           F          ji,jk->ik
// F           T          ji,jk->ik
// T           T          ji,kj->ik
void matrix_multiply_update(
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs,
  float const& beta);

void matrix_multiply(
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs);

// b<ij> , b<jk> -> b<ik>
//
// This kernel includes things like
//   bij,jk->ik
//   ji,bjk->bik
//   ij,jk->bik
//   bij,bjk->ik
// by just looping over the batched dimension
void broadcast_matrix_multiply(
  uint64_t const& nb,
  bool const& batched_out,
  bool const& batched_lhs,
  bool const& batched_rhs,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs);


