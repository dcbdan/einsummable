#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include "contraction.h"

#include <thread>

struct kernel_manager_t {
  kernel_manager_t();

  // Register this einsummable with the manager and return the required
  // workspace size. If the einsummable object cannot be registered,
  // return None.
  optional<uint64_t> build(einsummable_t const& e);

  // get the workspace size
  // (throw an error if e has not been built)
  uint64_t workspace_size(einsummable_t const& e) const;
  // TODO

  // TODO
  void operator()(
    touch_t const& touch,
    void* out,
    void const* inn) const;

  // TODO
  // If no workspace size is zero, a workspace does not need to be provided.
  // If not enough workspace is provided, an error is thrown.
  // If build was not called for this einsummable, an error may be thrown.
  void operator()(
    einsummable_t const& e,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

private:
  struct binfo_t {
    bool trans_lhs;
    bool trans_rhs;
    bool batched_out;
    bool batched_lhs;
    bool batched_rhs;
  };
  struct batch_matmul_t {
    dtype_t dtype;
    binfo_t info;
    uint64_t nb;
    uint64_t ni;
    uint64_t nj;
    uint64_t nk;
  };

  struct unary_straight_ew_t {
    uint64_t n;
    vector<uint8_t> data;
    void (*f)(uint8_t const*, uint64_t, void*, void const*);
  };

  struct binary_straight_ew_t {
    uint64_t n;
    vector<uint8_t> data;
    void (*f)(uint8_t const*, uint64_t, void*, void const*, void const*);
  };

  struct binary_212_ew_t {
    uint64_t na;
    uint64_t nb;
    vector<uint8_t> data;
    void (*f)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*);
  };

  struct tensor_permute_t {
    dtype_t dtype;
    vector<uint64_t> inn_shape;
    vector<int> out_perm;
  };

  struct reduction_ab_a_t {
    uint64_t na;
    uint64_t nb;
    std::function<void(uint64_t, uint64_t, void*, void const*)> f;
  };

  // kernel_t is a misc catchall that can just wrap a lambda
  using kernel_t = std::function<void(void*, vector<void const*>)>;

  using kernel_info_t = std::variant<
    batch_matmul_t, contraction_t,
    unary_straight_ew_t, binary_straight_ew_t, binary_212_ew_t, tensor_permute_t,
    reduction_ab_a_t,
    kernel_t>;

  std::unordered_map<einsummable_t, kernel_info_t> kernels;

  std::map<string, binfo_t> binfos;

  optional<batch_matmul_t>
  make_batch_matmul(einsummable_t const& e);
};

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, void*, void const*)> >
lookup_unary_straight_ew_kernel(scalarop_t op);

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, void*, void const*, void const*)> >
lookup_binary_straight_ew_kernel(
  scalarop_t binary_op);

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*)> >
lookup_binary_212_ew_kernel(
  scalarop_t binary_op,
  bool is_ab_a);

std::function<void(uint64_t, uint64_t, void*, void const*)>
build_ab_a_reduction_kernel(
  dtype_t dtype,
  castable_t castable);

using touch_kernel_t = std::function<void(void*, void const*)>;

touch_kernel_t
build_touch(touch_t const& touch);

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

void permute_kernel(
  dtype_t dtype,
  uint64_t permute_block_size,
  vector<uint64_t> const& inn_shape,
  vector<int> const& out_perm,
  void* out,
  void const* inn);



