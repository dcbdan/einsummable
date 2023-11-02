#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"
#include "../../einsummable/einsummable.h"

#include "../touch.h"

#include "contraction.h"

#include <thread>

struct cpu_kernel_executor_t {
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

  struct broadcast_b_ab_t {
    uint64_t sz_a;
    uint64_t sz_b;
  };

  // kernel_t is a misc catchall that can just wrap a lambda
  using kernel_t = std::function<void(void*, vector<void const*>)>;

public:

  cpu_kernel_executor_t();

  // Register this einsummable with the manager and return the required
  // workspace size. If the einsummable object cannot be registered,
  // return None.
  optional<uint64_t> build(einsummable_t const& e);

  // get the workspace size
  // (throw an error if e has not been built)
  uint64_t workspace_size(einsummable_t const& e) const;

  string as_str(einsummable_t const& e) const;

  // Return the inputs of e that may be donated,
  //   so if e  is out = inn0 + inn1, this might return {0,1}
  //   so that either
  //     inn0 = inn0 + inn1  or
  //     inn1 = inn0 + inn1
  // (throw an error if e has not been built)
  vector<int> donatables(einsummable_t const& e) const;

  void operator()(
    touch_t const& touch,
    void* out,
    void const* inn) const;

  // If no workspace size is zero, a workspace does not need to be provided.
  // If not enough workspace is provided, an error is thrown.
  // If build was not called for this einsummable, an error may be thrown.
  void operator()(
    einsummable_t const& e,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

  using kernel_info_t = std::variant<
    batch_matmul_t, contraction_t,
    unary_straight_ew_t, binary_straight_ew_t, binary_212_ew_t, tensor_permute_t,
    reduction_ab_a_t, broadcast_b_ab_t,
    kernel_t>;

  kernel_info_t const& get_built_kernel_info(einsummable_t const& e) const;

  static void call(
    kernel_info_t const& info,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt);

private:

  std::unordered_map<einsummable_t, kernel_info_t> kernels;

  // This is just a map from einsummable strs to the corresponding
  // batch matmul settings
  std::map<string, binfo_t> binfos;

  optional<batch_matmul_t>
  make_batch_matmul(einsummable_t const& e);
};

// This is just a function to create a standalone kernel
// that does not require a workspace
std::function<void(void*, vector<void const*>)>
build_einsummable(einsummable_t const& e);

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

void broadcast_b_ab_kernel(
  uint64_t sz_a,
  uint64_t sz_b,
  void* out,
  void const* inn);
