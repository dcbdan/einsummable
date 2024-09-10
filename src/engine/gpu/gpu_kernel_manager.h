#pragma once
#include "../../base/setup.h"

#include "../../einsummable/simplescalarop.h"

#include "kernels.h"

#include "cuda_kernels.h"
#include "../../einsummable/taskgraph.h" // touch_t

#include <cstdint>
#include <sys/types.h>
#include <thread>

#include <cuda_runtime.h>
#include <cutensor.h>

struct workspace_info_t {
  workspace_info_t() {}
  workspace_info_t(uint64_t sz): workspace_size(sz) {}

  optional<uint64_t> workspace_size;

  uint64_t const& value() const { return workspace_size.value(); }

  bool known() const { return bool(workspace_size); }
};

struct kernel_manager_t {
private:
  struct matmul_t {
    // Str = swap   ? R,L->ik : L,R->ij
    // L   = transL ? ji      : ij
    // R   = transR ? kj      : jk
    dtype_t dtype;
    uint64_t ni;
    uint64_t nj;
    uint64_t nk;
    bool trans_l;
    bool trans_r;
    bool swap;
  };
  optional<matmul_t> make_matmul(einsummable_t const& e);

  void execute_matmul(
    matmul_t const& m, 
    cudaStream_t stream,
    void* out,
    void const* lhs,
    void const* rhs) const;

  struct contraction_t {
    cutensorTensorDescriptor_t descA;
    cutensorTensorDescriptor_t descB;
    cutensorTensorDescriptor_t descC = nullptr;
    cutensorOperationDescriptor_t desc;
    cutensorPlan_t plan;
    dtype_t dtype;
    uint64_t worksize;
  };

  contraction_t make_contraction(einsummable_t const& e);

  void execute_contraction(
    contraction_t const& c,
    cudaStream_t stream,
    void* out,
    void const* lhs,
    void const* rhs,
    void* work,
    uint64_t given_worksize) const;

  struct reduction_t {
    dtype_t dtype;
    uint64_t worksize;
    cutensorPlan_t plan;
  };

  reduction_t make_reduction(einsummable_t const& e);
  reduction_t make_reduction_negate(einsummable_t const& e);

  void execute_reduction(
    reduction_t const& r,
    cudaStream_t stream,
    void* out,
    vector<void const*> inns,
    void* work,
    uint64_t given_worksize) const;

  struct elementwise_t {
    list_simple_scalarop_t sops;
    vector<cutensorPlan_t> plans;    
    vector<uint64_t> join_shape;
    vector<vector<int>> inns;
    int out_rank;

    vector<uint64_t> out_shape(int which) const {
      // always the same, regardless of which sop is being used
      return vector<uint64_t>(join_shape.begin(), join_shape.begin() + out_rank);
    }

    vector<int> get_inn_idxs_at(int arg) const {
      if(arg < 0) {
        return vector_iota<int>(out_rank);
      } else {
        return inns[arg];
      }
    }
  };

  uint64_t elementwise_workspace_size(elementwise_t const& e) const;

  vector<cutensorPlan_t> make_elementwise_plans(
    list_simple_scalarop_t const& sops,
    vector<uint64_t> const& join_shape,
    vector<vector<int>> const& inns,
    int out_rank) const;

  cutensorPlan_t sop_scale_plan(
    simple_scalarop_t::scale_t const& op,
    vector<int> const& inn_idxs,
    vector<uint64_t> const& out_shape) const;

  cutensorPlan_t sop_scale_add_plan(
    scalar_t const& scale,
    vector<int> const& inn_idxs_,
    vector<uint64_t> const& out_shape_) const;

  cutensorPlan_t sop_scale_mul_plan(
    scalar_t const& scale,
    vector<int> const& inn_idxs,
    vector<uint64_t> const& out_shape) const;

  cutensorPlan_t sop_unary_plan(
    simple_scalarop_t::unary_t const& op,
    vector<int> const& inn_idxs,
    vector<uint64_t> const& out_shape) const;

  cutensorPlan_t sop_binary_plan(
    simple_scalarop_t::binary_t const& op,
    vector<int> const& lhs_idxs,
    vector<int> const& rhs_idxs,
    vector<uint64_t> const& out_shape) const;   

  cutensorPlan_t sop_binary_plan_different_shape(
    simple_scalarop_t::binary_t const& op,
    vector<int> const& lhs_idxs,
    vector<int> const& rhs_idxs,
    vector<uint64_t> const& out_shape) const;                  

  void execute_sop_scale(
    simple_scalarop_t::scale_t const& op,
    uint64_t out_elem,
    cudaStream_t stream,
    void* out_mem,
    void const* inn_mem,
    cutensorPlan_t plan) const;

  void execute_sop_scale_add(
    scalar_t const& scale,
    uint64_t out_elem,
    cudaStream_t stream,
    void* out_mem,
    void const* inn_mem,
    cutensorPlan_t plan) const;

  void execute_sop_scale_mul(
    scalar_t const& scale,
    cudaStream_t stream,
    void* out_mem,
    void const* inn_mem,
    cutensorPlan_t plan) const;

  void execute_sop_unary(
    simple_scalarop_t::unary_t const& op,
    cudaStream_t stream,
    void* out_mem,
    void const* inn_mem,
    cutensorPlan_t plan) const;

  void execute_sop_binary(
    simple_scalarop_t::binary_t const& op,
    cudaStream_t stream,
    void* out_mem,
    void const* lhs_mem,
    void const* rhs_mem,
    cutensorPlan_t plan,
    bool swapped) const;

  void execute_sop_binary_different_shape(
    simple_scalarop_t::binary_t const& op,
    cudaStream_t stream,
    void* out_mem,
    void const* lhs_mem,
    void const* rhs_mem,
    cutensorPlan_t plan,
    bool swapped) const;

  void execute_elementwise(
    elementwise_t const& op,
    cudaStream_t stream,
    void* out,
    vector<void const*> inns,
    void* work_mem,
    uint64_t given_worksize) const;

  // special kernel struct here
  struct type_conversion_t{
    cutensor_elementwise_kernel_t kernel;
  };

  struct power_t{
    double power;
    cuda_kernel_t kernel;
  };

  struct scale_t{
    float scale;
    cuda_kernel_t kernel;
  }; 

  struct pow_and_elementwise_t{
    cutensor_kernel_t kernel;
    uint64_t worksize;
    uint64_t a_size;
  };

  struct custom_kernel_1_t{
    cutensor_elementwise_kernel_t kernel;
  };

  struct custom_kernel_2_t{
    uint64_t nrows;
    uint64_t ncols;
    dtype_t dtype;
  };

  struct custom_kernel_4_t{
    // three steps:
    // 1)rewrite the scalarop to an elementwise we can compile
    // 2)compile the elementwise
    // 3)compile the reduction
    elementwise_t elementwise;
    reduction_t reduction;
    // we need 3 workspace: 1. for elementwise 2. for output of elementwise 3.reduction
    uint64_t worksize;
    uint64_t elementwise_output_offset;
    uint64_t reduction_offset;
  };

  struct special_max_reduction_t{
    uint64_t a;
    uint64_t b;
  };

  struct special_sum_reduction_t{
    uint64_t a;
    uint64_t b;
  };

  struct special_negateSum_reduction_t{
    uint64_t a;
    uint64_t b;
  };

  struct v3_softmax_reduction_t{
    uint64_t a;
    uint64_t b;
    float constant;
  };

  struct v3_softmax_elementwise_t{
    uint64_t a;
    uint64_t b;
    float constant;
  };

  struct large_workspace_1_t{
    uint64_t a;
    uint64_t b;
  };

  struct large_workspace_2_t{
    uint64_t a;
    uint64_t b;
  };

  struct large_workspace_3_t{
    uint64_t a;
    uint64_t b;
  };

  struct large_workspace_4_t{
    uint64_t a;
    float constant1;
    float constant2;
  };

  struct special_contraction_t{
    elementwise_t permute;
    matmul_t matmul;
    uint64_t intermediate_size;
  };

public:
  using kernel_info_t = std::variant<matmul_t, contraction_t, reduction_t, elementwise_t,
                                      type_conversion_t, pow_and_elementwise_t, custom_kernel_1_t,
                                      power_t, scale_t, 
                                      custom_kernel_2_t, custom_kernel_4_t,
                                      special_max_reduction_t,
                                      special_sum_reduction_t,
                                      special_negateSum_reduction_t,
                                      v3_softmax_reduction_t,
                                      v3_softmax_elementwise_t,
                                      large_workspace_1_t,
                                      large_workspace_2_t,
                                      large_workspace_4_t,
                                      special_contraction_t>;

  kernel_manager_t();
  kernel_manager_t(int device);
  ~kernel_manager_t();

  // special kernel identification here
  static bool is_power_elementwise(einsummable_t e);
  static bool is_type_conversion(einsummable_t e);
  static bool is_elementwise_with_pow(einsummable_t e);
  static bool is_custom_kernel1(einsummable_t e);
  static bool is_custom_kernel2(einsummable_t e);
  static bool is_custom_kernel3(einsummable_t e);
  static bool is_custom_kernel4(einsummable_t e);
  static bool is_custom_kernel5(einsummable_t e);
  static bool is_large_workspace1(einsummable_t e);
  static bool is_large_workspace2(einsummable_t e);
  static bool is_large_workspace3(einsummable_t e);
  static bool is_large_workspace4(einsummable_t e);
  static bool is_softmax_v3_reduction(einsummable_t e);
  static bool is_softmax_v3_elementwise(einsummable_t e);
  static bool is_special_max_reduction(einsummable_t e);
  static bool is_special_sum_reduction(einsummable_t e);
  //+ adbe,cde->abc | *[hole|f32@0,hole|f32@1]
  static bool is_special_contraction(einsummable_t e);
  static bool is_c64_elementwise_multiply(einsummable_t e);
  static double get_power(einsummable_t e);
  static bool is_scale_and_increment(einsummable_t e);
  static tuple<float, float> get_increment_scale(einsummable_t e);

  custom_kernel_4_t build_custom_kernel4(einsummable_t const& e);

  optional<workspace_info_t> build(einsummable_t const& e);

  workspace_info_t workspace_size(einsummable_t const& e) const;

  void operator()(
    touch_t const& touch,
    cudaStream_t stream,
    void* out,
    void const* inn) const;

  void operator()(
    einsummable_t const& e,
    cudaStream_t stream,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

  void operator()(
    kernel_info_t const& k,
    cudaStream_t stream,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

  void operator()(
    list_simple_scalarop_t const& sop,
    cudaStream_t stream,
    void* out,
    void const* inns,
    optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

  void lowerTri_fill(fill_t::lowertri_t const& l, cudaStream_t stream, void* out) const;
  void constant_fill(fill_t::constant_t const& c, cudaStream_t stream, void* out) const;

  kernel_info_t const& 
  get_built_kernel_info(einsummable_t const& e) const;

private:
  workspace_info_t workspace_size(kernel_info_t const& kernel) const;

private:
  std::unordered_map<einsummable_t, kernel_info_t> kernels;
  cutensorHandle_t cutensor_handle;
  cublasHandle_t cublas_handle; 

  float16_t           one_half;
  float               one_float;
  double              one_double;
  std::complex<float> one_complex;

  float16_t           zero_half;
  float               zero_float;
  double              zero_double;
  std::complex<float> zero_complex;

  void const* get_one_ptr(dtype_t dtype) const;
  void const* get_zero_ptr(dtype_t dtype) const;

  int device;
};

