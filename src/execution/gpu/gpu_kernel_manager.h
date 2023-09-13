#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include "kernels.h"

#include "cuda_kernels.h"

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

struct kernel_manager_t{
private:
  struct contraction_t {
    cutensorContractionDescriptor_t desc;
    cutensorContractionFind_t find;
    cutensorContractionPlan_t plan;
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

  struct reduction_t{
    cutensor_kernel_t kernel;
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

  struct type_conversion_t{
    cutensor_elementwise_kernel_t kernel;
  };

  struct elementwise_t{
    cutensor_elementwise_kernel_t kernel;
  };

  struct custom_kernel_1_t{
    cutensor_elementwise_kernel_t kernel;
  };

public:
  kernel_manager_t();

  optional<workspace_info_t> build(einsummable_t const& e);

  workspace_info_t workspace_size(einsummable_t const& e) const;

  uint64_t known_workspace_size(
    einsummable_t const& e, 
    void* out, 
    vector<void const*> inns) const;

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

private:
  using kernel_info_t = std::variant<
    contraction_t, cutensor_kernel_t, scale_t, pow_and_elementwise_t,
    custom_kernel_1_t, void_cuda_kernel_t, type_conversion_t,
    touch_kernel_t, elementwise_t, power_t, reduction_t>;

  kernel_info_t const& 
  get_built_kernel_info(einsummable_t const& e) const;

  workspace_info_t workspace_size(kernel_info_t const& kernel) const;

  void call(
    kernel_info_t const& kernel,
    cudaStream_t stream,
    void* out,
    vector<void const*> inns,
    optional<tuple<void*, uint64_t>> maybe_workspace) const;

  static bool is_power_elementwise(einsummable_t e);

  static bool is_type_conversion(einsummable_t e);

  static bool is_elementwise_with_pow(einsummable_t e);

  static bool is_custom_kernel1(einsummable_t e);

  static bool is_c64_elementwise_multiply(einsummable_t e);

  static double get_power(einsummable_t e);

  static bool is_scale_and_increment(einsummable_t e);

  static tuple<float, float> get_increment_scale(einsummable_t e);
private:
  std::unordered_map<einsummable_t, kernel_info_t> kernels;
  cutensorHandle_t* handle;
};

