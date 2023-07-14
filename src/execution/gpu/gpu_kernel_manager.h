#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include "kernel.h"

#include <thread>


struct contraction_t{
public:
  uint64_t worksize;
  cutensorContractionDescriptor_t desc;
  dtype_t dtype;
private:
  static contraction_t make(einsummable_t const& e_);
}



struct kernel_manager_t{
private:

struct reduction_t{
  uint64_t worksize;
  cutensor_kernel_t kernel;
}

struct power_t{
  double power;
  cuda_kernel_t kernel;
}

struct type_conversion_t{
  cutensor_elementwise_kernel_t kernel;
}

struct elementwise_t{
  cutensor_elementwise_kernel_t kernel;
}


public:

  kernel_manager_t();

  void build(einsummable_t const& e);

  uint64_t workspace_size(einsummable_t const& e) const;

  void operator()(
  touch_t const& touch,
  cudastream_t stream,
  void* out,
  void const* inn) const;

  void operator()(
  einsummable_t const& e,
  cudastream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> workspace = std::nullopt) const;

  using kernel_info_t = std::variant<contraction_t,
  cutensor_kernel_t,
  touch_kernel_t>;

  kernel_info_t const& get_built_kernel_info(einsummable_t const& e) const;


private:

  std::unordered_map<einsummable_t, kernel_info_t> kernels;





}

bool is_power_elementwise(einsummable_t e);