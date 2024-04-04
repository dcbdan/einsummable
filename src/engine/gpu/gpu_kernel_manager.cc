#include "gpu_kernel_manager.h"

#include "kernels.h"
#include "utility.h"
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

kernel_manager_t::kernel_manager_t() 
  : kernel_manager_t(0)
{
  DOUT("!!! Note: Creating kernel manager without a device id !!!")
}

kernel_manager_t::kernel_manager_t(int device): device(device) {
  // DOUT("Creating kernel manager on device " << device);
  cudaSetDevice(device);
  handle_cutensor_error(
    cutensorCreate(&cutensor_handle),
    "cutensor create in kernel_manager constructor");
  handle_cublas_error(
    cublasCreate(&cublas_handle),
    "cublas create in kernel_manager constructor");

  one_half    = 1.0f;
  one_float   = 1.0f;
  one_double  = 1.0;
  one_complex = std::complex<float>(1.0f, 0.0f);

  zero_half    = 0.0f;
  zero_float   = 0.0f;
  zero_double  = 0.0;
  zero_complex = std::complex<float>(0.0f, 0.0f);
}

kernel_manager_t::~kernel_manager_t() {
  cudaSetDevice(device);
  handle_cutensor_error(
    cutensorDestroy(cutensor_handle),
    "cutensor destroy in kernel_manager destructor");
  handle_cublas_error(
    cublasDestroy(cublas_handle),
    "cublas destroy in kernel_manager destructor");
}

// we translate simple scalar ops' operators to cutensor operators
static cutensorOperator_t uop_to_cutensorOp(simple_scalarop_t::uop_t uop){
  switch (uop){
    case simple_scalarop_t::uop_t::identity: return CUTENSOR_OP_IDENTITY;
    case simple_scalarop_t::uop_t::neg: return CUTENSOR_OP_NEG;
    case simple_scalarop_t::uop_t::sqrt: return CUTENSOR_OP_SQRT;
    case simple_scalarop_t::uop_t::conj: return CUTENSOR_OP_CONJ;
    case simple_scalarop_t::uop_t::rcp: return CUTENSOR_OP_RCP;
    case simple_scalarop_t::uop_t::sigmoid: return CUTENSOR_OP_SIGMOID;
    case simple_scalarop_t::uop_t::log: return CUTENSOR_OP_LOG;
    case simple_scalarop_t::uop_t::exp: return CUTENSOR_OP_EXP;
    case simple_scalarop_t::uop_t::relu: return CUTENSOR_OP_RELU;
    default: throw std::runtime_error("Unknown uop");
  }
}

static cutensorOperator_t bop_to_cutensorOp(simple_scalarop_t::bop_t bop){
  switch (bop){
    case simple_scalarop_t::bop_t::add: return CUTENSOR_OP_ADD;
    case simple_scalarop_t::bop_t::mul: return CUTENSOR_OP_MUL;
    case simple_scalarop_t::bop_t::min: return CUTENSOR_OP_MIN;
    case simple_scalarop_t::bop_t::max: return CUTENSOR_OP_MAX;
    default: throw std::runtime_error("Unknown bop");
  }
}

// ----- special case kernels -----

bool kernel_manager_t::is_type_conversion(einsummable_t e){
  scalarop_t op = e.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  if(op_str=="float16_t(x0)"||op_str=="float(x0)"||op_str=="double(x0)"){
    // DOUT("Found type conversion: " << e);
    return true;
  }
  return false;
}

bool kernel_manager_t::is_power_elementwise(einsummable_t e){
  if(e.inns.size()==1){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();

    size_t endIndex = op_str.find(")");


    if(op_str.substr(0,8)=="_pow(x0,"&&endIndex==op_str.size() - 1){
      // DOUT("Found power elementwise: " << e);
      return true;
    }
  }
  return false;
}

bool kernel_manager_t::is_scale_and_increment(einsummable_t e){
  if(e.inns.size()==1){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();

    size_t endIndex = op_str.find("*x0))");

    if(op_str.substr(0,16)=="(f32|1e-05+(f32|"&&endIndex==op_str.size() - 5){
      // DOUT("Found scale and increment: " << e);
      return true;
    }

    if(op_str.substr(0,16)=="(f32|1e-06+(f32|"&&endIndex==op_str.size() - 5){
      // DOUT("Found scale and increment: " << e);
      return true;
    }
  }
  return false;
}

bool kernel_manager_t::is_elementwise_with_pow(einsummable_t e){
  if(e.inns.size()==2){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();


    if(op_str=="(x0*_pow(x1,-1))"){
      // DOUT("Found elementwise with pow: " << e);
      return true;
    }
  }
  return false;
}

bool kernel_manager_t::is_custom_kernel1(einsummable_t e){
  if(e.inns.size()==1){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();


    if(op_str=="(x0*_pow((f16|1+_exp((f16|-1*x0))),-1))"){
      // DOUT("Found custom kernel 1: " << e);
      return true;
    }
  }
  return false;
}

bool kernel_manager_t::is_c64_elementwise_multiply(einsummable_t e){
   if(e.inns.size()==2){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();


    if(op_str=="(x0*x1)"&&e.inn_dtype(0)==dtype_t::c64&&e.inn_dtype(1)==dtype_t::c64){
      // DOUT("Found c64 elementwise multiply: " << e);
      return true;
    }
  }
  return false;
}

double kernel_manager_t::get_power(einsummable_t e){
  scalarop_t op = e.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  size_t start_index = 8;
  
  size_t end_index = op_str.find(")");

  std::string number_str = op_str.substr(start_index, end_index - start_index);
  
  
  return std::stod(number_str);

}


tuple<float, float> kernel_manager_t::get_increment_scale(einsummable_t e){
  scalarop_t op = e.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  size_t start_index = 16;
  
  size_t end_index = op_str.find("*x0))");

  std::string number_str = op_str.substr(start_index, end_index - start_index);
  
  float increment = 1e-06;

  if(op_str.substr(5,10)=="1e-05"){
    increment = 1e-05;
  }

  return std::make_tuple(std::stof(number_str),increment);
}

// ----- end special case kernels -----

optional<workspace_info_t> 
kernel_manager_t::build(einsummable_t const& e_)
{
  cudaSetDevice(device);
  // DOUT("Building kernel for " << e_);
  auto einsummable = e_.merge_adjacent_dims();

  auto iter = kernels.find(einsummable);
  if(iter != kernels.end()) { 
    return workspace_size(iter->second);
  }

  {
    auto maybe_matmul = make_matmul(einsummable);
    if(maybe_matmul) {
      kernels.insert({einsummable, maybe_matmul.value()});
      return workspace_info_t(0);
    }
  }

  if(einsummable.is_contraction()) {
    auto c = make_contraction(einsummable);
    kernels.insert({einsummable,c});
    return workspace_info_t(c.worksize);
  }

  // Check for Reductions
  if(einsummable.has_aggregation()) {
    if(einsummable.inns.size() != 1) {
      DOUT("Warning: Reduction with more than one input tensor");
      return std::nullopt;
    }

    if (!einsummable.join.is_identity()){
      DOUT("Warning: Reduction with non-identity join");
      return std::nullopt;
    }

    if(einsummable.castable.value() == castable_t::add ||
        einsummable.castable.value() == castable_t::max) 
    {
      // this is something cutensor reduction should be able to do
      vector<int> const& inn_modes = einsummable.inns[0];

      auto inn_shape = einsummable.inn_shapes()[0];

      vector<int> out_modes(einsummable.out_rank);
      std::iota(out_modes.begin(), out_modes.end(), 0);

      auto out_shape = einsummable.out_shape();

      reduction_t reduct = make_reduction(einsummable);

      kernels.insert({einsummable, reduct});

      // we are not returning a workspace size here because we don't know yet
      // see known_workspace_size
      return workspace_info_t(reduct.worksize);
    }

    return std::nullopt;
  }
    // TODO: special case kernels here

  if(is_power_elementwise(einsummable)) {
    // DOUT("Building power kernel");
    double pow = get_power(einsummable);
    uint64_t size = einsummable.join_shape[0];
    auto lambda = [pow, size](cudaStream_t stream, float* out, const float* in) 
    {
      elementwise_power(out, in, stream, pow, size);
    };

    power_t power_kernel {pow, lambda};

    kernels.insert({einsummable, power_kernel});

    return workspace_info_t(0);
  }

  if(is_type_conversion(einsummable)){
    // DOUT("Building type conversion kernel");
    auto f = build_cutensor_type_conversion(einsummable);

    type_conversion_t conversion_kernel {f};

    kernels.insert({einsummable, conversion_kernel});

    return workspace_info_t(0);
  }

  if(is_scale_and_increment(einsummable)){
    auto [scale, increment] = get_increment_scale(einsummable);
    uint64_t size = einsummable.join_shape[0];
    auto lambda = [scale, increment, size]
    (cudaStream_t stream, float* out, const float* in){
      scale_and_increment(out, in, stream, scale, increment, size);
    };

    scale_t scale_kernel {scale, lambda};

    kernels.insert({einsummable, scale_kernel});

    return workspace_info_t(0);
  }


  if(is_custom_kernel1(einsummable)){
    // DOUT("Building custom kernel 1");
    uint64_t size = einsummable.join_shape[0];

    auto lambda = cutensor_silu_elementwise(size);

    custom_kernel_1_t custom_kernel {lambda};

    kernels.insert({einsummable, custom_kernel});

    return workspace_info_t(0);
  }

  if(is_elementwise_with_pow(einsummable)){
    // DOUT("Building elementwise with pow kernel");
    uint64_t a_size = einsummable.join_shape[0];
    cutensor_elementwise_op_t op_ele = make_mul_op(einsummable);
    auto func_elem = build_elementwise_and_pow(op_ele, a_size);

    uint64_t worksize = a_size*sizeof(float);

    pow_and_elementwise_t pow_ane_ele_kernel {func_elem, worksize, a_size};

    kernels.insert({einsummable, pow_ane_ele_kernel});

    return workspace_info_t(worksize);
  }

  if(is_c64_elementwise_multiply(einsummable)){
    // DOUT("Building c64 elementwise multiply kernel");
    auto c = make_contraction(einsummable);

    kernels.insert({einsummable,c});

    return workspace_info_t(c.worksize);
  }

  // Assumption: We can execute all simple scalarops
  // trying to build list of elewise after we tried all the special kernels
  optional<list_simple_scalarop_t> maybe_sops =
    list_simple_scalarop_t::make(einsummable.join);
  if (maybe_sops) {
    list_simple_scalarop_t const& sops = maybe_sops.value(); 
    
    elementwise_t kernel {
      .sops = sops,
      .plans = make_elementwise_plans(sops, einsummable.join_shape,
                                      einsummable.inns, einsummable.out_rank),
      .join_shape = einsummable.join_shape,
      .inns = einsummable.inns,
      .out_rank = einsummable.out_rank
    };

    kernels.insert({einsummable, kernel});

    return workspace_info_t(elementwise_workspace_size(kernel));
  }

  return std::nullopt;
}

workspace_info_t kernel_manager_t::workspace_size(
  kernel_manager_t::kernel_info_t const& kernel) const
{
  if(std::holds_alternative<contraction_t>(kernel)) {
    return workspace_info_t(std::get<contraction_t>(kernel).worksize);
  }else if(std::holds_alternative<elementwise_t>(kernel)) {
    return workspace_info_t(elementwise_workspace_size(std::get<elementwise_t>(kernel)));
  }else if(std::holds_alternative<reduction_t>(kernel)) {
    return workspace_info_t(std::get<reduction_t>(kernel).worksize);
  }else if(std::holds_alternative<pow_and_elementwise_t>(kernel)) {
    return workspace_info_t(std::get<pow_and_elementwise_t>(kernel).worksize);
  }else {
    return workspace_info_t(0);
  }

  throw std::runtime_error("workspace_size: should not reach");
}


void kernel_manager_t::operator()(
  touch_t const& touch,
  cudaStream_t stream,
  void* out,
  void const* inn) const
{
  cudaSetDevice(device);
  auto f = build_touch(touch);
  f(stream, out, inn);
}

void kernel_manager_t::operator()(
  einsummable_t const& e,
  cudaStream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  cudaSetDevice(device);
  auto const& info = get_built_kernel_info(e);
  call(info, stream, out, inns, maybe_workspace);
}

void kernel_manager_t::lowerTri_fill(fill_t::lowertri_t const& l, 
  cudaStream_t stream, void* out) const
{
  int type;
  if (l.lower.dtype == dtype_t::f16) {
    type = 0;
  } else if (l.lower.dtype == dtype_t::f32) {
    type = 1;
  } else if (l.lower.dtype == dtype_t::f64) {
    type = 2;
  } else if (l.lower.dtype == dtype_t::c64) {
    type = 3;
  } else {
    throw std::runtime_error("lowerTri_fill: unknown dtype");
  }
  cudaSetDevice(device);
  fillTri_dispatch(out, l.nrow, l.ncol, l.start, *reinterpret_cast<const uint64_t*>(l.lower.data), 
                    *reinterpret_cast<const uint64_t*>(l.upper.data), stream, type);
}

void kernel_manager_t::constant_fill(fill_t::constant_t const& c, 
  cudaStream_t stream, void* out) const
{
  int type;
  if (c.value.dtype == dtype_t::f16) {
    type = 0;
  } else if (c.value.dtype == dtype_t::f32) {
    type = 1;
  } else if (c.value.dtype == dtype_t::f64) {
    type = 2;
  } else if (c.value.dtype == dtype_t::c64) {
    type = 3;
  } else {
    throw std::runtime_error("constant_fill: unknown dtype");
  }
  cudaSetDevice(device);
  auto num_elements = 1;
  for (auto dim : c.shape) {
    num_elements *= dim;
  }
  // print value
  // printf("value: %f\n", *(float*)(c.value.raw()));
  fill_constant_dispatch(out, num_elements, *reinterpret_cast<const uint64_t*>(c.value.data), 
    stream, type);
}

void kernel_manager_t::call(
  kernel_manager_t::kernel_info_t const& kernel,
  cudaStream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  cudaSetDevice(device);
  using std::holds_alternative;
  using std::get;

  // std::cout << "Calling kernel" << std::endl;

  auto assert_num_inputs = [&inns](int n) {
    if(inns.size() != n) {
      throw std::runtime_error("kernel manager: incorrect number of input tensors");
    }
  };

  if(holds_alternative<matmul_t>(kernel)) {
    // std::cout << "Calling matmul" << std::endl;
    auto const& m = get<matmul_t>(kernel);
    auto const& [dtype, ni, nj, 
    nk, _0, _1, _2] = m;
    execute_matmul(
      m, stream,
      out, inns[0], inns[1]);
  } else if(holds_alternative<contraction_t>(kernel)) {
    assert_num_inputs(2);

    auto const& c = get<contraction_t>(kernel);

    void* workspace = nullptr;
    uint64_t wsz = 0;
    if(c.worksize != 0) {
      if(maybe_workspace) {
        workspace = std::get<0>(maybe_workspace.value());  
        wsz       = std::get<1>(maybe_workspace.value());  
      } else {
        throw std::runtime_error("workspace required; none given");
      }
    }
    // std::cout << "Calling contraction" << std::endl;
    execute_contraction(
      c, stream,
      out, inns[0], inns[1],
      workspace, wsz);
  } else if(holds_alternative<reduction_t>(kernel)) {
    assert_num_inputs(1);

    auto r = get<reduction_t>(kernel);
    auto [workspace, wsz] = maybe_workspace.value();
    execute_reduction(r, stream, out, inns, workspace, wsz);
  } else if(holds_alternative<power_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [pow,f] = get<power_t>(kernel);

    f(stream,(float*)out,(float*)inns[0]);
  } else if(holds_alternative<type_conversion_t>(kernel)) {
    assert_num_inputs(1);
    // DOUT("Calling type conversion");

    auto const& [f] = get<type_conversion_t>(kernel);

    f(stream,cutensor_handle,out,inns);
  } else if(holds_alternative<elementwise_t>(kernel)) {
    auto e = get<elementwise_t>(kernel);
    void* workspace = nullptr;
    uint64_t wsz = 0;
    if (workspace_size(kernel).value() != 0) {
      if(maybe_workspace) {
        workspace = std::get<0>(maybe_workspace.value());  
        wsz       = std::get<1>(maybe_workspace.value());  
      } else {
        // DOUT("NOTE: running a elementwise kernel with no workspace");
      }
    }
    execute_elementwise(e, stream, out, inns, workspace, wsz);

  } else if(holds_alternative<scale_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [scale,f] = get<scale_t>(kernel);

    f(stream,(float*)out,(float*)inns[0]);
  } else if(holds_alternative<pow_and_elementwise_t>(kernel)) {
    auto const& [f, worksizep, a_size] = get<pow_and_elementwise_t>(kernel);

    auto [workspace, wsz] = maybe_workspace.value();

    f(stream,cutensor_handle,out,inns,workspace,wsz);
  } else if(holds_alternative<custom_kernel_1_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [f] = get<custom_kernel_1_t>(kernel);

    f(stream,cutensor_handle,out,inns);
  }    
}

kernel_manager_t::kernel_info_t const&
kernel_manager_t::get_built_kernel_info(einsummable_t const& e) const
{
  auto iter = kernels.find(e.merge_adjacent_dims());
  if(iter == kernels.end()) {
    throw std::runtime_error("get_built_kernel_info: this einsummable has not been built");
  }
  return iter->second;
}

optional<kernel_manager_t::matmul_t>
kernel_manager_t::make_matmul(einsummable_t const& e)
{
  if(!e.has_aggregation()) {
    return std::nullopt;
  }
  if(e.castable.value() != castable_t::add) {
    return std::nullopt;
  }

  string s = e.str();
  auto fix = einsummable_t::normalize_str;

  matmul_t ret {
    .dtype = e.out_dtype(),
    .ni = e.join_shape[0],
    .nj = e.join_shape[2],
    .nk = e.join_shape[1],
    .trans_l = false,
    .trans_r = false,
    .swap = false
  };

  // Here our the 8 possible matmul strings
  if(s == fix("ij,jk->ik")) {
    return ret;
  } else if(s == fix("ij,kj->ik")) {
    ret.trans_r = true;
    return ret;
  } else if(s == fix("ji,jk->ik")) {
    ret.trans_l = true;
    return ret;
  } else if(s == fix("ji,kj->ik")) {
    ret.trans_l = true;
    ret.trans_r = true;
    return ret;
  } else if(s == fix("jk,ij->ik")) {
    ret.swap = true;
    return ret;
  } else if(s == fix("jk,ij->ik")) {
    ret.trans_r = true;
    ret.swap = true;
    return ret;
  } else if(s == fix("jk,ji->ik")) {
    ret.trans_l = true;
    ret.swap = true;
    return ret;
  } else if(s == fix("jk,ji->ik")) {
    ret.trans_l = true;
    ret.trans_r = true;
    ret.swap = true;
    return ret;
  }

  return std::nullopt;
}

kernel_manager_t::contraction_t 
kernel_manager_t::make_contraction(einsummable_t const& einsummable)
{
  auto const& e = einsummable; // an alias
  kernel_manager_t::contraction_t c;

  std::vector<int> modeA = e.inns[0];
  std::vector<int> modeB = e.inns[1];
  std::vector<int> modeC;
  for (int i = 0; i < e.out_rank; i++) {
    modeC.push_back(i);
  }

  int nmodeA = e.inns[0].size();
  int nmodeB = e.inns[1].size();
  int nmodeC = e.out_rank;

  std::reverse(modeA.begin(), modeA.end());
  std::reverse(modeB.begin(), modeB.end());
  std::reverse(modeC.begin(), modeC.end());
  dtype_t type = e.inn_dtype(0);
  c.dtype = type;
  cutensorDataType_t typeTensor = dtype_to_cudatype(type);

  const cutensorComputeDescriptor_t typeCompute = dtype_to_computetype(type);
  
  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(e.join_shape[mode]);
  }
  vector<int64_t> extent_B;
  for(auto const& mode: modeB) {
    extent_B.push_back(e.join_shape[mode]);
  }

  vector<int64_t> extent_C;
  for(auto const& mode: modeC) {
    extent_C.push_back(e.join_shape[mode]);
  }

  const uint32_t kAlignment = 128;
   

  // Set up Tensor Descriptors for A, B, and C
  handle_cutensor_error(
    cutensorCreateTensorDescriptor(
      cutensor_handle,
      &c.descA,
      nmodeA,
      extent_A.data(),
      NULL,/*stride*/
      typeTensor, kAlignment ) );

  cutensorTensorDescriptor_t descB;
  handle_cutensor_error(
    cutensorCreateTensorDescriptor(
      cutensor_handle,
      &c.descB,
      nmodeB,
      extent_B.data(),
      NULL,/*stride*/
      typeTensor, kAlignment ) );

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(
    cutensorCreateTensorDescriptor(
      cutensor_handle,
      &descC,
      nmodeC,
      extent_C.data(),
      NULL,/*stride*/
      typeTensor, kAlignment ));


  // Init Contraction Descriptor need to be in the format of
  // D = alpha * A * B + beta * C
  // so we should probably use a C for both C and D
  handle_cutensor_error(
    cutensorCreateContraction(
      cutensor_handle,
      &c.desc,
      c.descA, modeA.data(), CUTENSOR_OP_IDENTITY,
      c.descB, modeB.data(), CUTENSOR_OP_IDENTITY,
      descC, modeC.data(), CUTENSOR_OP_IDENTITY,
      descC, modeC.data(),
      typeCompute) );

  
  
  //const cutensorAlgo_t algo = CUTENSOR_ALGO_TTGT;
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  handle_cutensor_error(
    cutensorCreatePlanPreference(
    cutensor_handle,
    &planPref,
    algo,
    CUTENSOR_JIT_MODE_NONE));
  
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  handle_cutensor_error(
    cutensorEstimateWorkspaceSize(cutensor_handle,
    c.desc,
    planPref,
    workspacePref,
    &workspaceSizeEstimate));

  /*
  handle_cutensor_error(
    cutensorCreatePlan(cutensor_handle,
      &c.plan,
      c.desc,
      planPref,
      workspaceSizeEstimate));*/
  
  cutensorStatus_t status = cutensorCreatePlan(cutensor_handle,
      &c.plan,
      c.desc,
      planPref,
      workspaceSizeEstimate);

  if(status != CUTENSOR_STATUS_SUCCESS){
    const cutensorAlgo_t algo = CUTENSOR_ALGO_TTGT;

    handle_cutensor_error(
      cutensorCreatePlanPreference(
      cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));
    
    
    handle_cutensor_error(
      cutensorEstimateWorkspaceSize(cutensor_handle,
      c.desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

    
    handle_cutensor_error(
      cutensorCreatePlan(cutensor_handle,
        &c.plan,
        c.desc,
        planPref,
        workspaceSizeEstimate));
  }
  
  uint64_t actualWorkspaceSize = 0;
  handle_cutensor_error(
    cutensorPlanGetAttribute(cutensor_handle,
      c.plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));
  
  c.worksize = actualWorkspaceSize;


  return c; 
}

void kernel_manager_t::execute_matmul(
  kernel_manager_t::matmul_t const& matmul,  
  cudaStream_t stream,
  void* out,
  void const* lhs,
  void const* rhs) const
{
  // TODO: use the 64bit version when using latest version of cublas

  // convert from row to column major

  // DOUT("Calling cublas");

  auto const& [dtype, ni, nj, 
    nk, _0, _1, _2] = matmul;

  // print all the parameters
  // std::cout << "dtype: " << dtype << std::endl;
  // std::cout << "ni: " << ni << std::endl;
  // std::cout << "nj: " << nj << std::endl;
  // std::cout << "nk: " << nk << std::endl;
  // std::cout << "TL: " << _0 << std::endl;
  // std::cout << "TR: " << _1 << std::endl;
  // std::cout << "SW: " << _2 << std::endl;

  // row major      column major   
  // ij,jk->ik      ji,kj->ki
  // ij,kj->ik      ji,jk->ki
  // ji,jk->ik      ij,kj->ki
  // ji,kj->ik      ij,jk->ki
  //
  // jk,ij->ik      kj,ji->ki
  // kj,ij->ik      jk,ji->ki
  // jk,ji->ik      kj,ij->ki
  // kj,ji->ik      jk,ij->ki

  // To go from row to column major, 
  // these get flipped
  bool trans_l = !matmul.trans_l;
  bool trans_r = !matmul.trans_r;
  bool swap    = !matmul.swap;

  int m = nk; // num rows of c
  int n = ni; // num cols of c
  int k = nj; // the other dimension

  int ldl = trans_l ? ni : nj ;
  int ldr = trans_r ? nj : nk ;
  int ldo = m;
  
  handle_cublas_error(cublasSetStream(cublas_handle, stream));

  if(swap) {
    std::swap(trans_l, trans_r);
    std::swap(lhs, rhs);
    std::swap(ldl, ldr);
  } 

  // auto start = std::chrono::high_resolution_clock::now();
  if(dtype == dtype_t::f16) {
    // DOUT("calling cublasSgemm");
    cublasHgemm(
     cublas_handle, 
     trans_l ? CUBLAS_OP_T : CUBLAS_OP_N,
     trans_r ? CUBLAS_OP_T : CUBLAS_OP_N,
     m, n, k, 
     reinterpret_cast<__half const*>(&one_half), 
     reinterpret_cast<__half const*>(lhs), ldl,
     reinterpret_cast<__half const*>(rhs), ldr,
     reinterpret_cast<__half const*>(&zero_half),
     reinterpret_cast<__half*>(out), ldo);
  } else if (dtype == dtype_t::f32){
    cublasSgemm(
     cublas_handle, 
     trans_l ? CUBLAS_OP_T : CUBLAS_OP_N,
     trans_r ? CUBLAS_OP_T : CUBLAS_OP_N,
     m, n, k, 
     &one_float, 
     static_cast<float const*>(lhs), ldl,
     static_cast<float const*>(rhs), ldr,
     &zero_float,
     static_cast<float*>(out), ldo);
  } else if (dtype == dtype_t::f64){
    cublasDgemm(
     cublas_handle, 
     trans_l ? CUBLAS_OP_T : CUBLAS_OP_N,
     trans_r ? CUBLAS_OP_T : CUBLAS_OP_N,
     m, n, k, 
     &one_double, 
     static_cast<double const*>(lhs), ldl,
     static_cast<double const*>(rhs), ldr,
     &zero_double,
     static_cast<double*>(out), ldo);
  } else if (dtype == dtype_t::c64){
    cublasCgemm(
     cublas_handle, 
     trans_l ? CUBLAS_OP_T : CUBLAS_OP_N,
     trans_r ? CUBLAS_OP_T : CUBLAS_OP_N,
     m, n, k, 
     reinterpret_cast<cuComplex const*>(&one_complex), 
     reinterpret_cast<cuComplex const*>(lhs), ldl,
     reinterpret_cast<cuComplex const*>(rhs), ldr,
     reinterpret_cast<cuComplex const*>(&zero_complex),
     reinterpret_cast<cuComplex*>(out), ldo);
  }
  else {
    throw std::runtime_error("not implemented: the other cublas matmul dtypes");
  }
}

void kernel_manager_t::execute_contraction(
    kernel_manager_t::contraction_t const& c,
    cudaStream_t stream,
    void* out,
    void const* lhs,
    void const* rhs,
    void* work,
    uint64_t given_worksize) const
{
  if(given_worksize < c.worksize) {
    throw std::runtime_error("not enough workspace given for this contraction");
  }

  void const* one_ptr  = 
    c.dtype == dtype_t::f16 ? get_one_ptr( dtype_t::f32) : get_one_ptr(c.dtype);
  void const* zero_ptr = 
    c.dtype == dtype_t::f16 ? get_zero_ptr(dtype_t::f32) : get_zero_ptr(c.dtype);

  //std::cout << "Calling cutensor contraction" << std::endl;
  handle_cutensor_error(
    cutensorContract(
      cutensor_handle, 
      c.plan,              // TODO: c can change locations, can cutensorContrcation
                            //       is async... this might be nitpicky
      one_ptr,   // alpha
      lhs, rhs,  // A,B
      zero_ptr,  // beta
      out,       // C
      out,       // D
      work, 
      given_worksize, 
      stream),
    "contraction operator");
}

kernel_manager_t::reduction_t
kernel_manager_t::make_reduction(einsummable_t const& e)
{
  std::vector<int> modeA = e.inns[0];
  std::vector<int> modeC;
  for (int i = 0; i < e.out_rank; i++) {
    modeC.push_back(i);
  }

  int32_t nmodeA = modeA.size();
  int32_t nmodeC = modeC.size();

  std::reverse(modeA.begin(), modeA.end());
  std::reverse(modeC.begin(), modeC.end());
  dtype_t type = e.inn_dtype(0);
  cutensorDataType_t typeTensor = dtype_to_cudatype(type);
  const cutensorComputeDescriptor_t typeCompute = dtype_to_computetype(type);

  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(e.join_shape[mode]);
  }

  vector<int64_t> extent_C;
  for(auto const& mode: modeC) {
    extent_C.push_back(e.join_shape[mode]);
  }

  const uint32_t kAlignment = 256; 

  cutensorTensorDescriptor_t descA;
  handle_cutensor_error(
    cutensorCreateTensorDescriptor(
      cutensor_handle,
      &descA,
      nmodeA,
      extent_A.data(),
      NULL,/*stride*/
      typeTensor, kAlignment ) );


  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(
    cutensorCreateTensorDescriptor(
      cutensor_handle,
      &descC,
      nmodeC,
      extent_C.data(),
      NULL,/*stride*/
      typeTensor, kAlignment ) );
  

  cutensorOperator_t opReduce;
  if(e.castable == castable_t::add) {
    opReduce = CUTENSOR_OP_ADD;
  } else if(e.castable == castable_t::mul) {
    opReduce = CUTENSOR_OP_MUL;
  } else if(e.castable == castable_t::min) {
    opReduce = CUTENSOR_OP_MIN;
  } else if(e.castable == castable_t::max) {
    opReduce = CUTENSOR_OP_MAX;
  } else {
    throw std::runtime_error("should not reach: missing castable");
  }

  cutensorOperationDescriptor_t desc;
  handle_cutensor_error(
    cutensorCreateReduction(
      cutensor_handle,
      &desc,
      descA, modeA.data(), CUTENSOR_OP_IDENTITY,
      descC, modeC.data(), CUTENSOR_OP_IDENTITY,
      descC, modeC.data(),
      opReduce, typeCompute) );
  
  
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  handle_cutensor_error(
    cutensorCreatePlanPreference(
    cutensor_handle,
    &planPref,
    algo,
    CUTENSOR_JIT_MODE_NONE));
  
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  handle_cutensor_error(
    cutensorEstimateWorkspaceSize(cutensor_handle,
    desc,
    planPref,
    workspacePref,
    &workspaceSizeEstimate));


  cutensorPlan_t plan;
  handle_cutensor_error(
    cutensorCreatePlan(cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));
  
  uint64_t actualWorkspaceSize = 0;
  handle_cutensor_error(
    cutensorPlanGetAttribute(cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));
  

  uint64_t worksize = actualWorkspaceSize;

  return reduction_t {
    .dtype = type,
    .worksize = worksize,
    .plan = plan
    
  };
}

void kernel_manager_t::execute_reduction(
  kernel_manager_t::reduction_t const& r,
  cudaStream_t stream,
  void* out,
  vector<void const*> inns,
  void* work,
  uint64_t given_worksize) const
{
  void* ptr1;
  void* ptr2;
  float16_t alpha1, beta1;
  float alpha2, beta2;
  double alpha3, beta3;
  std::complex<float> alpha4(1.0f, 0.0f);
  std::complex<float> beta4(0.0f, 0.0f);
  auto dtype = r.dtype;
  const cutensorComputeDescriptor_t typeCompute = dtype_to_computetype(dtype);

  if(dtype == dtype_t::f16){
    alpha2 = 1.0f;
    ptr1 = static_cast<void*>(&alpha2);
    beta2 = 0.0f;
    ptr2 = static_cast<void*>(&beta2);
  }
  else if(dtype == dtype_t::f32){
    alpha2 = 1.0f;
    ptr1 = static_cast<void*>(&alpha2);
    beta2 = 0.0f;
    ptr2 = static_cast<void*>(&beta2);
  }
  else if(dtype == dtype_t::f64){
    alpha3 = 1.0;
    ptr1 = static_cast<void*>(&alpha3);
    beta3 = 0.0;
    ptr2 = static_cast<void*>(&beta3);
  }
  else if(dtype == dtype_t::c64){
    ptr1 =  static_cast<void*>(&alpha4);
    ptr2 =  static_cast<void*>(&beta4);
  }

  void const* alpha = ptr1; 
  void const* beta = ptr2; 

  handle_cutensor_error(cutensorReduce(cutensor_handle, r.plan,
                alpha, inns[0],
                (const void*)&beta, out, 
                out, work, given_worksize, stream));
}

static 
vector<int> _which_ew_memory(list_simple_scalarop_t const& sops)
{
  // Here is the idea: for each scalarop, where does the output memory
  // need to be set at?
  // -1     : the output memory
  //  i >= 0: workspace at memory i
  // We assume that every workspace tensor has the same number of output elements
  // N. We assume that the largest dtype used is D. Then the maximum size that
  // any temporary or the output can be is S = dtype_size(D)*N
  // The total amount of workspace memory used by
  // this scheme is S*(1 + max(ret)).
  //
  // TODO: We are assuming that mixed dtype scalarops that use workspace memory
  //       is unlikely. As in, we expect that the only list_simple_scalarop 
  //       that converts between dtypes is writes directly to the output memory.
  //       If this assumption does not hold, maybe it is worthwhile to come up
  //       with a better memory scheme.

  vector<int> avail;
  int next_avail = 0; // this always equals max(ret) + 1, 
                      // except when ret.size() == 0, then it is 0

  // Always reuse memory in avail, otherwise use next_avail.
  // Update state(= avail,next_avail) accordingly
  auto next_workspace = [&] {
    if(avail.size() > 0) {
      int x = avail.back();
      avail.pop_back();
      return x;
    } else {
      int x = next_avail;
      next_avail++;
      return x;
    }
  };

  // Figure out where the last usage of each temporary is,
  // used for deleting later.
  vector<int> last_usage(sops.ops.size()-1, -1);
  for(int i = 0; i != sops.ops.size(); ++i) {
    // TODO: check all auto const& [op, args] = sops.ops
    auto const& [op, args] = sops.ops[i];
    for(int which = 0; which != op.num_inns(); ++which) {
      int const& arg = args[which];
      if(arg < 0) {
        int tmp = -1*arg - 1;
        last_usage[tmp] = std::max(i, last_usage.at(tmp));
      }
    }
  }

  // Fill out the return vector.
  //   1. "allocate" (via next_workspace)
  //   2. "free"     (by adding to avail)
  vector<int> ret;
  for(int i = 0; i != sops.ops.size() - 1; ++i) {
    // A workspace is needed for all ops[i], i < sop.ops.size() - 1.
    ret.push_back(next_workspace());    

    // If ops[i] is using an argument for the last time, then
    // "free" it by pushing it back into avail so it can be
    // reused the next time a workspace is needed.
    auto const& [op, args] = sops.ops[i];
    for(int which = 0; which != op.num_inns(); ++which) {
      int const& arg = args[which];
      if(arg < 0) {
        int tmp = -1*arg - 1;
        // ^ tmp was used by the sops[i]. Was it the last usage?
        // If so, "delete"
        if(last_usage[tmp] == i) {
          avail.push_back(ret[tmp]);
        }
      }
    }
  }

  // the last spot is always the output memory
  ret.push_back(-1);
  return ret;
}

uint64_t kernel_manager_t::elementwise_workspace_size(
  kernel_manager_t::elementwise_t const& e) const
{
  vector<int> usage_mems = _which_ew_memory(e.sops);
  dtype_t max_dtype = e.sops.max_dtype(); // TODO: need to be able to get max_dtype 
  uint64_t max_size = product(e.join_shape)*dtype_size(max_dtype);
  int which_last = *std::max_element(usage_mems.begin(), usage_mems.end());
  return max_size * (1 + which_last);
}

void kernel_manager_t::execute_elementwise(
  elementwise_t const& e,
  cudaStream_t stream,
  void* out_mem,
  vector<void const*> inn_mems,
  void* work_mem,
  uint64_t given_worksize) const
{
  dtype_t max_dtype = e.sops.max_dtype(); // TODO: need to be able to get max_dtype 
  uint64_t max_size = product(e.join_shape)*dtype_size(max_dtype);
  vector<int> usage_mems = _which_ew_memory(e.sops);
  vector<int> out_idxs = vector_iota<int>(e.out_rank);

  // Check we have enough workspace memory
  {
    int which_last = *std::max_element(usage_mems.begin(), usage_mems.end());
    uint64_t required = max_size * (1 + which_last);
    if(required > given_worksize) {
      throw std::runtime_error("elementwise: need more workspace");
    }
  }

  auto get_workspace_at = [&](int which_tmp) -> void* {
    int const& which = usage_mems[which_tmp];
    return increment_void_ptr(work_mem, which*max_size);
  };
  auto get_inn_memory_at = [&](int arg) -> void const* {
    if(arg < 0) {
      return get_workspace_at(-1*arg-1);
    } else {
      return inn_mems[arg];
    }
  };
  auto get_inn_idxs_at = [&](int arg) -> vector<int> const& {
    if(arg < 0) {
      return out_idxs;
    } else {
      return e.inns[arg];
    }
  };
  for(int which_sop = 0; which_sop != e.sops.ops.size(); ++which_sop) {
    auto const& [sop, args] = e.sops.ops[which_sop];
    auto plan = e.plans[which_sop];

    void* this_out_mem = 
      which_sop == e.sops.ops.size() - 1 ?
      out_mem                        :
      get_workspace_at(which_sop)    ;

    if(sop.is_scale()) {
      execute_sop_scale(
        sop.get_scale(), stream, this_out_mem,
        get_inn_memory_at(args[0]),
        plan);
    } else if(sop.is_unary()) {
      execute_sop_unary(
        sop.get_unary(), stream, this_out_mem, 
        get_inn_memory_at(args[0]),
        plan);
    } else if(sop.is_binary()) {
      // first check if we need to swap the arguments
      auto lhs = get_inn_idxs_at(args[0]);
      auto rhs = get_inn_idxs_at(args[1]);
      bool swap = false;
      if (rhs.size() > lhs.size()) {
        swap = true;
      }
      execute_sop_binary(
        sop.get_binary(), stream, this_out_mem,
        get_inn_memory_at(args[0]), get_inn_memory_at(args[1]),
        plan, swap);
    } else {
      throw std::runtime_error("missing sop case!");
    }
  }
}

vector<cutensorPlan_t> kernel_manager_t::make_elementwise_plans(
  list_simple_scalarop_t const& sops,
  vector<uint64_t> const& join_shape,
  vector<vector<int>> const& inns,
  int out_rank) const
{
  vector<cutensorPlan_t> ret;
  vector<int> out_idxs = vector_iota<int>(out_rank);

  auto get_inn_idxs_at = [&](int arg) -> vector<int> const& {
    if(arg < 0) {
      return out_idxs;
    } else {
      return inns[arg];
    }
  };
  for(int which_sop = 0; which_sop != sops.ops.size(); ++which_sop) {
    auto const& [sop, args] = sops.ops[which_sop];

    if(sop.is_scale()) {
      ret.push_back(
        sop_scale_plan(
          sop.get_scale(), get_inn_idxs_at(args[0]), join_shape));
    }
    else if(sop.is_unary()) {
      ret.push_back(
        sop_unary_plan(
          sop.get_unary(), get_inn_idxs_at(args[0]), join_shape));
    }
    else if(sop.is_binary()) {
      ret.push_back(
        sop_binary_plan(
          sop.get_binary(), get_inn_idxs_at(args[0]), get_inn_idxs_at(args[1]), join_shape));
    }
    else {
      throw std::runtime_error("missing sop case!");
    }
  }
  return ret;
}

cutensorPlan_t kernel_manager_t::sop_scale_plan(
  simple_scalarop_t::scale_t const& op,
  vector<int> const& inn_idxs,
  vector<uint64_t> const& out_shape) const
{
  if (op.bop == simple_scalarop_t::bop_t::mul) {
    return sop_scale_mul_plan(op.scale, inn_idxs, out_shape);
  } else if (op.bop == simple_scalarop_t::bop_t::add) {
    return sop_scale_add_plan(op.scale, inn_idxs, out_shape);
  } else {
    throw std::runtime_error("not implemented: sop_scale");
  }
}

cutensorPlan_t kernel_manager_t::sop_scale_add_plan(
  scalar_t const& scale,
  vector<int> const& inn_idxs_,
  vector<uint64_t> const& out_shape_) const
{
  vector<uint64_t> out_shape = out_shape_;
  out_shape.push_back(1);

  vector<int> lhs_idxs = inn_idxs_;
  lhs_idxs.push_back(inn_idxs_.size());

  vector<int> rhs_idxs;
  rhs_idxs.push_back(inn_idxs_.size());

  simple_scalarop_t::unary_t unary {
    .scale = scalar_t::one(scale.dtype),
    .op = simple_scalarop_t::uop_t::identity,
  };

  simple_scalarop_t::binary_t binary {
    .op = simple_scalarop_t::bop_t::add,
    .lhs = unary,
    .rhs = unary
  };

  return sop_binary_plan(binary, lhs_idxs, rhs_idxs, out_shape);
}

cutensorPlan_t kernel_manager_t::sop_scale_mul_plan(
  scalar_t const& scale,
  vector<int> const& inn_idxs,
  vector<uint64_t> const& out_shape) const
{
  uint32_t const kAlignment = 128;
  cutensorOperator_t opElementwise = CUTENSOR_OP_IDENTITY;
  auto type = scale.dtype;
  auto typeCompute = dtype_to_computetype(type);
  auto typeA = dtype_to_cudatype(type);

  vector<int> const& inn_modes = inn_idxs;
  std::vector<int32_t> modeA(inn_modes.begin(),inn_modes.end());
  int32_t nmodeA = modeA.size();
  
  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(out_shape[mode]);
  }

  void const* alpha = scale.raw();
  cutensorTensorDescriptor_t descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descA,
                                              nmodeA,
                                              extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descC,
                                              nmodeA,
                                              extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));
  
  
  cutensorOperationDescriptor_t desc;
  handle_cutensor_error(cutensorCreatePermutation(cutensor_handle,
                                          &desc,
                                          descA,
                                          modeA.data(),
                                          opElementwise,
                                          descC,
                                          modeA.data(),
                                          typeCompute));

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(cutensor_handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));
  
  cutensorPlan_t plan;
  handle_cutensor_error(cutensorCreatePlan(cutensor_handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));
  return plan;
}

cutensorPlan_t kernel_manager_t::sop_unary_plan(
  simple_scalarop_t::unary_t const& op,
  vector<int> const& inn_idxs,
  vector<uint64_t> const& out_shape) const
{
  uint32_t const kAlignment = 128;
  auto type = op.scale.dtype;
  auto unary = op;
  auto scale = unary.scale;
  auto uop = unary.op;
  if (uop == simple_scalarop_t::uop_t::square) {
    simple_scalarop_t::unary_t unit_unary {
      .scale = scalar_t::one(type),
      .op = simple_scalarop_t::uop_t::identity
    }; 
    simple_scalarop_t::unary_t scale_unary {
      .scale = scale,
      .op = simple_scalarop_t::uop_t::identity
    };
    simple_scalarop_t::binary_t new_binary {
      .op = simple_scalarop_t::bop_t::mul,
      .lhs = scale_unary,
      .rhs = unit_unary
    };

    return sop_binary_plan(
      new_binary, inn_idxs, inn_idxs, out_shape);
  }

  auto opElementwise = uop_to_cutensorOp(uop);
  auto typeCompute = dtype_to_computetype(type);
  auto typeA = dtype_to_cudatype(type);

  vector<int> const& inn_modes = inn_idxs;
  std::vector<int32_t> modeA(inn_modes.begin(),inn_modes.end());
  int32_t nmodeA = modeA.size();
  
  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(out_shape[mode]);
  }

  void const* alpha = scale.raw();
  cutensorTensorDescriptor_t descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descA,
                                              nmodeA,
                                              extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descC,
                                              nmodeA,
                                              extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));
  
  
  cutensorOperationDescriptor_t desc;
  handle_cutensor_error(cutensorCreatePermutation(cutensor_handle,
                                          &desc,
                                          descA,
                                          modeA.data(),
                                          opElementwise,
                                          descC,
                                          modeA.data(),
                                          typeCompute));

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(cutensor_handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));
  
  cutensorPlan_t plan;
  handle_cutensor_error(cutensorCreatePlan(cutensor_handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));
  return plan;
}

cutensorPlan_t kernel_manager_t::sop_binary_plan(
  simple_scalarop_t::binary_t const& op,
  vector<int> const& lhs_idxs,
  vector<int> const& rhs_idxs,
  vector<uint64_t> const& out_shape) const
{
  uint32_t const kAlignment = 128;
  auto bop = op.op;
  auto lhs = op.lhs;
  auto rhs = op.rhs;
  assert(lhs.scale.dtype == rhs.scale.dtype);
  cutensorOperator_t opElementwise = bop_to_cutensorOp(bop);
  cutensorOperator_t opElemmentwise_lhs = uop_to_cutensorOp(lhs.op);
  cutensorOperator_t opElemmentwise_rhs = uop_to_cutensorOp(rhs.op);

  std::vector<int> modeA = lhs_idxs;
  std::vector<int> modeC = rhs_idxs;

  bool swapped = (modeA.size() > modeC.size());
  if(swapped){
    std::swap(modeA, modeC);
    std::swap(lhs, rhs);
  }
  int nmodeA = modeA.size();
  int nmodeC = modeC.size();

  std::reverse(modeA.begin(), modeA.end());
  std::reverse(modeC.begin(), modeC.end());

  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(out_shape[mode]);
  }
  vector<int64_t> extent_C;
  for(auto const& mode: modeC) {
    extent_C.push_back(out_shape[mode]);
  }

  // should this be true
  // assert(lhs.scale.dtype == rhs.scale.dtype);
  auto typeA = dtype_to_cudatype(lhs.scale.dtype);
  auto typeC = dtype_to_cudatype(rhs.scale.dtype);
  auto typeCompute = dtype_to_computetype(lhs.scale.dtype);
  
  void const* alpha = lhs.scale.raw(); 
  void const* beta = rhs.scale.raw();

  cutensorTensorDescriptor_t  descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descA, nmodeA, extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t  descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(cutensor_handle,
                                              &descC, nmodeC, extent_C.data(),
                                              nullptr /* stride */,
                                              typeC,
                                              kAlignment));

  cutensorOperationDescriptor_t  desc;
  
  handle_cutensor_error(cutensorCreateElementwiseBinary(cutensor_handle, &desc,
                                                  descA, modeA.data(), opElemmentwise_lhs,
                                                  descC, modeC.data(), opElemmentwise_rhs,
                                                  descC, modeC.data(), opElementwise,
                                                  typeCompute));
  
  

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(cutensor_handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));

  

  cutensorPlan_t plan; 
  handle_cutensor_error(cutensorCreatePlan(cutensor_handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));

  return plan;
}

void kernel_manager_t::execute_sop_scale(
  simple_scalarop_t::scale_t const& op,
  cudaStream_t stream,
  void* out_mem,
  void const* inn_mem,
  cutensorPlan_t plan) const
{
  if (op.bop == simple_scalarop_t::bop_t::mul) {
    execute_sop_scale_mul(op.scale, stream, out_mem, inn_mem, plan);
  } else if (op.bop == simple_scalarop_t::bop_t::add) {
    execute_sop_scale_add(op.scale, stream, out_mem, inn_mem, plan);
  } else {
    throw std::runtime_error("not implemented: sop_scale");
  }
}

void kernel_manager_t::execute_sop_scale_add(
  scalar_t const& scale,
  cudaStream_t stream,
  void* out_mem,
  void const* inn_mem,
  cutensorPlan_t plan) const
{
  void* scale_mem = gpu_allocate_memory(dtype_size(scale.dtype), device);
  cudaMemcpy(scale_mem, scale.raw(), dtype_size(scale.dtype), cudaMemcpyHostToDevice);

  simple_scalarop_t::unary_t unary {
    .scale = scalar_t::one(scale.dtype),
    .op = simple_scalarop_t::uop_t::identity,
  };

  simple_scalarop_t::binary_t binary {
    .op = simple_scalarop_t::bop_t::add,
    .lhs = unary,
    .rhs = unary
  };

  execute_sop_binary(binary, stream, out_mem, inn_mem, scale_mem, plan, false);

  cudaFree(scale_mem);
}

void kernel_manager_t::execute_sop_scale_mul(
  scalar_t const& scale,
  cudaStream_t stream,
  void* out_mem,
  void const* inn_mem,
  cutensorPlan_t plan) const
{
  void const* alpha = scale.raw();
  handle_cutensor_error(cutensorPermute(cutensor_handle,
                        plan,
                        alpha, inn_mem, out_mem, stream /* stream */));     
}

void kernel_manager_t::execute_sop_unary(
  simple_scalarop_t::unary_t const& op,
  cudaStream_t stream,
  void* out_mem,
  void const* inn_mem,
  cutensorPlan_t plan) const
{
  uint32_t const kAlignment = 128;
  auto type = op.scale.dtype;
  auto unary = op;
  auto scale = unary.scale;
  auto uop = unary.op;
  if (uop == simple_scalarop_t::uop_t::square) {
    simple_scalarop_t::unary_t unit_unary {
      .scale = scalar_t::one(type),
      .op = simple_scalarop_t::uop_t::identity
    }; 
    simple_scalarop_t::unary_t scale_unary {
      .scale = scale,
      .op = simple_scalarop_t::uop_t::identity
    };
    simple_scalarop_t::binary_t new_binary {
      .op = simple_scalarop_t::bop_t::mul,
      .lhs = scale_unary,
      .rhs = unit_unary
    };

    // for square both sides are the same so we don't need to swap
    return execute_sop_binary(
      new_binary, stream, out_mem, inn_mem, inn_mem, plan, false);
  }
  void const* alpha = scale.raw();
  handle_cutensor_error(cutensorPermute(cutensor_handle,
                        plan,
                        alpha, inn_mem, out_mem, stream /* stream */));                       
}

void kernel_manager_t::execute_sop_binary(
  simple_scalarop_t::binary_t const& op,
  cudaStream_t stream,
  void* out_mem,
  void const* lhs_mem,
  void const* rhs_mem,
  cutensorPlan_t plan,
  bool swapped) const
{
  auto lhs = op.lhs;
  auto rhs = op.rhs;
  assert(lhs.scale.dtype == rhs.scale.dtype);
  void const* alpha = lhs.scale.raw(); 
  void const* beta = rhs.scale.raw();

  if(swapped){
    handle_cutensor_error(cutensorElementwiseBinaryExecute(cutensor_handle,
              plan, alpha, rhs_mem, 
              beta, lhs_mem, 
              out_mem, stream));
  }else{
    handle_cutensor_error(cutensorElementwiseBinaryExecute(cutensor_handle,
              plan, alpha, lhs_mem, 
              beta, rhs_mem, 
              out_mem, stream));
  }
}

void const* kernel_manager_t::get_one_ptr(dtype_t dtype) const {
  if(dtype == dtype_t::f16) {
    return static_cast<void const*>(&one_half);
  } else if(dtype == dtype_t::f32) {
    return static_cast<void const*>(&one_float);
  } else if(dtype == dtype_t::f64) {
    return static_cast<void const*>(&one_double);
  } else if(dtype == dtype_t::c64) {
    return static_cast<void const*>(&one_complex);
  } else {
    throw std::runtime_error("should not reach: unknown data type");
  }
}

void const* kernel_manager_t::get_zero_ptr(dtype_t dtype) const {
  if(dtype == dtype_t::f16) {
    return static_cast<void const*>(&zero_half);
  } else if(dtype == dtype_t::f32) {
    return static_cast<void const*>(&zero_float);
  } else if(dtype == dtype_t::f64) {
    return static_cast<void const*>(&zero_double);
  } else if(dtype == dtype_t::c64) {
    return static_cast<void const*>(&zero_double);
  } else {
    throw std::runtime_error("should not reach: unknown data type");
  }
}



