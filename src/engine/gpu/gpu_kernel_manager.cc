#include "gpu_kernel_manager.h"

#include "utility.h"

kernel_manager_t::kernel_manager_t() {
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
  handle_cutensor_error(
    cutensorDestroy(cutensor_handle),
    "cutensor destroy in kernel_manager destructor");
  handle_cublas_error(
    cublasDestroy(cublas_handle),
    "cublas destroy in kernel_manager destructor");
}

optional<workspace_info_t> 
kernel_manager_t::build(einsummable_t const& e_)
{
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

  // Check for Reudctions
  if(einsummable.has_aggregation()) {
    if(einsummable.inns.size() != 1) {
      return std::nullopt;
    }

    if(einsummable.castable.value() == castable_t::add) {
      // this is something cutensor reduction should be able to do
      vector<int> const& inn_modes = einsummable.inns[0];

      auto inn_shape = einsummable.inn_shapes()[0];

      vector<int> out_modes(einsummable.out_rank);
      std::iota(out_modes.begin(), out_modes.end(), 0);

      auto out_shape = einsummable.out_shape();

      auto reduct = build_cutensor_reduction(
        inn_modes, inn_shape,
        out_modes, out_shape,
        einsummable.castable.value(),
        einsummable.inn_dtype(0));
      
      reduction_t reduct_kernel {reduct};

      kernels.insert({einsummable, reduct_kernel});

      return workspace_info_t();
    }

    return std::nullopt;
  }

  if(is_power_elementwise(einsummable)) {
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
    uint64_t size = einsummable.join_shape[0];

    auto lambda = cutensor_silu_elementwise(size);

    custom_kernel_1_t custom_kernel {lambda};

    kernels.insert({einsummable, custom_kernel});

    return workspace_info_t(0);
  }

  if(is_elementwise_with_pow(einsummable)){
    uint64_t a_size = einsummable.join_shape[0];
    cutensor_elementwise_op_t op_ele = make_mul_op(einsummable);
    auto func_elem = build_elementwise_and_pow(op_ele, a_size);

    uint64_t worksize = a_size*sizeof(float);

    pow_and_elementwise_t pow_ane_ele_kernel {func_elem, worksize, a_size};

    kernels.insert({einsummable, pow_ane_ele_kernel});

    return workspace_info_t(worksize);
  }

  if(is_c64_elementwise_multiply(einsummable)){
    auto c = make_contraction(einsummable);

    kernels.insert({einsummable,c});

    return workspace_info_t(c.worksize);
  }


  auto maybe = make_cutensor_elementwise_op(einsummable); 

  if(maybe&&einsummable.out_dtype()!=dtype_t::c64){

    cutensor_elementwise_op_t op = *maybe;

    auto func = build_cutensor_elementwise(op);

    elementwise_t elementwise_kernel {func};

    kernels.insert({einsummable, elementwise_kernel});

    return workspace_info_t(0);
  }

  return std::nullopt;
}

workspace_info_t kernel_manager_t::workspace_size(
  einsummable_t const& e) const
{
  return workspace_size(get_built_kernel_info(e));
}

workspace_info_t kernel_manager_t::workspace_size(
  kernel_manager_t::kernel_info_t const& kernel) const
{
  if(std::holds_alternative<contraction_t>(kernel)) {
    return workspace_info_t(std::get<contraction_t>(kernel).worksize);
  }else if(std::holds_alternative<pow_and_elementwise_t>(kernel)) {
    return workspace_info_t(std::get<pow_and_elementwise_t>(kernel).worksize);
  }else if(std::holds_alternative<reduction_t>(kernel)) {
    return workspace_info_t();
  }else {
    return workspace_info_t(0);
  }

  throw std::runtime_error("workspace_size: should not reach");
}

uint64_t kernel_manager_t::known_workspace_size(
  einsummable_t const& e, 
  void* out,
  vector<void const*> inns) const
{
  auto const& kernel = get_built_kernel_info(e);

  if(std::holds_alternative<reduction_t>(kernel)) {
    return reduction_worksize(e, out, inns, cutensor_handle);
  } else {
    return workspace_size(kernel).value();
  }
}

void kernel_manager_t::operator()(
  touch_t const& touch,
  cudaStream_t stream,
  void* out,
  void const* inn) const
{
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
  auto const& info = get_built_kernel_info(e);
  call(info, stream, out, inns, maybe_workspace);
}

void kernel_manager_t::call(
  kernel_manager_t::kernel_info_t const& kernel,
  cudaStream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  using std::holds_alternative;
  using std::get;

  auto assert_num_inputs = [&inns](int n) {
    if(inns.size() != n) {
      throw std::runtime_error("kernel manager: incorrect number of input tensors");
    }
  };

  if(holds_alternative<matmul_t>(kernel)) {
    auto const& m = get<matmul_t>(kernel);
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

    execute_contraction(
      c, stream,
      out, inns[0], inns[1],
      workspace, wsz);
  } else if(holds_alternative<reduction_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [f] = get<reduction_t>(kernel);
    auto [workspace, wsz] = maybe_workspace.value();

    f(stream,cutensor_handle,out,inns,workspace,wsz);
  } else if(holds_alternative<power_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [pow,f] = get<power_t>(kernel);

    f(stream,(float*)out,(float*)inns[0]);
  } else if(holds_alternative<type_conversion_t>(kernel)) {
    assert_num_inputs(1);

    auto const& [f] = get<type_conversion_t>(kernel);

    f(stream,cutensor_handle,out,inns);
  } else if(holds_alternative<elementwise_t>(kernel)) {
    auto const& [f] = get<elementwise_t>(kernel);

    f(stream,cutensor_handle,out,inns);

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
  contraction_t c;

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
  cudaDataType_t typeTensor = dtype_to_cudatype(type);

  cutensorComputeType_t typeCompute = dtype_to_computetype(type);
  
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

  // Set up Tensor Descriptors for A, B, and C
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      cutensor_handle,
      &c.descA,
      nmodeA,
      extent_A.data(),
      NULL,/*stride*/
      typeTensor, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descB;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      cutensor_handle,
      &c.descB,
      nmodeB,
      extent_B.data(),
      NULL,/*stride*/
      typeTensor, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      cutensor_handle,
      &c.descC,
      nmodeC,
      extent_C.data(),
      NULL,/*stride*/
      typeTensor, CUTENSOR_OP_IDENTITY ) );

  // get the memory pointers to the tensors
  uint32_t alignmentRequirementA = 16;
  uint32_t alignmentRequirementB = 16;
  uint32_t alignmentRequirementC = 16;

  // Init Contraction Descriptor need to be in the format of
  // D = alpha * A * B + beta * C
  // so we should probably use a C for both C and D
  handle_cutensor_error(
    cutensorInitContractionDescriptor(
      cutensor_handle,
      &c.desc,
      &c.descA, modeA.data(), alignmentRequirementA,
      &c.descB, modeB.data(), alignmentRequirementB,
      &c.descC, modeC.data(), alignmentRequirementC,
      &c.descC, modeC.data(), alignmentRequirementC,
      typeCompute) );

  handle_cutensor_error(cutensorInitContractionFind(
    cutensor_handle, 
    &c.find,
    CUTENSOR_ALGO_DEFAULT));

  c.worksize = 0;
  
  handle_cutensor_error(cutensorContractionGetWorkspaceSize(
     cutensor_handle,
     &c.desc,
     &c.find,
     CUTENSOR_WORKSPACE_RECOMMENDED,
     &c.worksize));

  handle_cutensor_error(cutensorInitContractionPlan(
      cutensor_handle,
      &c.plan,
      &c.desc,
      &c.find,
      c.worksize));

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

  auto const& [dtype, ni, nj, nk, _0, _1, _2] = matmul;

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

  int ldl = trans_l ? nj : ni ;
  int ldr = trans_r ? nk : nj ;
  int ldo = nk;
  
  handle_cublas_error(cublasSetStream(cublas_handle, stream));

  if(swap) {
    std::swap(trans_l, trans_r);
    std::swap(lhs, rhs);
    std::swap(ldl, ldr);
  } 

  if(dtype == dtype_t::f32) {
    //cublasSgemm(
    //  cublas_handle, 
    //  trans_l ? CUBLAS_OP_T : CUBLAS_OP_N,
    //  trans_r ? CUBLAS_OP_T : CUBLAS_OP_N,
    //  m, n, k, 
    //  &one_float, 
    //  static_cast<float const*>(lhs), ldl,
    //  static_cast<float const*>(rhs), ldr,
    //  &zero_float,
    //  static_cast<float*>(out), ldo);
  } else {
    throw std::runtime_error("not implemented: the other cublas matmul dtypes");
  }

  handle_cuda_error(cudaDeviceSynchronize());
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
  //// TODO: remove
  //handle_cuda_error(cudaDeviceSynchronize(), "sync " + write_with_ss(__LINE__));
  //cudaStream_t s01;
  //handle_cuda_error(cudaStreamCreate(&s01), "ec " + write_with_ss(__LINE__));
  //handle_cuda_error(cudaStreamDestroy(s01), "ec " + write_with_ss(__LINE__));
  //handle_cuda_error(cudaDeviceSynchronize(), "sync " + write_with_ss(__LINE__));

  if(given_worksize < c.worksize) {
    throw std::runtime_error("not enough workspace given for this contraction");
  }

  void const* one_ptr  = 
    c.dtype == dtype_t::f16 ? get_one_ptr( dtype_t::f32) : get_one_ptr( c.dtype);
  void const* zero_ptr = 
    c.dtype == dtype_t::f16 ? get_zero_ptr(dtype_t::f32) : get_zero_ptr(c.dtype);

  handle_cutensor_error(
    cutensorContraction(
      cutensor_handle, 
      &c.plan,              // TODO: c can change locations, can cutensorContrcation
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

  //// TODO: remove
  //handle_cuda_error(cudaDeviceSynchronize(), "sync " + write_with_ss(__LINE__));
  //cudaStream_t s02;
  //handle_cuda_error(cudaStreamCreate(&s02), "ec " + write_with_ss(__LINE__));
  //handle_cuda_error(cudaStreamDestroy(s02), "ec " + write_with_ss(__LINE__));
  //handle_cuda_error(cudaDeviceSynchronize(), "sync " + write_with_ss(__LINE__));
}

bool kernel_manager_t::is_power_elementwise(einsummable_t e){
  if(e.inns.size()==1){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();

    size_t endIndex = op_str.find(")");


    if(op_str.substr(0,8)=="_pow(x0,"&&endIndex==op_str.size() - 1){
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
      return true;
    }

    if(op_str.substr(0,16)=="(f32|1e-06+(f32|"&&endIndex==op_str.size() - 5){
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
      return true;
    }
  }
  return false;
}


bool kernel_manager_t::is_type_conversion(einsummable_t e){
  scalarop_t op = e.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  if(op_str=="float16_t(x0)"||op_str=="float(x0)"||op_str=="double(x0)"){
    return true;
  }


  return false;


}

bool kernel_manager_t::is_c64_elementwise_multiply(einsummable_t e){
   if(e.inns.size()==2){
    scalarop_t op = e.join;

    op = op.simplify();

    auto op_str = op.to_cppstr();


    if(op_str=="(x0*x1)"&&e.inn_dtype(0)==dtype_t::c64&&e.inn_dtype(1)==dtype_t::c64){
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
    throw std::runtime_error("should not reach");
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
    throw std::runtime_error("should not reach");
  }
}

