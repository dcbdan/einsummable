#include "gpu_kernel_manager.h"

#include "utility.h"

build_result_t kernel_manager_t::build(einsummable_t const& e_)
{
  auto einsummable = e_.merge_adjacent_dims();

  if(kernels.count(einsummable) > 0) {
    auto worksize = workspace_size(einsummable);
    //if worksize doesn't exist, it means that the kernel is a reduction kernel
    if(worksize){
      build_result_t repeat_result{.built = true,.workspace_size = worksize};
      return repeat_result;
    }else{
      build_result_t reduction_result{.built = true};
      return reduction_result;
    }
  }

  if(einsummable.is_contraction()) {
    auto c = contraction_t::make(einsummable);
    kernels.insert({einsummable,c});
    build_result_t result_contraction{
      .built = true, 
      .workspace_size = c.worksize};
    return result_contraction;
  }

  string err_msg =
    "could not build a kernel for einsummable_t: " 
    + write_with_ss(einsummable);


  // Check for Reudctions
  if(einsummable.has_aggregation()) {
    if(einsummable.inns.size() != 1) {
      throw std::runtime_error(err_msg);
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

      build_result_t reduction_result{.built = true};
      return reduction_result;
    }

    

    throw std::runtime_error(err_msg);
  }

  if(is_power_elementwise(einsummable)){

    double pow = get_power(einsummable);
    uint64_t size = einsummable.join_shape[0];
    auto lambda = [pow, size]
    (cudaStream_t stream, float* out, const float* in){
      elementwise_power(out, in, stream, pow, size);
    };

    power_t power_kernel {pow, lambda};

    kernels.insert({einsummable, power_kernel});

    build_result_t result_power{
      .built = true, 
      .workspace_size = 0};
    return result_power;
  }

  if(is_type_conversion(einsummable)){

    auto f = build_cutensor_type_conversion(einsummable);

    type_conversion_t conversion_kernel {f};

    kernels.insert({einsummable, conversion_kernel});
    build_result_t result_conversion{
      .built = true, 
      .workspace_size = 0};
    return result_conversion;
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

    build_result_t result_scale {
      .built = true,
      .workspace_size = 0};
    
    return result_scale;
  }

  if(is_custom_kernel1(einsummable)){
    uint64_t size = einsummable.join_shape[0];

    auto lambda = cutensor_silu_elementwise(size);

    custom_kernel_1_t custom_kernel {lambda};

    kernels.insert({einsummable, custom_kernel});

    build_result_t result_custom_1 {
      .built = true, 
      .workspace_size = 0};
    return result_custom_1;
  }


  if(is_elementwise_with_pow(einsummable)){
    uint64_t a_size = einsummable.join_shape[0];
    cutensor_elementwise_op_t op_ele = make_mul_op(einsummable);
    auto func_elem = build_elementwise_and_pow(op_ele, a_size);

    uint64_t worksize = a_size*sizeof(float);

    pow_and_elementwise_t pow_ane_ele_kernel {func_elem, worksize, a_size};

    kernels.insert({einsummable, pow_ane_ele_kernel});
    build_result_t result_pow_ele {
      .built = true, 
      .workspace_size = worksize};
    return result_pow_ele;
  }

  if(is_c64_elementwise_multiply(einsummable)){
    auto c = contraction_t::make(einsummable);
    kernels.insert({einsummable,c});
    build_result_t result_contraction{
      .built = true, 
      .workspace_size = c.worksize};
    return result_contraction;
  }


  auto maybe = make_cutensor_elementwise_op(einsummable); 

  if(maybe&&einsummable.out_dtype()!=dtype_t::c64){

    cutensor_elementwise_op_t op = *maybe;

    auto func = build_cutensor_elementwise(op);

    elementwise_t elementwise_kernel {func};


    kernels.insert({einsummable, elementwise_kernel});
    build_result_t result_elementwise{
      .built = true, 
      .workspace_size = 0};
    return result_elementwise;

  }

  build_result_t failed_result{.built=false};

  return failed_result;
}


optional<uint64_t> kernel_manager_t::workspace_size(
  einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  if(std::holds_alternative<contraction_t>(kernel)) {
    return std::get<contraction_t>(kernel).worksize;
  }else if(std::holds_alternative<pow_and_elementwise_t>(kernel)) {
    return std::get<pow_and_elementwise_t>(kernel).worksize;
  }else if(std::holds_alternative<reduction_t>(kernel)) {
    return std::nullopt;
  }else {
    return 0;
  }

  return std::nullopt;
}

uint64_t kernel_manager_t::workspace_size(
  einsummable_t const& e, void* out,
  vector<void const*> inns, 
  cutensorHandle_t const* handle) const
{
  
  auto const& kernel = get_built_kernel_info(e);


  if(std::holds_alternative<reduction_t>(kernel)) {
    return reduction_worksize(e, out, inns, handle);
  }
  return 0;
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
  optional<tuple<void*, uint64_t>> maybe_workspace)
{
  using std::holds_alternative;
  using std::get;

  cutensorHandle_t* handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  auto assert_num_inputs = [&inns](int n) {
    if(inns.size() != n) {
      throw std::runtime_error("kernel manager: incorrect number of input tensors");
    }
  };

  if(holds_alternative<contraction_t>(kernel)) {
    assert_num_inputs(2);
    auto const& c = get<contraction_t>(kernel);
    if(c.worksize == 0) {
      execute_contraction(stream,handle,&c.desc,
        out,inns[0],inns[1],c.dtype,nullptr,0);
    } else if(!maybe_workspace) {
      throw std::runtime_error("workspace required; none given");
    } else {
      auto [workspace, wsz] = maybe_workspace.value();
      if(wsz < c.worksize) {
        throw std::runtime_error("provided workspace is too small");
      }
      execute_contraction(stream,handle,&c.desc,
        out,inns[0],inns[1],c.dtype,workspace,c.worksize);
    }
  }else if(holds_alternative<reduction_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [f] = get<reduction_t>(kernel);
    auto [workspace, wsz] = maybe_workspace.value();

    f(stream,handle,out,inns,workspace,wsz);
  }else if(holds_alternative<power_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [pow,f] = get<power_t>(kernel);
    f(stream,(float*)out,(float*)inns[0]);
  }else if(holds_alternative<type_conversion_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [f] = get<type_conversion_t>(kernel);
    f(stream,handle,out,inns);
  }else if(holds_alternative<elementwise_t>(kernel)) {
    auto const& [f] = get<elementwise_t>(kernel);
    f(stream,handle,out,inns);
  }else if(holds_alternative<scale_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [scale,f] = get<scale_t>(kernel);
    f(stream,(float*)out,(float*)inns[0]);
  }else if(holds_alternative<pow_and_elementwise_t>(kernel)) {
    auto const& [f, worksizep, a_size] = get<pow_and_elementwise_t>(kernel);
    auto [workspace, wsz] = maybe_workspace.value();

    f(stream,handle,out,inns,workspace,wsz);
  }else if(holds_alternative<custom_kernel_1_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [f] = get<custom_kernel_1_t>(kernel);
    f(stream,handle,out,inns);
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



contraction_t contraction_t::make(einsummable_t const&  einsummable){
  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  build_contraction(&desc,handle,einsummable);

  cutensorContractionFind_t find;

  cutensorInitContractionFind(
      handle, &find,
      CUTENSOR_ALGO_DEFAULT);
  
  uint64_t worksize = 0;

  cutensorContractionGetWorkspaceSize(handle,
  &desc,
  &find,
  CUTENSOR_WORKSPACE_RECOMMENDED,
  &worksize);

  contraction_t con {.worksize=worksize,.desc = desc,.dtype = einsummable.out_dtype()};

  return con;

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


