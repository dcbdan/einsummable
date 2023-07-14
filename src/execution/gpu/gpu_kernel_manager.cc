#include "gpu_kernel_manager.h"

kernel_manager_t::kernel_manager_t()
{



}

optional<uint64_t> kernel_manager_t::build(einsummable_t const& e_)
{
  auto einsummable = e_.merge_adjacent_dims();

  if(kernels.count(einsummable) > 0) {
    return workspace_size(einsummable);
  }

  if(einsummable.is_contraction()) {
    auto c = contraction_t::make(einsummable);
    kernels.insert({einsummable,c});
    return c.worksize;
  }

  string err_msg =
    "could not build a kernel for einsummable_t: " + write_with_ss(e);

  if(einsummable.has_aggregation()) {
    if(einsummable.inns.size() != 1) {
      throw std::runtime_error(err_msg);
    }

    if(einsummable.castable.value() == castable_t::add) {
      // this is something cutensor reduction should be able to do
      vector<int> const& inn_modes = e.inns[0];

      auto inn_shape = einsummable.inn_shapes()[0];

      vector<int> out_modes(einsummable.out_rank);
      std::iota(out_modes.begin(), out_modes.end(), 0);

      // TODO: this is incorrect: also need to check that the join op
      //       is the identity!

      auto out_shape = einsummable.out_shape();

      auto reduct = build_cutensor_reduction(
        inn_modes, inn_shape,
        out_modes, out_shape,e.castable.value(),e.inn_dtype(0));
      
      auto const& [worksize, f] = reduct.value(); 

      kernels.insert({einsummable,
      reduction_t {
        .worksize = worksize,
        .kernel = f
      }
      });

      return worksize;
    }

    if(is_power_elementwise(einsummable)){
      double pow = einsummable.join.get_power();
      uint64_t size = einsummable.join_shape[0];
      auto lambda = [pow, size](cudaStream_t stream, float* out, const float* in){
        elementwise_power(out, in, stream, pow, size);
      }
      kernels.insert({einsummable,
      power_t{
        .power = pow,
        .kernel = lambda
      }
      });

      return 0;
    }

    throw std::runtime_error(err_msg);
  }

  if(is_type_conversion(e)){
    auto f = build_cutensor_type_conversion(e);

    kernels.insert({einsummable,
    type_conversion_t{
        .kernel = f
      }
    });
  }

  return std::nullopt;
}


uint64_t kernel_manager_t::workspace_size(einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  if(std::holds_alternative<contraction_t>(kernel)) {
    return std::get<contraction_t>(kernel).worksize;
  }else if(std::holds_alternative<reduction_t>(kernel)) {
    return std::get<reduction_t>(kernel).worksize;
  }else {
    return 0;
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
  cudastream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  auto const& info = get_built_kernel_info(e);
  call(info, stream, out, inns, maybe_workspace);
}

void kernel_manager_t::call(
  kernel_manager_t::kernel_info_t const& kernel,
  cudastream_t stream,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace)
{
  using std::holds_alternative;
  using std::get;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  if(holds_alternative<contraction_t>(kernel)) {
    assert_num_inputs(2);
    auto const& c = get<contraction_t>(kernel);
    if(c.worksize == 0) {
      execute_contraction(stream,handle,c.desc,out,inns,nullptr,0);
    } else if(!maybe_workspace) {
      throw std::runtime_error("workspace required; none given");
    } else {
      auto [workspace, wsz] = maybe_workspace.value();
      if(wsz < c.worksize) {
        throw std::runtime_error("provided workspace is too small");
      }
      execute_contraction(stream,handle,c.desc,out,inns,workspace,c.worksize);
    }
  }else if(holds_alternative<reduction_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [worksize,f] = get<reduction_t>(kernel);
    f(stream,handle,out,inns,worksize);
  }else if(holds_alternative<power_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [pow,f] = get<power_t>(kernel);
    f(stream,out,inns[0]);
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



contraction_t contraction_t::make(einsummable_t einsummable){
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
  desc,
  &find,
  CUTENSOR_WORKSPACE_RECOMMENDED,
  &worksize);


  contraction_t con {.worksize=worksize,.desc = desc,.dtype = einsummable.out_dtype()};

  return con;

}

bool is_power_elementwise(einsummable_t e){
  if((e.inns.size()==1)&&(e.join.is_unary)){
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


bool is_type_conversion(einsummable_t e){
  scalarop_t op = pow.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  if(op_str=="float16_t(x0)"||op_str=="float(x0)"||op_str=="double(x0)"){
    return true;
  }


  return false;


}


