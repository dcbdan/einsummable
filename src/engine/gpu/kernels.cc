#include "kernels.h"
#include "cuda_kernels.h"

void touch1(touchdim_t const& t0, void* out,
            void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch1_dispatch(out, inn, t0.offset_inn,
  t0.offset_out, t0.size, t0.d_inn, t0.d_out,
  stream, choice, dtype_info);
}

void touch2(touchdim_t const& t0, touchdim_t const& t1,
            void* out, void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch2_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t0.offset_out, t1.offset_out,
  t0.size, t1.size, t1.d_inn, t1.d_out,
  stream, choice, dtype_info);
}

void touch3(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, void* out, void const* inn,
            cudaStream_t stream, int choice, int dtype_info) {
  touch3_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t2.offset_inn, t0.offset_out,
  t1.offset_out, t2.offset_out,t0.size,
  t1.size, t2.size, t1.d_inn, t1.d_out,
  t2.d_inn, t2.d_out, stream, choice, dtype_info);
}

void touch4(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, touchdim_t const& t3,
            void* out, void const* inn, cudaStream_t stream,
            int choice, int dtype_info) {
  touch4_dispatch(out, inn, t0.offset_inn, t1.offset_inn,
  t2.offset_inn, t3.offset_inn,t0.offset_out, t1.offset_out,
  t2.offset_out, t3.offset_out,t0.size, t1.size, t2.size,
  t3.size, t1.d_inn, t1.d_out, t2.d_inn, t2.d_out, t3.d_inn,
  t3.d_out, stream, choice, dtype_info);
}

#define _touch_lambda_1(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch1(ts[0], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_2(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch2(ts[0], ts[1], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_3(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice, dtype_info); \
}

#define _touch_lambda_4(choice) \
  [ts, dtype_info](cudaStream_t stream, void* out, const void* inn) -> void { \
    touch4(ts[0], ts[1], ts[2], ts[3],out, inn, stream, choice, dtype_info); \
}


#define _touch_dispatch(i) \
  [&]() -> touch_kernel_t { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_##i(1); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_##i(2); \
      } else if(c == castable_t::min) { \
        return _touch_lambda_##i(3); \
      } else if(c == castable_t::max) { \
        return  _touch_lambda_##i(4); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_##i(0); \
    } \
  }()

touch_kernel_t build_touch(touch_t const& touch_)
{
  touch_t touch = touch_.simplify();

  auto const& ts = touch.selection;

  int dtype_info = 0;

  auto const& dtype = touch.dtype;

  if(dtype == dtype_t::f32) {
    dtype_info = 1;
  } else if(dtype == dtype_t::f64) {
    dtype_info = 2;
  } else if(dtype == dtype_t::c64) {
    dtype_info = 3;
  }

  if(ts.size() == 1) {
    return _touch_dispatch(1);
  }

  if(ts.size() == 2) {
    return _touch_dispatch(2);
  }

  if(ts.size() == 3) {
    return _touch_dispatch(3);
  }

  if(ts.size() == 4) {
    return _touch_dispatch(4);
  }

  throw std::runtime_error("touch kernel not implemented");
}

void launch_touch_kernel(
  touch_t const& touch_,
  cudaStream_t stream,
  void* out,
  void const* inn)
{
  touch_t touch = touch_.simplify();

  auto const& ts = touch.selection;

  int dtype_info = 0;

  auto const& dtype = touch.dtype;

  if(dtype == dtype_t::f32) {
    dtype_info = 1;
  } else if(dtype == dtype_t::f64) {
    dtype_info = 2;
  } else if(dtype == dtype_t::c64) {
    dtype_info = 3;
  }

  int choice = 0;
  if(touch.castable) { 
    castable_t const& c = touch.castable.value(); 
    if(c == castable_t::add) { 
      choice = 1;
    } else if(c == castable_t::mul) { 
      choice = 2;
    } else if(c == castable_t::min) { 
      choice = 3;
    } else if(c == castable_t::max) { 
      choice = 4;
    } else { 
      throw std::runtime_error("castable should not reach"); 
    } 
  }

  if(ts.size() == 1) {
    touch1(ts[0], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 2) {
    touch2(ts[0], ts[1], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 3) {
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice, dtype_info);
  } else if(ts.size() == 4) {
    touch4(ts[0], ts[1], ts[2], ts[3], out, inn, stream, choice, dtype_info);
  } else {
    throw std::runtime_error("touch kernel not implemented");
  }
}

cutensorDataType_t dtype_to_cudatype(dtype_t type){
  if(type == dtype_t::f16){
    return CUTENSOR_R_16F;
  }
  else if(type == dtype_t::f32){
    return CUTENSOR_R_32F;
  }
  else if(type == dtype_t::f64){
    return CUTENSOR_R_64F;
  }
  else if(type == dtype_t::c64){
    return CUTENSOR_C_32F;
  }
  return CUTENSOR_R_32F;
}

cutensorComputeDescriptor_t dtype_to_computetype(dtype_t type){
  if(type == dtype_t::f16){
    return CUTENSOR_COMPUTE_DESC_16F;
  }
  else if(type == dtype_t::f32){
    return CUTENSOR_COMPUTE_DESC_32F;
  }
  else if(type == dtype_t::f64){
    return CUTENSOR_COMPUTE_DESC_64F;
  }
  else if(type == dtype_t::c64){
    return CUTENSOR_COMPUTE_DESC_32F;
  }

  return CUTENSOR_COMPUTE_DESC_32F;
}

cudaDataType_t dtype_to_elementwise_computetype(dtype_t type){
  if(type == dtype_t::f16){
    return CUDA_R_16F;
  }
  else if(type == dtype_t::f32){
    return CUDA_R_32F;
  }
  else if(type == dtype_t::f64){
    return CUDA_R_64F;
  }
  else if(type == dtype_t::c64){
    return CUDA_C_32F;
  }
  return CUDA_R_32F;
}

cutensor_elementwise_kernel_t
cutensor_silu_elementwise(uint64_t size)
{
  cutensorDataType_t typeA = CUTENSOR_R_16F;
  cutensorDataType_t typeC = CUTENSOR_R_16F;
  cutensorComputeDescriptor_t typeCompute = CUTENSOR_COMPUTE_DESC_16F;
  cutensorHandle_t handle;
  handle_cutensor_error(cutensorCreate(&handle));
  uint32_t const kAlignment = 128;

  std::vector<int> modeA = {0};
  std::vector<int> modeC = {0};

  int nmodeA = modeA.size();
  int nmodeC = modeC.size();

  vector<int64_t> extent_A = {static_cast<int64_t>(size)};
  vector<int64_t> extent_C = {static_cast<int64_t>(size)};

  
  cutensorTensorDescriptor_t  descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descA, nmodeA, extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t  descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descC, nmodeC, extent_C.data(),
                                              nullptr /* stride */,
                                              typeC,
                                              kAlignment));

  cutensorOperationDescriptor_t  desc;
  
  handle_cutensor_error(cutensorCreateElementwiseBinary(handle, &desc,
                                                  descA, modeA.data(), /* unary operator A  */ CUTENSOR_OP_SIGMOID,
                                                  descC, modeC.data(), /* unary operator C  */ CUTENSOR_OP_IDENTITY,
                                                  descC, modeC.data(), /* unary operator AC */ CUTENSOR_OP_MUL,
                                                  typeCompute));
  
  

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t  planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));


  cutensorPlan_t  plan;
  handle_cutensor_error(cutensorCreatePlan(handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));

  return [plan]
  (cudaStream_t stream, cutensorHandle_t handle, void* out, vector<void const*> inns){

    void* ptr1;
    void* ptr2;
    float16_t alpha1 = float16_t(1.0);
    float16_t beta1 = float16_t(1.0);
    ptr1 = static_cast<void*>(&alpha1);
    ptr2 = static_cast<void*>(&beta1);
    void const* alpha = ptr1; 
    void const* beta = ptr2;

    
    handle_cutensor_error(cutensorElementwiseBinaryExecute(handle,
                  plan, alpha, inns[0], 
                  beta, inns[0], 
                  out, stream));
  };
}

bool isVectorSequential(const std::vector<int>& vec, int n) {
    if (vec.size() != n) {
        return false;  
    }
    for (int i = 0; i < n; ++i) {
        if (vec[i] != i) {
            return false;  
        }
    }

    return true;  
}

cutensor_elementwise_op_t make_mul_op(
  einsummable_t const& e)
{
  cutensor_elementwise_op_t op;
  op.join_shape = e.join_shape;

  cutensor_elementwise_op_t::arg_t a0 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,e.inns[0]};
  cutensor_elementwise_op_t::arg_t a1 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,e.inns[1]};

  cutensorOperator_t op_0_1 = CUTENSOR_OP_MUL;

  cutensor_elementwise_op_t::binary_t bi_op{
    op_0_1,
    a0,
    a1
  };

  op.op = bi_op;

  return op;
}


cutensorComputeDescriptor_t dtypes_to_scalartype(dtype_t src, dtype_t dst){
  if(src == dtype_t::f64||dst == dtype_t::f64){
    return CUTENSOR_COMPUTE_DESC_64F;
  }
  else if(src == dtype_t::c64){
    return CUTENSOR_COMPUTE_DESC_32F;
  }
  return CUTENSOR_COMPUTE_DESC_32F;
}

cutensor_elementwise_kernel_t
build_cutensor_type_conversion(einsummable_t const& e){
  cutensorDataType_t typeA = CUTENSOR_R_32F;
  cutensorDataType_t typeB = CUTENSOR_R_32F;
  cutensorDataType_t typeC = CUTENSOR_R_32F;
  cutensorComputeDescriptor_t typeCompute = CUTENSOR_COMPUTE_DESC_32F;
  cutensorHandle_t handle;
  handle_cutensor_error(cutensorCreate(&handle));
  uint32_t const kAlignment = 128;


  dtype_t src = e.inn_dtype(0);
  dtype_t dst = e.out_dtype();

  
  std::vector<int> modeA = e.inns[0];
  std::vector<int> modeC = modeA;

  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(e.join_shape[mode]);
  }

  int nmodeA = modeA.size();
  int nmodeC = modeC.size();

  vector<int64_t> extent_C = extent_A;

  typeA = dtype_to_cudatype(src);
  typeC = dtype_to_cudatype(dst);
  //cudaDataType_t typeCompute = dtypes_to_scalartype(src,dst);
  typeCompute = dtypes_to_scalartype(src,dst);

  //if(typeA==CUTENSOR_R_16F&&typeC==CUTENSOR_R_32F&&typeCompute==CUTENSOR_COMPUTE_DESC_32F){
  //  printf("the type is correct!");
  //}
  
  
  cutensorTensorDescriptor_t  descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descA,
                                              nmodeA,
                                              extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t  descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descC,
                                              nmodeC,
                                              extent_C.data(),
                                              nullptr /* stride */,
                                              typeC,
                                              kAlignment));
  
  cutensorOperationDescriptor_t  desc;
    handle_cutensor_error(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           typeCompute));


  
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t  planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));


  cutensorPlan_t  plan;
  handle_cutensor_error(cutensorCreatePlan(handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));

  return [typeCompute, plan]
    (
      cudaStream_t stream,
      cutensorHandle_t handle,
      void* out,
      vector<void const*> inns
    )
  {
    void* ptr;
    float alpha2;
    double alpha3;

    if(typeCompute == CUTENSOR_COMPUTE_DESC_32F){
      alpha2 = 1.0f;
      ptr = static_cast<void*>(&alpha2);
    }
    else if(typeCompute == CUTENSOR_COMPUTE_DESC_64F){
      alpha3 = 1.0;
      ptr = static_cast<void*>(&alpha3);
    }

    void const* alpha = ptr;

    handle_cutensor_error(cutensorPermute(handle,
                        plan,
                        alpha, inns[0], out, stream /* stream */));



  };
}

cutensor_kernel_t
build_elementwise_and_pow(cutensor_elementwise_op_t op, uint64_t a_size){
  cutensorDataType_t typeA = CUTENSOR_R_32F;
  cutensorDataType_t typeB = CUTENSOR_R_32F;
  cutensorDataType_t typeC = CUTENSOR_R_32F;
  cutensorComputeDescriptor_t typeCompute = CUTENSOR_COMPUTE_DESC_32F;
  cutensorHandle_t handle;
  handle_cutensor_error(cutensorCreate(&handle));
  uint32_t const kAlignment = 128;


  auto binary = std::get<cutensor_elementwise_op_t::binary_t>(op.op);

  std::vector<int> modeA = binary.lhs.modes;
  std::vector<int> modeC = binary.rhs.modes;

  bool swapped = (modeA.size() > modeC.size());

  

  if(swapped){
    std::swap(modeA, modeC);
    std::swap(binary.lhs, binary.rhs);
  }

  

  int nmodeA = modeA.size();
  int nmodeC = modeC.size();

  std::reverse(modeA.begin(), modeA.end());
  std::reverse(modeC.begin(), modeC.end());



  vector<int64_t> extent_A;
  for(auto const& mode: modeA) {
    extent_A.push_back(op.join_shape[mode]);
  }
  vector<int64_t> extent_C;
  for(auto const& mode: modeC) {
    extent_C.push_back(op.join_shape[mode]);
  }


  typeA = dtype_to_cudatype(binary.lhs.scale.dtype);
  typeC = dtype_to_cudatype(binary.rhs.scale.dtype);
  typeCompute = dtype_to_computetype(binary.lhs.scale.dtype);
  dtype_t type = binary.lhs.scale.dtype;

  
  cutensorTensorDescriptor_t  descA;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descA, nmodeA, extent_A.data(),
                                              nullptr /* stride */,
                                              typeA,
                                              kAlignment));

  cutensorTensorDescriptor_t  descC;
  handle_cutensor_error(cutensorCreateTensorDescriptor(handle,
                                              &descC, nmodeC, extent_C.data(),
                                              nullptr /* stride */,
                                              typeC,
                                              kAlignment));

  cutensorOperationDescriptor_t  desc;
  
  handle_cutensor_error(cutensorCreateElementwiseBinary(handle, &desc,
                                                  descA, modeA.data(), /* unary operator A  */ binary.lhs.op,
                                                  descC, modeC.data(), /* unary operator C  */ binary.rhs.op,
                                                  descC, modeC.data(), /* unary operator AC */ binary.op,
                                                  typeCompute));
  
  

  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t  planPref;
  handle_cutensor_error(cutensorCreatePlanPreference(handle,
                                            &planPref,
                                            algo,
                                            CUTENSOR_JIT_MODE_NONE));


  cutensorPlan_t  plan;
  handle_cutensor_error(cutensorCreatePlan(handle,
                                  &plan,
                                  desc,
                                  planPref,
                                  0 /* workspaceSizeLimit */));

  return [plan,binary,swapped,a_size,type]
    (
      cudaStream_t stream,
      cutensorHandle_t handle,
      void* out,
      vector<void const*> inns, 
      void* work, uint64_t worksize
    )
  {
    double pow = -1.0;
    elementwise_power((float*)work,(float*)inns[1],stream,pow,a_size);
    void* ptr1;
    void* ptr2;
    float16_t alpha1, beta1;
    float alpha2, beta2;
    double alpha3, beta3;
    std::complex<float> alpha4(1.0f, 0.0f);
    std::complex<float> beta4(1.0f, 0.0f);

    if(type == dtype_t::f16){
      alpha1 = binary.lhs.scale.f16();
      ptr1 = static_cast<void*>(&alpha1);
      beta1 = binary.rhs.scale.f16();
      ptr2 = static_cast<void*>(&beta1);
      //printf("why is it here\n");
    }
    else if(type == dtype_t::f32){
      alpha2 = binary.lhs.scale.f32();
      ptr1 = static_cast<void*>(&alpha2);
      beta2 = binary.rhs.scale.f32();
      ptr2 = static_cast<void*>(&beta2);
    }
    else if(type == dtype_t::f64){
      alpha3 = binary.lhs.scale.f64();
      ptr1 = static_cast<void*>(&alpha3);
      beta3 = binary.rhs.scale.f64();
      ptr2 = static_cast<void*>(&beta3);
    }
    else if(type == dtype_t::c64){
      alpha4 = binary.lhs.scale.c64();
      ptr1 =  static_cast<void*>(&alpha4);
      beta4 = binary.rhs.scale.c64();
      ptr2 =  static_cast<void*>(&alpha2);
    }

    void const* alpha = ptr1; 
    void const* beta = ptr2;
    

    if(swapped){
      handle_cutensor_error(cutensorElementwiseBinaryExecute(handle,
                plan, alpha, work, 
                beta, inns[0], 
                out, stream));
    }else{
      handle_cutensor_error(cutensorElementwiseBinaryExecute(handle,
                plan, alpha, inns[0], 
                beta, work, 
                out, stream));
    }
  };
  
}
