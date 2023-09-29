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

/**
cutensor_kernel_t
build_einsummable(einsummable_t const& e_)
{
  einsummable_t e = e_.merge_adjacent_dims();

  if(e.is_contraction()) {
    throw std::runtime_error("build_einsummable must not be given a constraction");
  }

  string err_msg =
    "could not build a kernel for einsummable_t: " + write_with_ss(e);

  if(e.has_aggregation()) {
    if(e.inns.size() != 1) {
      throw std::runtime_error(err_msg);
    }

    if(e.castable.value() == castable_t::add) {
      // this is something cutensor reduction should be able to do
      vector<int> const& inn_modes = e.inns[0];

      auto inn_shape = e.inn_shapes()[0];

      vector<int> out_modes(e.out_rank);
      std::iota(out_modes.begin(), out_modes.end(), 0);

      // TODO: this is incorrect: also need to check that the join op
      //       is the identity!

      auto out_shape = e.out_shape();

      return build_cutensor_reduction(
        inn_modes, inn_shape,
        out_modes, out_shape,e.castable.value(),e.inn_dtype(0));
    }

    throw std::runtime_error(err_msg);
  }

  // is this something that cutensor elementwise can do?
  auto maybe_cutensor_ew = make_cutensor_elementwise_op(e);
  if(maybe_cutensor_ew) {
    return build_cutensor_elementwise(maybe_cutensor_ew.value());
  }

  // is this straight elementwise?
  if(e.is_straight_elementwise()) {
    return build_straight_elementwise(e.join, product(e.join_shape));
  }

  throw std::runtime_error(err_msg);
}

*/

cudaDataType_t dtype_to_cudatype(dtype_t type){
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

cutensorComputeType_t dtype_to_computetype(dtype_t type){
  if(type == dtype_t::f16){
    return CUTENSOR_COMPUTE_16F;
  }
  else if(type == dtype_t::f32){
    return CUTENSOR_COMPUTE_32F;
  }
  else if(type == dtype_t::f64){
    return CUTENSOR_COMPUTE_64F;
  }
  else if(type == dtype_t::c64){
    return CUTENSOR_COMPUTE_32F;
  }
  return CUTENSOR_COMPUTE_32F;
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


cutensor_kernel_t
build_cutensor_reduction(
  vector<int> inn_modes, vector<uint64_t> inn_shape,
  vector<int> out_modes, vector<uint64_t> out_shape,
  castable_t castable,dtype_t type)
{
  std::vector<int32_t> modeA(inn_modes.begin(),inn_modes.end());
  std::vector<int32_t> modeC(out_modes.begin(),out_modes.end());
  int32_t nmodeA = modeA.size();
  int32_t nmodeC = modeC.size();

  std::reverse(modeA.begin(), modeA.end());

  std::vector<int64_t> extent_A;
  extent_A.reserve(inn_shape.size());
  for (const auto& element : inn_shape) {
    extent_A.push_back(static_cast<int64_t>(element));
  }

  std::vector<int64_t> extent_C;
  extent_C.reserve(out_shape.size());
  for (const auto& element : out_shape) {
    extent_C.push_back(static_cast<int64_t>(element));
  }

  std::reverse(extent_A.begin(), extent_A.end());

  cutensorOperator_t opReduce;
  if(castable == castable_t::add) {
    opReduce = CUTENSOR_OP_ADD;
  } else if(castable == castable_t::mul) {
    opReduce = CUTENSOR_OP_MUL;
  } else if(castable == castable_t::min) {
    opReduce = CUTENSOR_OP_MIN;
  } else if(castable == castable_t::max) {
    opReduce = CUTENSOR_OP_MAX;
  } else {
    throw std::runtime_error("should not reach: missing castable");
  }

  size_t elementsA = 1;
  for (int i=0;i<inn_shape.size();++i){
    elementsA *= inn_shape[i];
  }
  size_t elementsC = 1;
  for (int i=0;i<out_shape.size();++i){
    elementsC *= out_shape[i];
  }

  size_t sizeA = sizeof(float) * elementsA;
  size_t sizeC = sizeof(float) * elementsC;
  
  cudaDataType_t typeA = dtype_to_cudatype(type);
  cudaDataType_t typeC = dtype_to_cudatype(type);
  cutensorComputeType_t typeCompute = dtype_to_computetype(type);

  return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,opReduce,sizeA,sizeC,type,typeA,typeC,typeCompute](
    cudaStream_t stream,
    cutensorHandle_t const* handle,
    void* out,
    vector<void const*> inns, void* work, uint64_t worksize)
  {
    

    //typedef float floatTypeCompute;
    //floatTypeCompute alpha = (floatTypeCompute)1.0f;
    //floatTypeCompute beta  = (floatTypeCompute)0.0f;

    

    void* ptr1;
    void* ptr2;
    float16_t alpha1, beta1;
    float alpha2, beta2;
    double alpha3, beta3;
    std::complex<float> alpha4(1.0f, 0.0f);
    std::complex<float> beta4(0.0f, 0.0f);

    if(type == dtype_t::f16){
      //alpha1 = float16_t(1.0f);
      //ptr1 = static_cast<void*>(&alpha1);
      //beta1 = float16_t(0.0f);
      //ptr2 = static_cast<void*>(&beta1);
      alpha2 = 1.0f;
      ptr1 = static_cast<void*>(&alpha2);
      beta2 = 0.0f;
      ptr2 = static_cast<void*>(&beta2);
    }
    else if(type == dtype_t::f32){
      alpha2 = 1.0f;
      ptr1 = static_cast<void*>(&alpha2);
      beta2 = 0.0f;
      ptr2 = static_cast<void*>(&beta2);
    }
    else if(type == dtype_t::f64){
      alpha3 = 1.0;
      ptr1 = static_cast<void*>(&alpha3);
      beta3 = 0.0;
      ptr2 = static_cast<void*>(&beta3);
    }
    else if(type == dtype_t::c64){
      ptr1 =  static_cast<void*>(&alpha4);
      ptr2 =  static_cast<void*>(&beta4);
    }

    void const* alpha = ptr1; 
    void const* beta = ptr2; 

    cutensorTensorDescriptor_t descA;
    handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    handle_cutensor_error(
      cutensorInitTensorDescriptor(handle,
                  &descC,
                  nmodeC,
                  extent_C.data(),
                  NULL /* stride */,
                  typeC, CUTENSOR_OP_IDENTITY));

    cutensorStatus_t err;
    err = cutensorReduction(handle, 
                alpha, inns[0], &descA, modeA.data(),
                beta,  out, &descC, modeC.data(), 
                       out, &descC, modeC.data(), 
                opReduce, typeCompute, work, worksize,stream);

    if(err != CUTENSOR_STATUS_SUCCESS)
            printf("ERROR: %s\n", cutensorGetErrorString(err) );
  };
}

uint64_t reduction_worksize(einsummable_t einsummable, void* out, vector<void const*> inns, cutensorHandle_t const* handle)
{
  castable_t castable = einsummable.castable.value();
  dtype_t type = einsummable.inn_dtype(0);

  vector<int> const& inn_modes = einsummable.inns[0];

  auto inn_shape = einsummable.inn_shapes()[0];

  vector<int> out_modes(einsummable.out_rank);
  std::iota(out_modes.begin(), out_modes.end(), 0);


  auto out_shape = einsummable.out_shape();

  std::vector<int32_t> modeA(inn_modes.begin(),inn_modes.end());
  std::vector<int32_t> modeC(out_modes.begin(),out_modes.end());
  int32_t nmodeA = modeA.size();
  int32_t nmodeC = modeC.size();

  std::reverse(modeA.begin(), modeA.end());

  std::vector<int64_t> extent_A;
  extent_A.reserve(inn_shape.size());
  for (const auto& element : inn_shape) {
    extent_A.push_back(static_cast<int64_t>(element));
  }

  std::vector<int64_t> extent_C;
  extent_C.reserve(out_shape.size());
  for (const auto& element : out_shape) {
    extent_C.push_back(static_cast<int64_t>(element));
  }

  std::reverse(extent_A.begin(), extent_A.end());

  cutensorOperator_t opReduce;
  if(castable == castable_t::add) {
    opReduce = CUTENSOR_OP_ADD;
  } else if(castable == castable_t::mul) {
    opReduce = CUTENSOR_OP_MUL;
  } else if(castable == castable_t::min) {
    opReduce = CUTENSOR_OP_MIN;
  } else if(castable == castable_t::max) {
    opReduce = CUTENSOR_OP_MAX;
  } else {
    throw std::runtime_error("should not reach: missing castable");
  }

  size_t elementsA = 1;
  for (int i=0;i<inn_shape.size();++i){
    elementsA *= inn_shape[i];
  }
  size_t elementsC = 1;
  for (int i=0;i<out_shape.size();++i){
    elementsC *= out_shape[i];
  }

  size_t sizeA = sizeof(float) * elementsA;
  size_t sizeC = sizeof(float) * elementsC;
  
  cudaDataType_t typeA = dtype_to_cudatype(type);
  cudaDataType_t typeC = dtype_to_cudatype(type);
  cutensorComputeType_t typeCompute = dtype_to_computetype(type);

  cutensorTensorDescriptor_t descA;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(handle,
                &descA,
                nmodeA,
                extent_A.data(),
                NULL /* stride */,
                typeA, CUTENSOR_OP_IDENTITY));

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(handle,
                &descC,
                nmodeC,
                extent_C.data(),
                NULL /* stride */,
                typeC, CUTENSOR_OP_IDENTITY));

  uint64_t worksize = 0;
  handle_cutensor_error(cutensorReductionGetWorkspaceSize(handle, 
                inns[0], &descA, modeA.data(),
                out, &descC, modeC.data(),
                out, &descC, modeC.data(),
                opReduce, typeCompute, &worksize));
  
  return worksize;
}

cutensor_elementwise_kernel_t
build_cutensor_elementwise(cutensor_elementwise_op_t op)
{
  typedef float floatTypeCompute;
  cudaDataType_t typeA = CUDA_R_32F;
  cudaDataType_t typeB = CUDA_R_32F;
  cudaDataType_t typeC = CUDA_R_32F;
  cudaDataType_t typeCompute = CUDA_R_32F;
  if(std::holds_alternative<cutensor_elementwise_op_t::unary_t>(op.op)){
    auto unary = std::get<cutensor_elementwise_op_t::unary_t>(op.op);

    std::vector<int> modeA = unary.arg.modes;
    int nmodeA = modeA.size();

    vector<int64_t> extent_A;
    for(auto const& mode: modeA) {
      extent_A.push_back(op.join_shape[mode]);
    }

    dtype_t type = unary.arg.scale.dtype;
    
    typeA = dtype_to_cudatype(unary.arg.scale.dtype);
    typeCompute = dtype_to_cudatype(unary.arg.scale.dtype);
 

    return [modeA,nmodeA,extent_A,type,typeA,typeCompute,unary]
    (cudaStream_t stream, cutensorHandle_t const* handle, void* out, vector<void const*> inns){
      void* ptr;
      float16_t alpha1;
      float alpha2;
      double alpha3;
      std::complex<float> alpha4(1.0f, 0.0f);

      if(type == dtype_t::f16){
        alpha1 = unary.arg.scale.f16();
        ptr = static_cast<void*>(&alpha1);
      }
      else if(type == dtype_t::f32){
        alpha2 = unary.arg.scale.f32();
        ptr = static_cast<void*>(&alpha2);
      }
      else if(type == dtype_t::f64){
        alpha3 = unary.arg.scale.f64();
        ptr = static_cast<void*>(&alpha3);
      }
      else if(type == dtype_t::c64){
        alpha4 = unary.arg.scale.c64();
        ptr =  static_cast<void*>(&alpha4);
      }

      void const* alpha = ptr;
      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, unary.arg.op));


      cutensorPermutation(handle,
                alpha, inns[0], &descA, modeA.data(),
                out, &descA, modeA.data(),
                typeCompute, stream);
    };
  }
  else if(std::holds_alternative<cutensor_elementwise_op_t::binary_t>(op.op)){
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
    typeCompute = dtype_to_elementwise_computetype(binary.lhs.scale.dtype);
    dtype_t type = binary.lhs.scale.dtype;


    //std::cout << modeA[0] << std::endl;
    //std::cout << modeA[1] << std::endl;
    //std::cout << modeC[0] << std::endl;
    //std::cout << modeC[1] << std::endl;

    //std::cout << extent_A[0] << std::endl;
    //std::cout << extent_A[1] << std::endl;
    //std::cout << extent_C[0] << std::endl;
    //std::cout << extent_C[1] << std::endl;

    //std::cout << nmodeA << std::endl;
    //std::cout << nmodeC << std::endl;
     
    return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,
    type,typeA,typeC,typeCompute,binary,swapped]
    (cudaStream_t stream, cutensorHandle_t const* handle, void* out, vector<void const*> inns){
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
      
      if(typeA==CUDA_R_32F&&typeC==CUDA_R_32F&&typeCompute==CUDA_R_32F&&binary.op==CUTENSOR_OP_ADD&&type==dtype_t::f32
      &&binary.lhs.op==CUTENSOR_OP_IDENTITY&&binary.rhs.op==CUTENSOR_OP_IDENTITY){
        //printf("HERE\n");
      }

      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, binary.lhs.op));

      cutensorTensorDescriptor_t descC;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descC,
                  nmodeC,
                  extent_C.data(),
                  NULL /* stride */,
                  typeC, binary.rhs.op));
      float alphayi = 1.0f;
      float betayi = 1.0f;
      //printf("HERERIN\n");


      if(swapped){
        handle_cutensor_error(cutensorElementwiseBinary(handle,
                  alpha, inns[1], &descA, modeA.data(),
                  beta, inns[0], &descC, modeC.data(),
                  out, &descC, modeC.data(),
                  binary.op, typeCompute, stream));
      }else{
        handle_cutensor_error(cutensorElementwiseBinary(handle,
                  alpha, inns[0], &descA, modeA.data(),
                  beta, inns[1], &descC, modeC.data(),
                  out, &descC, modeC.data(),
                  binary.op, typeCompute, stream));
      }
    };
  }
  else if(std::holds_alternative<cutensor_elementwise_op_t::ternary_t>(op.op)){
    auto ternary = std::get<cutensor_elementwise_op_t::ternary_t>(op.op);

    std::vector<int> modeA = ternary.a0.modes;
    std::vector<int> modeB = ternary.a1.modes;
    std::vector<int> modeC = ternary.a2.modes;
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    vector<int64_t> extent_A;
    for(auto const& mode: modeA) {
      extent_A.push_back(op.join_shape[mode]);
    }
    vector<int64_t> extent_B;
    for(auto const& mode: modeB) {
      extent_B.push_back(op.join_shape[mode]);
    }

    vector<int64_t> extent_C;
    for(auto const& mode: modeC) {
      extent_C.push_back(op.join_shape[mode]);
    }

    typeA = dtype_to_cudatype(ternary.a0.scale.dtype);
    typeB = dtype_to_cudatype(ternary.a1.scale.dtype);
    typeC = dtype_to_cudatype(ternary.a2.scale.dtype);
    typeCompute = dtype_to_cudatype(ternary.a0.scale.dtype);
    dtype_t type = ternary.a0.scale.dtype;
    
    
    return [
      modeA,modeB,modeC,nmodeA,nmodeB,nmodeC,
      extent_A,extent_B,extent_C,
      type,typeA,typeB,typeC,typeCompute,ternary]
      (
        cudaStream_t stream,
        cutensorHandle_t const* handle,
        void* out,
        vector<void const*> inns
      )
    {
      void* ptr1;
      void* ptr2;
      void* ptr3;
      float16_t alpha1, beta1, gamma1;
      float alpha2, beta2, gamma2;
      double alpha3, beta3, gamma3;
      std::complex<float> alpha4(1.0f, 0.0f);
      std::complex<float> beta4(1.0f, 0.0f);
      std::complex<float> gamma4(1.0f, 0.0f);

      if(type == dtype_t::f16){
        alpha1 = ternary.a0.scale.f16();
        ptr1 = static_cast<void*>(&alpha1);
        beta1 = ternary.a1.scale.f16();
        ptr2 = static_cast<void*>(&beta1);
        gamma1 = ternary.a2.scale.f16();
        ptr3 = static_cast<void*>(&gamma1);
      }
      else if(type == dtype_t::f32){
        alpha2 = ternary.a0.scale.f32();
        ptr1 = static_cast<void*>(&alpha2);
        beta2 = ternary.a1.scale.f32();
        ptr2 = static_cast<void*>(&beta2);
        gamma2 = ternary.a2.scale.f32();
        ptr3 = static_cast<void*>(&gamma2);
      }
      else if(type == dtype_t::f64){
        alpha3 = ternary.a0.scale.f64();
        ptr1 = static_cast<void*>(&alpha3);
        beta3 = ternary.a1.scale.f64();
        ptr2 = static_cast<void*>(&beta3);
        gamma3 = ternary.a2.scale.f64();
        ptr3 = static_cast<void*>(&gamma3);
      }
      else if(type == dtype_t::c64){
        alpha4 = ternary.a0.scale.c64();
        ptr1 =  static_cast<void*>(&alpha4);
        beta4 = ternary.a1.scale.c64();
        ptr2 =  static_cast<void*>(&beta4);
        gamma4 = ternary.a2.scale.c64();
        ptr3 = static_cast<void*>(&gamma4);
      }

      void const* alpha = ptr1; 
      void const* beta = ptr2; 
      void const* gamma = ptr3;
      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, ternary.a0.op));

      cutensorTensorDescriptor_t descB;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descB,
                  nmodeB,
                  extent_B.data(),
                  NULL /* stride */,
                  typeB, ternary.a1.op));

      cutensorTensorDescriptor_t descC;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descC,
                  nmodeC,
                  extent_C.data(),
                  NULL /* stride */,
                  typeC, ternary.a2.op));
      //if(typeA==CUDA_R_32F&&typeB==CUDA_R_32F&&typeC==CUDA_R_32F&&typeCompute==CUDA_R_32F){
      //  printf("HERE\n");
      //}
      //cutensorElementwiseTrinary(handle,
      //          alpha, inns[0], &descA, modeA.data(),
      //          beta , inns[1], &descB, modeB.data(),
      //          gamma, inns[2], &descC, modeC.data(),
      //                        out, &descC, modeC.data(),
      //          ternary.op_0_1, ternary.op_01_2, typeCompute, stream);
      cudaDeviceSynchronize();

      float alphayi = 1.0f;
      float betayi = 1.0f;

      cutensorElementwiseBinary(handle,
                (void*)&alphayi, inns[0], &descB, modeB.data(),
                (void*)&betayi , inns[0], &descB, modeB.data(),
                              out, &descC, modeC.data(),
                CUTENSOR_OP_ADD,  typeCompute, 0);
    };
  }

  return {};
}

cutensor_elementwise_kernel_t
cutensor_silu_elementwise(uint64_t size)
{
  cudaDataType_t typeA = CUDA_R_16F;
  cudaDataType_t typeC = CUDA_R_16F;
  cudaDataType_t typeCompute = CUDA_R_16F;

  std::vector<int> modeA = {0};
  std::vector<int> modeC = {0};

  int nmodeA = modeA.size();
  int nmodeC = modeC.size();

  vector<int64_t> extent_A = {static_cast<int64_t>(size)};
  vector<int64_t> extent_C = {static_cast<int64_t>(size)};

  return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,
  typeA,typeC,typeCompute]
  (cudaStream_t stream, cutensorHandle_t const* handle, void* out, vector<void const*> inns){

    void* ptr1;
    void* ptr2;
    float16_t alpha1 = float16_t(1.0);
    float16_t beta1 = float16_t(1.0);
    ptr1 = static_cast<void*>(&alpha1);
    ptr2 = static_cast<void*>(&beta1);
    void const* alpha = ptr1; 
    void const* beta = ptr2;

    
    cutensorTensorDescriptor_t descA;
    handle_cutensor_error(
      cutensorInitTensorDescriptor(handle,
                &descA,
                nmodeA,
                extent_A.data(),
                NULL /* stride */,
                typeA, CUTENSOR_OP_SIGMOID));

    cutensorTensorDescriptor_t descC;
    handle_cutensor_error(
      cutensorInitTensorDescriptor(handle,
                &descC,
                nmodeC,
                extent_C.data(),
                NULL /* stride */,
                typeC, CUTENSOR_OP_IDENTITY));
    
    handle_cutensor_error(cutensorElementwiseBinary(handle,
                  alpha, inns[0], &descA, modeA.data(),
                  beta, inns[0], &descC, modeC.data(),
                  out, &descC, modeC.data(),
                  CUTENSOR_OP_MUL, typeCompute, stream));


  };

}

cutensor_elementwise_op_t::arg_t convert_arg(cutensor_scalarop_t::arg_t arg, vector<int> modes){
  scalar_t scale = arg.scale;
  cutensorOperator_t op;
  if(arg.op==cutensor_scalarop_t::cop_t::exp){
    op = CUTENSOR_OP_EXP;
  }else if(arg.op==cutensor_scalarop_t::cop_t::identity){
    op = CUTENSOR_OP_IDENTITY;
  }else{
    throw std::runtime_error("Unary op not found");
  }
  cutensor_elementwise_op_t::arg_t new_arg {scale,op,modes};
  return new_arg;
}

cutensorOperator_t convert_op(cutensor_scalarop_t::cop_t op){
  cutensorOperator_t new_op;
  if(op==cutensor_scalarop_t::cop_t::add){
    new_op = CUTENSOR_OP_ADD;
  }else if(op==cutensor_scalarop_t::cop_t::mul){
    new_op = CUTENSOR_OP_MUL;
  }else{
    throw std::runtime_error("Binary op not found");
  }


  return new_op;
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

optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e)
{
  // TODO
  cutensor_elementwise_op_t op;
  op.join_shape = e.join_shape;

  if(e.inns.size()==1){
    scalarop_t join = e.join;

    auto potential_scalarop = join.compile_cutensor_scalarop();

    if(!potential_scalarop){
      return std::nullopt;
    }

    cutensor_scalarop_t cutensor_scalarop = *potential_scalarop;

    auto unary = std::get<cutensor_scalarop_t::unary_t>(cutensor_scalarop.op);

    cutensor_elementwise_op_t::unary_t unary_op {convert_arg(unary.arg,e.inns[0])};

    op.op = unary_op;

    return op;

  }else if(e.inns.size()==2){
    scalarop_t join = e.join;

    auto potential_scalarop = join.compile_cutensor_scalarop();

    if(!potential_scalarop){
      return std::nullopt;
    }

    cutensor_scalarop_t cutensor_scalarop = *potential_scalarop;

    auto binary = std::get<cutensor_scalarop_t::binary_t>(cutensor_scalarop.op);

    cutensor_elementwise_op_t::arg_t a0 = convert_arg(binary.lhs, e.inns[0]);
    cutensor_elementwise_op_t::arg_t a1 = convert_arg(binary.rhs, e.inns[1]);
    
    //if a0 is the same as output shape, that swap a0 to the a1 spot
    //if(isVectorSequential(e.inns[0],e.out_rank)){
    //  std::swap(a0, a1);
    //}
    
    cutensorOperator_t op_0_1 = convert_op(binary.op);

    cutensor_elementwise_op_t::binary_t bi_op{
      op_0_1,
      a0,
      a1
    };

    op.op = bi_op;

    return op;

  }else if(e.inns.size()==3){
    scalarop_t join = e.join;

    auto potential_scalarop = join.compile_cutensor_scalarop();

    if(!potential_scalarop){
      return std::nullopt;
    }

    cutensor_scalarop_t cutensor_scalarop = *potential_scalarop;
    
    auto ternary = std::get<cutensor_scalarop_t::ternary_t>(cutensor_scalarop.op);

    cutensor_elementwise_op_t::arg_t a0 = convert_arg(ternary.a0, e.inns[0]);
    cutensor_elementwise_op_t::arg_t a1 = convert_arg(ternary.a1, e.inns[1]);
    cutensor_elementwise_op_t::arg_t a2 = convert_arg(ternary.a2, e.inns[2]);
    cutensorOperator_t op_01_2 = convert_op(ternary.op_01_2);
    cutensorOperator_t op_0_1 = convert_op(ternary.op_0_1);

    cutensor_elementwise_op_t::ternary_t ter_op{
      op_01_2,
      op_0_1,
      a0,
      a1,
      a2
    };

    op.op = ter_op;

    return op;
    
    //op.op = cutensor_elementwise_op_t::ternary_t{e.castable,e.castable,{,,e.inns[0]},{,,e.inns[1]},{,,e.inns[2]}};

    //cutensor_elementwise_op_t op = {e.joinshape,{e.castable,e.castable,{,,e.inns[0]},{,,e.inns[1]},{,,e.inns[2]}}}
    //return op;

  }else{
    throw std::runtime_error("Invalid einsummable input.");
  }

  //(arg0 - val * arg1)
  // op.op = cutensor_elementwise_op_t::binary_t{/*/,{1.0f,CUTENSOR_OP_IDENTITY,e.inns[0]},{val,CUTENSOR_OP_IDENTITY,e.inns[1]}}

  //(relu(arg0))
  // op.op = cutensor_elementwise_op_t::unary_t{{1.0f,CUTENSOR_OP_RELU,e.inns[1]}}

  // max(arg0, arg1)
  // op.op = cutensor_elementwise_op_t::binary_t{CUTENSOR_OP_MAX,{1.0f,CUTENSOR_OP_IDENTITY,e.inns[0]},{1.0f,CUTENSOR_OP_IDENTITY,e.inns[1]}}
  //

  return std::nullopt;
}

cutensor_kernel_t
build_straight_elementwise(
  scalarop_t op,
  uint64_t size)
{
  // TODO: dispatch to canned elementwise kernels here
  throw std::runtime_error("build_straight_elementwise not implemented");
  return {};
}

cudaDataType_t dtypes_to_scalartype(dtype_t src, dtype_t dst){
  if(src == dtype_t::f64||dst == dtype_t::f64){
    return CUDA_R_64F;
  }
  else if(src == dtype_t::c64){
    return CUDA_C_32F;
  }
  return CUDA_R_32F;
}

cutensor_elementwise_kernel_t
build_cutensor_type_conversion(einsummable_t const& e){
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

  cudaDataType_t typeA = dtype_to_cudatype(src);
  cudaDataType_t typeC = dtype_to_cudatype(dst);
  cudaDataType_t typeCompute = dtypes_to_scalartype(src,dst);
  

  

  return [modeA,typeCompute, nmodeA,nmodeC,
  extent_A, extent_C, typeA, typeC]
    (
      cudaStream_t stream,
      cutensorHandle_t const* handle,
      void* out,
      vector<void const*> inns
    )
  {
    void* ptr;
    float alpha2;
    double alpha3;

    if(typeCompute == CUDA_R_32F){
      alpha2 = 1.0f;
      ptr = static_cast<void*>(&alpha2);
    }
    else if(typeCompute == CUDA_R_64F){
      alpha3 = 1.0;
      ptr = static_cast<void*>(&alpha3);
    }

    void const* alpha = ptr;

    cutensorTensorDescriptor_t descA;
    cutensorInitTensorDescriptor(handle,
              &descA,
              nmodeA,
              extent_A.data(),
              NULL /* stride */,
              typeA, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descC;
    cutensorInitTensorDescriptor(handle,
              &descC,
              nmodeA,
              extent_A.data(),
              NULL /* stride */,
              typeC, CUTENSOR_OP_IDENTITY);
    cutensorStatus_t err;
    err =  cutensorPermutation(handle,
              alpha, inns[0], &descA, modeA.data(),
              out, &descC, modeA.data(),
              typeCompute, stream);
    
    if(err != CUTENSOR_STATUS_SUCCESS)
            printf("ERROR: %s\n", cutensorGetErrorString(err) );



  };


}



cutensor_kernel_t
build_elementwise_and_pow(cutensor_elementwise_op_t op, uint64_t a_size){
  cudaDataType_t typeA = CUDA_R_32F;
  cudaDataType_t typeB = CUDA_R_32F;
  cudaDataType_t typeC = CUDA_R_32F;
  cudaDataType_t typeCompute = CUDA_R_32F;


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
  typeCompute = dtype_to_cudatype(binary.lhs.scale.dtype);
  dtype_t type = binary.lhs.scale.dtype;

    
  return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,
  type,typeA,typeC,typeCompute,binary,swapped,a_size]
    (
      cudaStream_t stream,
      cutensorHandle_t const* handle,
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
        printf("here!\n");
      }

      void const* alpha = ptr1; 
      void const* beta = ptr2;
      
      if(typeA==CUDA_R_32F&&typeC==CUDA_R_32F&&typeCompute==CUDA_R_32F&&binary.op==CUTENSOR_OP_ADD&&type==dtype_t::f32
      &&binary.lhs.op==CUTENSOR_OP_IDENTITY&&binary.rhs.op==CUTENSOR_OP_IDENTITY){
        //printf("HERE\n");
      }

      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, binary.lhs.op));

      cutensorTensorDescriptor_t descC;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descC,
                  nmodeC,
                  extent_C.data(),
                  NULL /* stride */,
                  typeC, binary.rhs.op));
      float alphayi = 1.0f;
      float betayi = 1.0f;
      //printf("HERERIN\n");


      if(swapped){
        handle_cutensor_error(cutensorElementwiseBinary(handle,
                  alpha, work, &descA, modeA.data(),
                  beta, inns[0], &descC, modeC.data(),
                  out, &descC, modeC.data(),
                  binary.op, typeCompute, stream));
      }else{
        handle_cutensor_error(cutensorElementwiseBinary(handle,
                  alpha, inns[0], &descA, modeA.data(),
                  beta, work, &descC, modeC.data(),
                  out, &descC, modeC.data(),
                  binary.op, typeCompute, stream));
      }
  };


}


