#include "kernels.h"
#include "cuda_kernels.h"

void touch1(touchdim_t const& t0, float* out,
            float const* inn, cudaStream_t stream,
            int choice) {
  touch1_dispatch(out, inn, t0.offset_inn,
  t0.offset_out, t0.size, t0.d_inn, t0.d_out,
  stream, choice);
}

void touch2(touchdim_t const& t0, touchdim_t const& t1,
            float* out, float const* inn, cudaStream_t stream,
            int choice) {
  touch2_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t0.offset_out, t1.offset_out,
  t0.size, t1.size, t1.d_inn, t1.d_out,
  stream, choice);
}

void touch3(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, float* out, float const* inn,
            cudaStream_t stream, int choice) {
  touch3_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t2.offset_inn, t0.offset_out,
  t1.offset_out, t2.offset_out,t0.size,
  t1.size, t2.size, t1.d_inn, t1.d_out,
  t2.d_inn, t2.d_out, stream, choice);
}

void touch4(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, touchdim_t const& t3,
            float* out, float const* inn, cudaStream_t stream,
            int choice) {
  touch4_dispatch(out, inn, t0.offset_inn, t1.offset_inn,
  t2.offset_inn, t3.offset_inn,t0.offset_out, t1.offset_out,
  t2.offset_out, t3.offset_out,t0.size, t1.size, t2.size,
  t3.size, t1.d_inn, t1.d_out, t2.d_inn, t2.d_out, t3.d_inn,
  t3.d_out, stream, choice);
}

#define _touch_lambda_1(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch1(ts[0], out, inn, stream, choice); \
}

#define _touch_lambda_2(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch2(ts[0], ts[1], out, inn, stream, choice); \
}

#define _touch_lambda_3(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice); \
}

#define _touch_lambda_4(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch4(ts[0], ts[1], ts[2], ts[3],out, inn, stream, choice); \
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

void build_contraction(
  cutensorContractionDescriptor_t* desc,
  cutensorHandle_t const* handle,
  einsummable_t const& e_)
{
  einsummable_t e = e_.merge_adjacent_dims();

  if(!e.is_contraction()) {
    throw std::runtime_error("build_contraction must be given a contraction");
  }

  std::vector<int> modeA = e.inns[0];
  std::vector<int> modeB = e.inns[1];
  std::vector<int> modeC;
  for (int i = 0; i < e.out_rank; i++) {
    modeC.push_back(i);
  }

  int nmodeA = e.inns[0].size();
  int nmodeB = e.inns[1].size();
  int nmodeC = e.out_rank;

  // CUDA types
  dtype_t type = e.inn_dtype(0);
  cudaDataType_t typeTensor = dtype_to_cudatype(type);

  cutensorComputeType_t typeCompute = dtype_to_computetype(type);

  // extent = size of each dimension
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
  cutensorTensorDescriptor_t descA;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      handle,
      &descA,
      nmodeA,
      extent_A.data(),
      NULL,/*stride*/
      typeTensor, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descB;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      handle,
      &descB,
      nmodeB,
      extent_B.data(),
      NULL,/*stride*/
      typeTensor, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descC;
  handle_cutensor_error(
    cutensorInitTensorDescriptor(
      handle,
      &descC,
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
      handle,
      desc,
      &descA, modeA.data(), alignmentRequirementA,
      &descB, modeB.data(), alignmentRequirementB,
      &descC, modeC.data(), alignmentRequirementC,
      &descC, modeC.data(), alignmentRequirementC,
      typeCompute) );
}

void execute_contraction(
  cudaStream_t stream,
  cutensorHandle_t const* handle,
  cutensorContractionDescriptor_t const* desc,
  void* out,
  void const* lhs,
  void const* rhs,
  void const* alpha)
{
  // Set the algorithm to use
  cutensorContractionFind_t find;
  handle_cutensor_error(
    cutensorInitContractionFind(
      handle, &find,
      CUTENSOR_ALGO_DEFAULT) );

  
  size_t worksize = 0;
  void *work = nullptr;


  cutensorContractionPlan_t plan;
  handle_cutensor_error(
    cutensorInitContractionPlan(
      handle,
      &plan,
      desc,
      &find,
      0) );

  cutensorStatus_t err;

  handle_cutensor_error(
    cutensorContraction(handle, &plan, alpha, lhs,
                        rhs, alpha, out, out,
                        nullptr, 0, stream) );

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

  return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,opReduce,sizeA,sizeC,type](
    cudaStream_t stream,
    cutensorHandle_t const* handle,
    float* out,
    vector<float const*> inns)
  {
    cudaDataType_t typeA = dtype_to_cudatype(type);
    cudaDataType_t typeC = dtype_to_cudatype(type);
    cutensorComputeType_t typeCompute = dtype_to_computetype(type);

    typedef float floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

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

    void *A_d = (void*)inns[0];
    void* C_d = (void*)out;

    uint64_t worksize = 0;
    handle_cutensor_error(cutensorReductionGetWorkspaceSize(handle, 
                 A_d, &descA, modeA.data(),
                 C_d, &descC, modeC.data(),
                 C_d, &descC, modeC.data(),
                 opReduce, typeCompute, &worksize));
    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    }
    
    cutensorStatus_t err;
     err = cutensorReduction(handle, 
                (const void*)&alpha, A_d, &descA, modeA.data(),
                (const void*)&beta,  C_d, &descC, modeC.data(), 
                                     C_d, &descC, modeC.data(), 
                opReduce, typeCompute, work, worksize,stream);

  };

  
}

cutensor_kernel_t
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

    floatTypeCompute alpha = (floatTypeCompute)unary.arg.scale;

    return [modeA,nmodeA,extent_A,alpha,typeA,typeCompute]
    (cudaStream_t stream, cutensorHandle_t const* handle, float* out, vector<float const*> inns){
      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, CUTENSOR_OP_IDENTITY));


      cutensorPermutation(handle,
                (void*)&alpha, inns[0], &descA, modeA.data(),
                out, &descA, modeA.data(),
                typeCompute, stream);
    };
  }
  else if(std::holds_alternative<cutensor_elementwise_op_t::binary_t>(op.op)){
    auto binary = std::get<cutensor_elementwise_op_t::binary_t>(op.op);

    std::vector<int> modeA = binary.lhs.modes;
    std::vector<int> modeC = binary.rhs.modes;
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    vector<int64_t> extent_A;
    for(auto const& mode: modeA) {
      extent_A.push_back(op.join_shape[mode]);
    }
    vector<int64_t> extent_C;
    for(auto const& mode: modeC) {
      extent_C.push_back(op.join_shape[mode]);
    }

    floatTypeCompute alpha = (floatTypeCompute)binary.lhs.scale;
    floatTypeCompute gamma = (floatTypeCompute)binary.rhs.scale;
    return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C,alpha,gamma,typeA,typeC,typeCompute,binary]
    (cudaStream_t stream, cutensorHandle_t const* handle, float* out, vector<float const*> inns){
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
      cutensorElementwiseBinary(handle,
                (void*)&alpha, inns[0], &descA, modeA.data(),
                (void*)&gamma, inns[1], &descC, modeC.data(),
                out, &descC, modeC.data(),
                binary.op, typeCompute, stream);
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

    floatTypeCompute alpha = (floatTypeCompute)ternary.a0.scale;
    floatTypeCompute beta  = (floatTypeCompute)ternary.a1.scale;
    floatTypeCompute gamma = (floatTypeCompute)ternary.a2.scale;
    return [
      modeA,modeB,modeC,nmodeA,nmodeB,nmodeC,
      extent_A,extent_B,extent_C,
      alpha,beta,gamma,typeA,typeB,typeC,typeCompute,ternary]
      (
        cudaStream_t stream,
        cutensorHandle_t const* handle,
        float* out,
        vector<float const*> inns
      )
    {
      cutensorTensorDescriptor_t descA;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descA,
                  nmodeA,
                  extent_A.data(),
                  NULL /* stride */,
                  typeA, CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descB;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descB,
                  nmodeB,
                  extent_B.data(),
                  NULL /* stride */,
                  typeB, CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descC;
      handle_cutensor_error(
        cutensorInitTensorDescriptor(handle,
                  &descC,
                  nmodeC,
                  extent_C.data(),
                  NULL /* stride */,
                  typeC, CUTENSOR_OP_IDENTITY));
      cutensorElementwiseTrinary(handle,
                (void*)&alpha, inns[0], &descA, modeA.data(),
                (void*)&beta , inns[1], &descB, modeB.data(),
                (void*)&gamma, inns[2], &descC, modeC.data(),
                              out, &descC, modeC.data(),
                ternary.op_0_1, ternary.op_01_2, typeCompute, stream);
    };
  }

  return {};
}

optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e)
{
  // TODO
  cutensor_elementwise_op_t op;
  op.join_shape = e.join_shape;

  if(e.inns.size()==1){

  }else if(e.inns.size()==2){

  }else if(e.inns.size()==3){
    //op.op = cutensor_elementwise_op_t::ternary_t{e.castable,e.castable,{,,e.inns[0]},{,,e.inns[1]},{,,e.inns[2]}};

    //cutensor_elementwise_op_t op = {e.joinshape,{e.castable,e.castable,{,,e.inns[0]},{,,e.inns[1]},{,,e.inns[2]}}}
    //return op;

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

void handle_cutensor_error(cutensorStatus_t const& err) {
  if(err != CUTENSOR_STATUS_SUCCESS) {
    string msg = cutensorGetErrorString(err);
    throw std::runtime_error("handle_cutensor_error: " + msg);
  }
}
