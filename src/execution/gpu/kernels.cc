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
        out_modes, out_shape);
    }

    if(e.inns[0] == vector<int>{0,1} && e.out_rank == 1) {
      // call a canned ij->i kernel
      return build_simple_reduction(
        e.join_shape[0],
        e.join_shape[1],
        e.castable.value());
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

//************************************
//I think we need handle for this one
void build_contraction(
  cutensorContractionDescriptor_t* desc,
  cutensorHandle_t const* handle,
  einsummable_t const& e_,
  float* out,
  float const* lhs,
  float const* rhs)
{
  einsummable_t e = e_.merge_adjacent_dims();

  if(!e.is_contraction()) {
    throw std::runtime_error("build_contraction must be given a contraction");
  }
  //Assuming we are doing C = A contraction B


  // if we have mhkn, ukvh -> munv
  // then we should have:
  //std::vector<int> modeC{'m','u','n','v'};
  //std::vector<int> modeA{'m','h','k','n'};
  //std::vector<int> modeB{'u','k','v','h'};
  //*************************
  std::vector<int> modeA = e.inns[0];
  std::vector<int> modeB = e.inns[1];
  std::vector<int> modeC;
  for (int i = 0; i < e.out_rank; i++) {
    modeC.push_back(i);
  }

  // TODO
  auto nmodeA = e.inns[0].size();
  auto nmodeB = e.inns[1].size();

  //***************************
  //dimension of C 
  auto nmodeC = e.out_rank;

  // CUDA types
  cudaDataType_t typeA = CUDA_R_32F;
  cudaDataType_t typeB = CUDA_R_32F;
  cudaDataType_t typeC = CUDA_R_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

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
  HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descA,
                nmodeA,
                extent_A.data(),
                NULL,/*stride*/
                typeA, CUTENSOR_OP_IDENTITY ) );
  
  cutensorTensorDescriptor_t descB;
  HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descB,
                nmodeB,
                extent_B.data(),
                NULL,/*stride*/
                typeB, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
              &descC,
              nmodeC,
              extent_C,
              NULL,/*stride*/
              typeC, CUTENSOR_OP_IDENTITY ) );
  
  //******************************************
  // get the memory pointers to the tensors
  auto ptrA = lhs;
  auto ptrB = rhs;
  auto ptrC = out;

  uint32_t alignmentRequirementA;
  HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
              ptrA,
              &descA,
              &alignmentRequirementA) );

  uint32_t alignmentRequirementB;
  HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
              ptrB,
              &descB,
              &alignmentRequirementB) );

  uint32_t alignmentRequirementC;
  HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
              ptrC,
              &descC,
              &alignmentRequirementC) );


  // Init Contraction Descriptor need to be in the format of
  // D = alpha * A * B + beta * C
  // so we should probably use a C for both C and D

  HANDLE_ERROR( cutensorInitContractionDescriptor( handle,
                &desc,
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
  float* out,
  float const* lhs,
  float const* rhs)
{
  // TODO

  // Set the algorithm to use
  cutensorContractionFind_t find;
  HANDLE_ERROR( cutensorInitContractionFind(
              handle, &find,
              CUTENSOR_ALGO_DEFAULT) );

  size_t worksize = 0;
  HANDLE_ERROR( cutensorContractionGetWorkspace(handle,
              &desc,
              &find,
              CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );
  
  void *work = nullptr;
  if(worksize > 0)
  {
      if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
      {
          work = nullptr;
          worksize = 0;
      }
  }


  cutensorContractionPlan_t plan;
  HANDLE_ERROR( cutensorInitContractionPlan(handle,
                                            &plan,
                                            &desc,
                                            &find,
                                            worksize) );
  
  cutensorStatus_t err;


  err = cutensorContraction(handle, &plan, NULL, lhs,
                            rhs, NULL, out, out,
                            work, worksize, stream);

  cudaDeviceSynchronize();

  
}

//******************************
//What are inn_modes and inn_shape?
cutensor_kernel_t
build_cutensor_reduction(
  vector<int> inn_modes, vector<uint64_t> inn_shape,
  vector<int> out_modes, vector<uint64_t> out_shape)
{
  //********************************
  //Same problem as contraction
  //If we have mhkv->mv
  //Then we should have:
  //std::vector<int32_t> modeA{'m','h','k','v'};
  //std::vector<int32_t> modeC{'m','v'};

  std::vector<int32_t> modeA = inn_modes;
  std::vector<int32_t> modeC = out_modes;
  int32_t nmodeA = modeA.size();
  int32_t nmodeC = modeC.size();


  // extent = size of each dimension
  vector<int64_t> extent_A = inn_shape;
  vector<int64_t> extent_C = out_shape;


  


  // ****************************************
  // What is float*, vector<float const*> here?
  return [modeA,modeC,nmodeA,nmodeC,extent_A,extent_C]
  (cudaStream_t stream, cutensorHandle_t const* handle, float* out, vector<float const*> inns){
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL /* stride */,
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL /* stride */,
                 typeC, CUTENSOR_OP_IDENTITY));
    

    //Specify reduce OP (this should be the one?)
    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    //get the memory pointer
    float* A_d = inns[0];
    float* C_d = out;
 
    //Workspace
    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspaceSize(&handle, 
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
    err = cutensorReduction(&handle, 
                NULL/*alpha*/, A_d,
                &descA, modeA.data(),NULL/*beta*/,  
                out,
                &descC, modeC.data(), out, &descC, modeC.data(), 
                opReduce, typeCompute, work, worksize, stream);


  };
}

cutensor_kernel_t
build_simple_reduction(
  uint64_t ni, uint64_t nj,
  castable_t castable)
{
  // TODO
  return [](cudaStream_t stream, cutensorHandle_t const* handle, float* out, vector<float const*> inns){
    for(int i=0; i<ni;i++){
      float num = 0;
      for(int j=0; j<nj;j++){
        if(c == castable_t::mul) { 
          num*=inns[0][i*nj+j];
        } else if(c == castable_t::min) { 
          num = std::min(num, inns[0][i*nj+j]); 
        } else if(c == castable_t::max) { 
          num = std::max(num, inns[0][i*nj+j]);
        }
      }
      out[i] = num;
    }
  
  };
}

cutensor_kernel_t
build_cutensor_elementwise(cutensor_elementwise_op_t op)
{
  // TODO
  return {};
}

optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e)
{
  // TODO
  return std::nullopt;
}

cutensor_kernel_t
build_straight_elementwise(
  scalarop_t op,
  uint64_t size)
{
  // TODO: dispatch to canned elementwise kernels here
  return {};
}
