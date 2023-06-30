#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/reference.h"
#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"

#include <chrono>

#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

void test_mm(dtype_t dtype) {
  // ij,jk->ik
  uint64_t i = 5;
  uint64_t j = 6;
  uint64_t k = 7;

  einsummable_t matmul = einsummable_t::from_matmul(i,j,k, dtype);

  dbuffer_t lhs = make_dbuffer(dtype, i*j);
  dbuffer_t rhs = make_dbuffer(dtype, j*k);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  dbuffer_t out = make_dbuffer(dtype, i*k);

  out.zeros();
  
  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);
  build_contraction(&desc,handle,matmul);


  size_t sizeA = sizeof(float) * i*j;
  size_t sizeB = sizeof(float) * j*k;
  size_t sizeC = sizeof(float) * i*k;
  
  void *lh, *rh, *ou;
  cudaMalloc((void**)&lh, sizeA);
  cudaMalloc((void**)&rh, sizeB);
  cudaMalloc((void**)&ou, sizeC);

  float* A = (float*)lhs.ptr();

  float* B = (float*)rhs.ptr();

  float* C = (float*)out.ptr();

  cudaMemcpy(ou, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lh, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(rh, B, sizeB, cudaMemcpyHostToDevice);

  

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  cudaStreamDestroy(stream);

  //cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);
  
  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  float* C_out = (float*)ou;
  const int totalSize = static_cast<int>(sizeC);
  float* out2 = new float[totalSize]();

  //cudaMemcpy(out2, C_out,sizeC, cudaMemcpyDeviceToHost);

  //for (int i = 0; i < 35; ++i) {
  //    std::cout << "The element " << i <<  " for out 2 " << out2[i] << std::endl;
  //}

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("MM ARE NOT CLOSE!");
  }

  DOUT(out_ref);
  DOUT(out);
}

void test_contraction(dtype_t dtype) {
  // bij,jk->bik
  // 013 32  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 5;
  uint64_t b = 6;
  uint64_t c = 4;
  uint64_t d = 8;
  uint64_t e = 3;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 2, 1, 4}, {0, 1, 3, 4} },
    3,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a*c*b*e);
  dbuffer_t rhs = make_dbuffer(dtype, a*d*b*e);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b*c*d);
  out.zeros();

  printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);
  build_contraction(&desc,handle,matmul);

  printf("here\n");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  //NEW



  size_t sizeA = lhs.size();
  size_t sizeB = rhs.size();
  size_t sizeC = out.size();

  //std::cout << sizeC << std::endl;
  //std::cout << out.size() << std::endl;
  
  void *lh, *rh, *ou;
  cudaMalloc((void**)&lh, sizeA);
  cudaMalloc((void**)&rh, sizeB);
  cudaMalloc((void**)&ou, sizeC);

  float* A = (float*)lhs.ptr();

  float* B = (float*)rhs.ptr();

  float* C = (float*)out.ptr();

  cudaMemcpy(ou, out.ptr(), sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lh, lhs.ptr(), sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(rh, rhs.ptr(), sizeB, cudaMemcpyHostToDevice);

  //NEW


  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    //DOUT(out_ref);
    //DOUT(out);
    throw std::runtime_error("CONTRACTION ARE NOT CLOSE!");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void dumbTest(dtype_t dtype){
  uint64_t i = 2;
  uint64_t j = 3;
  uint64_t k = 2;

  einsummable_t matmul = einsummable_t::from_matmul(i,j,k, dtype);

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);
  build_contraction(&desc,handle,matmul);


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  //float *A = (float*) malloc(sizeof(float)*6);
  //float *B = (float*) malloc(sizeof(float)*6);
  float *C = (float*) malloc(sizeof(float)*4);

  //for(int64_t i = 0; i < 6; i++)
  //    A[i] = i+1;
  //for(int64_t i = 0; i < 6; i++)
  //    B[i] = i+1;
  for(int64_t i = 0; i < 4; i++)
      C[i] = 0;

  dbuffer_t lhs1 = make_dbuffer(dtype, i*j);
  dbuffer_t rhs1 = make_dbuffer(dtype, j*k);

  lhs1.iota(1);
  rhs1.iota(1);
  //lhs1.random();
  //rhs1.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs1, rhs1});

  float* A = (float*)lhs1.ptr();

  float* B = (float*)rhs1.ptr();


  size_t sizeA = sizeof(float) * 6;
  size_t sizeB = sizeof(float) * 6;
  size_t sizeC = sizeof(float) * 4;
  
  void *lhs, *rhs, *out;
  cudaMalloc((void**)&lhs, sizeA);
  cudaMalloc((void**)&rhs, sizeB);
  cudaMalloc((void**)&out, sizeC);

  cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lhs, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(rhs, B, sizeB, cudaMemcpyHostToDevice);

  execute_contraction(stream,handle,&desc,out,lhs,rhs,dtype);

  cudaStreamDestroy(stream);

  float* C_out = (float*)out;

  const int totalSize = static_cast<int>(sizeC);
  float* out2 = new float[totalSize]();
  //cudaMemcpy(out2, C_out,sizeC, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; ++i) {
       std::cout << "The element " << i << " for out 2 " << out2[i] << std::endl;
    }

  dbuffer_t out3 = make_dbuffer(dtype, i*k);

  cudaMemcpy(out3.ptr(), out,sizeC, cudaMemcpyDeviceToHost);

  DOUT(out3);

  if(!is_close(out_ref, out3)) {
    DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("MM ARE NOT CLOSE!");
  }

  
}


void dumbTest2(dtype_t dtype){
  uint64_t i = 2;
  uint64_t j = 3;
  uint64_t k = 2;


  einsummable_t matmul = einsummable_t::from_matmul(i,j,k, dtype);

  dbuffer_t lhs = make_dbuffer(dtype, i*j);
  dbuffer_t rhs = make_dbuffer(dtype, j*k);

  lhs.iota(1);
  rhs.iota(1);

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  DOUT(out_ref);
  float* floatlhs = (float*)lhs.ptr(); 

  for (int i = 0; i < 4; ++i) {
    std::cout << "The element " << i << " for out 2 " << floatlhs[i] << std::endl;
  }

}

void dumbTest3(dtype_t dtype){
  uint64_t i = 2;
  uint64_t j = 3;
  uint64_t k = 2;

  einsummable_t matmul = einsummable_t::from_matmul(i,j,k, dtype);

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);
  build_contraction(&desc,handle,matmul);


  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *A = (float*) malloc(sizeof(float)*6);
  float *B = (float*) malloc(sizeof(float)*6);
  float *C = (float*) malloc(sizeof(float)*4);

  //for(int64_t i = 0; i < 6; i++)
  //    A[i] = i+1;
  //for(int64_t i = 0; i < 6; i++)
  //    B[i] = i+1;
  for(int64_t i = 0; i < 4; i++)
      C[i] = 0;

  A[0] = 5;
  A[1] = 7;
  A[2] = 6;
  A[3] = 3;
  A[4] = 4;
  A[5] = 2;

  B[0] = 9;
  B[1] = 4;
  B[2] = 5;
  B[3] = 6;
  B[4] = 1;
  B[5] = 7;
  
  //lhs1.random();
  //rhs1.random();

  


  size_t sizeA = sizeof(float) * 6;
  size_t sizeB = sizeof(float) * 6;
  size_t sizeC = sizeof(float) * 4;
  
  void *lhs, *rhs, *out;
  cudaMalloc((void**)&lhs, sizeA);
  cudaMalloc((void**)&rhs, sizeB);
  cudaMalloc((void**)&out, sizeC);

  cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lhs, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(rhs, B, sizeB, cudaMemcpyHostToDevice);

  execute_contraction(stream,handle,&desc,out,lhs,rhs,dtype);

  cudaStreamDestroy(stream);

  float* C_out = (float*)out;

  const int totalSize = static_cast<int>(sizeC);
  float* out2 = new float[totalSize]();
  //cudaMemcpy(out2, C_out,sizeC, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; ++i) {
       std::cout << "The element " << i << " for out 2 " << out2[i] << std::endl;
    }

  dbuffer_t out3 = make_dbuffer(dtype, i*k);

  cudaMemcpy(out3.ptr(), out,sizeC, cudaMemcpyDeviceToHost);

  DOUT(out3);

  

  
}

void dumbTest4(dtype_t dtype){

  vector<int> inn_modes {0, 1};
  vector<int> out_modes {0};

  vector<uint64_t> inn_shape {2,3};
  vector<uint64_t> out_shape {2};

  auto func = build_cutensor_reduction(inn_modes,inn_shape,out_modes,out_shape,castable_t::add,dtype);


  float *A = (float*) malloc(sizeof(float)*6);
  float *C = (float*) malloc(sizeof(float)*2);

  //for(int64_t i = 0; i < 6; i++)
  //    A[i] = i+1;
  //for(int64_t i = 0; i < 6; i++)
  //    B[i] = i+1;
  for(int64_t i = 0; i < 2; i++)
      C[i] = 0;

  A[0] = 5;
  A[1] = 7;
  A[2] = 6;
  A[3] = 3;
  A[4] = 4;
  A[5] = 2;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t sizeA = sizeof(float) * 6;
  size_t sizeC = sizeof(float) * 2;

  void *lhs,  *out;
  cudaMalloc((void**)&lhs, sizeA);
  cudaMalloc((void**)&out, sizeC);

  cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lhs, A, sizeA, cudaMemcpyHostToDevice);

  vector<void const*> inns;
  inns.push_back(lhs);

  func(stream, handle, out, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(C, out,sizeC, cudaMemcpyDeviceToHost);

  float* out1 = new float[2]();

  out1 = C;

  for (size_t i = 0; i < 2; i++) {
    std::cout << out1[i] << " ";
  }
  std::cout << std::endl;



}

void dumbTest5(dtype_t dtype){
  uint64_t b = 2;
  uint64_t i = 3;
  

  einsummable_t reduction = einsummable_t(
    {b, i},
    { {0, 1}},
    1,
    scalarop_t::make_identity(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b*i);

  lhs.iota(1);


  dbuffer_t out_ref = reference_einsummable(reduction, {lhs});

  DOUT(out_ref);
}

void test_reduction(dtype_t dtype){
  uint64_t b = 5;
  uint64_t i = 7;
  

  einsummable_t reduction = einsummable_t(
    {b, i},
    { {0, 1}},
    1,
    scalarop_t::make_identity(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b*i);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(reduction, {lhs});


  vector<int> inn_modes = reduction.inns[0];
  vector<int> out_modes;
  for(int i = 0; i<reduction.out_rank;i++){
    out_modes.push_back(i);
  }

  vector<uint64_t> inn_shape;
  vector<uint64_t> out_shape;
  for(auto const& mode: inn_modes) {
    inn_shape.push_back(reduction.join_shape[mode]);
  }
  for(auto const& mode: out_modes) {
    out_shape.push_back(reduction.join_shape[mode]);
  }

  auto func = build_cutensor_reduction(inn_modes,inn_shape,out_modes,out_shape,castable_t::add,dtype);


  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t sizeA = lhs.size();
  size_t sizeC = out.size();

  void *lh, *ou;

  cudaMalloc((void**)&lh, sizeA);
  cudaMalloc((void**)&ou, sizeC);

  cudaMemcpy(ou, out.ptr(), sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(lh, lhs.ptr(), sizeA, cudaMemcpyHostToDevice);


  vector<void const*> inns;
  inns.push_back(lh);

  func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("CONTRACTION ARE NOT CLOSE!");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << std::endl;
  }
}

void dumbTest6(dtype_t dtype){
  //bi, b -> bi
  //01, 0 -> 01

  uint64_t b = 2;
  uint64_t i = 3;
  

  einsummable_t elementwise = einsummable_t(
    {b, i},
    { {0},{0, 1}},
    2,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);
  dbuffer_t rhs = make_dbuffer(dtype, b*i);

  lhs.iota(1);
  rhs.iota(1);


  dbuffer_t out_ref = reference_einsummable(elementwise, {lhs, rhs});

  DOUT(out_ref);
}

void print_cutensor_op(cutensorOperator_t op){

    if(op==CUTENSOR_OP_ADD){
        printf("CUTENSOR_OP_ADD\n");
    }else if(op==CUTENSOR_OP_MUL){
        printf("CUTENSOR_OP_MUL\n");
    }else if(op==CUTENSOR_OP_EXP){
        printf("CUTENSOR_OP_EXP\n");
    }else if(op==CUTENSOR_OP_IDENTITY){
        printf("CUTENSOR_OP_IDENTITY\n");
    }


}


void dumbTest7(dtype_t dtype){
  //bi, b -> bi
  //01, 0 -> 01

  uint64_t b = 2;
  uint64_t i = 3;
  

  einsummable_t elementwise = einsummable_t(
    {b, i},
    { {0},{0,1}},
    2,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  size_t sizeA = sizeof(float) * 2;
  size_t sizeB = sizeof(float) * 6;
  size_t sizeC = sizeof(float) * 6;

  float *A = (float*) malloc(sizeof(float)*2);
  float *B = (float*) malloc(sizeof(float)*6);
  float *C = (float*) malloc(sizeof(float)*6);

  //for(int64_t i = 0; i < 6; i++)
  //    A[i] = i+1;
  //for(int64_t i = 0; i < 6; i++)
  //    B[i] = i+1;
  for(int64_t i = 0; i <6; i++)
      C[i] = 0;

  A[0] = 1;
  A[1] = 2;
  //A[2] = 3;
  //A[3] = 4;
  //A[4] = 5;
  //A[5] = 6;

  B[0] = 1;
  B[1] = 2;
  B[2] = 3;
  B[3] = 4;
  B[4] = 5;
  B[5] = 6;

  auto ceot = make_cutensor_elementwise_op(elementwise);  

  cutensor_elementwise_op_t op = *ceot;
/*
  if(std::holds_alternative<cutensor_elementwise_op_t::binary_t>(op.op)){
        printf("This is binary\n");
        printf("OP is\n");
        auto binary = std::get<cutensor_elementwise_op_t::binary_t>(op.op);
        print_cutensor_op(binary.op);
        printf("Arg 1\n");
        printf("Scale is: %.2f\n",binary.lhs.scale.f32());
        printf("OP is ");
        print_cutensor_op(binary.lhs.op);
        printf("Arg 2\n");
        printf("Scale is: %.2f\n",binary.rhs.scale.f32());
        printf("OP is ");
        print_cutensor_op(binary.rhs.op);
    }
*/

  //cutensor_elementwise_op_t op;
  //op.join_shape = elementwise.join_shape;
  //vector<uint64_t> join_shape{2,3};
  //op.join_shape = join_shape;
  //cutensor_elementwise_op_t::arg_t lhs {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
  //cutensor_elementwise_op_t::arg_t rhs {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
  //cutensor_elementwise_op_t::arg_t a2 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
  //cutensorOperator_t op1 = CUTENSOR_OP_ADD;


  //cutensor_elementwise_op_t::binary_t ter_op{
   // op1,
  //  lhs,
  //  rhs
  //};

  //op.op = ter_op;

  auto func = build_cutensor_elementwise(op);

  void* out;
  void* inn0;
  void* inn1;

  cudaMalloc(&inn0, sizeA);
  cudaMalloc(&inn1, sizeB);
  cudaMalloc(&out, sizeC);

  cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(inn0, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(inn1, B, sizeB, cudaMemcpyHostToDevice);

  std::vector<void const*> inns;
  inns.push_back(inn0);
  inns.push_back(inn1);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  func(stream,handle,out, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(C, out,sizeC, cudaMemcpyDeviceToHost);

  float* out1 = new float[6]();

  out1 = (float*)C;

  for (int i = 0; i < 6; ++i) {
      std::cout << "The element " << i << " for out 1 " << out1[i] << std::endl;
  }
  std::cout << op.join_shape << std::endl;

  
}



int main() {
  //test_mm(dtype_t::f32);
  //test_contraction(dtype_t::f16);
  test_contraction(dtype_t::f32);
  //test_contraction(dtype_t::f64);
  //test_contraction(dtype_t::c64);
  //dumbTest(dtype_t::f32);
  //dumbTest2(dtype_t::f32);
  //dumbTest3(dtype_t::f32);
  //dumbTest4(dtype_t::f32);
  //dumbTest5(dtype_t::f32);
  //test_reduction(dtype_t::f16);
  //test_reduction(dtype_t::f32);
  //test_reduction(dtype_t::f64);
  //test_reduction(dtype_t::c64);

  //dumbTest6(dtype_t::f32);
  //dumbTest7(dtype_t::f32);
}