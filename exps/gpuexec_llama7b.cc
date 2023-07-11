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


void test_contraction(dtype_t dtype) {
  // bij,jk->bik
  // 013 32  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 256;
  uint64_t b = 64;
  uint64_t c = 32;
  uint64_t d = 28;
  uint64_t e = 28;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 2, 1, 4}, {0, 1, 3, 4} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a*c*b*e);
  dbuffer_t rhs = make_dbuffer(dtype, a*d*b*e);

  lhs.random();
  rhs.random();

  //dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b*c*d);
  out.zeros();

  //matmul.merge_adjacent_dims();

  printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  auto startTime1 = std::chrono::high_resolution_clock::now();




    
  build_contraction(&desc,handle,matmul);

  auto endTime1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
  std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;


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
  auto startTime2 = std::chrono::high_resolution_clock::now();

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  auto endTime2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  //if(!is_close(out_ref, out)) {
  //  DOUT(dtype);
  //  DOUT(out_ref);
  //  DOUT(out);
  //  throw std::runtime_error("CONTRACTION ARE NOT CLOSE!");
  //}else{
  //   std::cout << "Contraction operation successful for dtype "<<dtype << std::endl;
  //}

  //DOUT(out_ref);
  //DOUT(out);

}


void kernel3(dtype_t dtype){
  uint64_t b = 32;
  uint64_t i = 256;
  

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
    printf("KERNEL3 ARE NOT CLOSE!");
  }else{
     std::cout << "Reduction operation successful for dtype "<<dtype << "for kernel 3" <<std::endl;
  }
}


void kernel9(dtype_t dtype) {
  // ac,bc->ab
  // 02,12   01
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 32;
  uint64_t c = 32;
  uint64_t d = 28;
  uint64_t e = 28;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c},
    { {0, 2}, {1, 2} },
    2,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a*c);
  dbuffer_t rhs = make_dbuffer(dtype, b*c);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  build_contraction(&desc,handle,matmul);

  //auto endTime1 = std::chrono::high_resolution_clock::now();
  //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
  //std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;


  //printf("here\n");
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
  //auto startTime2 = std::chrono::high_resolution_clock::now();

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {\
    //printf("KERNEL9:\n");
    //DOUT(dtype);
    //DOUT(out_ref);
    //DOUT(out);
    printf("**KERNEL9 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << "of Kernel 9" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void kernel11(dtype_t dtype) {
  // acbe, adbe -> abcd
  // 0214, 0314   0123
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 16;
  uint64_t c = 8;
  uint64_t d = 8;
  uint64_t e = 16;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 2, 1, 4}, {0, 3, 1, 4} },
    4,
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

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  build_contraction(&desc,handle,matmul);

  //auto endTime1 = std::chrono::high_resolution_clock::now();
  //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
  //std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;


  //printf("here\n");
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
  //auto startTime2 = std::chrono::high_resolution_clock::now();

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {\
    //printf("KERNEL11:\n");
    //DOUT(dtype);
    //DOUT(out_ref);
    //DOUT(out);
    printf("KERNEL11 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 11" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}


void kernel21(dtype_t dtype) {
  // abce, aebd -> abcd
  // 0124, 0413  0123
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 2;
  uint64_t b = 8;
  uint64_t c = 4;
  uint64_t d = 8;
  uint64_t e = 8;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 1, 2, 4}, {0, 4, 1, 3} },
    4,
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

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  build_contraction(&desc,handle,matmul);

  //auto endTime1 = std::chrono::high_resolution_clock::now();
  //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
  //std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;


  //printf("here\n");
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
  //auto startTime2 = std::chrono::high_resolution_clock::now();

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {\
    printf("KERNEL21:\n");
    //DOUT(dtype);
    //DOUT(out_ref);
    //DOUT(out);
    printf("KERNEL21 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 21" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void kernel22(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 2;
  uint64_t b = 4;
  uint64_t c = 8;
  uint64_t d = 4;
  uint64_t e = 4;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 3, 1, 4}, {2, 3, 4} },
    3,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a*d*b*e);
  dbuffer_t rhs = make_dbuffer(dtype, c*d*e);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b*c);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  cutensorContractionDescriptor_t desc;

  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  build_contraction(&desc,handle,matmul);

  //auto endTime1 = std::chrono::high_resolution_clock::now();
  //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
  //std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;


  //printf("here\n");
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
  //auto startTime2 = std::chrono::high_resolution_clock::now();

  execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {\
    printf("KERNEL22:\n");
    //DOUT(dtype);
    //DOUT(out_ref);
    //DOUT(out);
    printf("KERNEL22 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 22" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void kernel13(dtype_t dtype){
  //elementwise +(addition)
  //ab, b -> ab

  uint64_t a = 8;
  uint64_t b = 4;

  
  einsummable_t elementwise = einsummable_t(
    {a, b},
    { {0,1},{1}},
    2,
    scalarop_t::make_add(dtype),
    castable_t::add);
  
  
  dbuffer_t lhs = make_dbuffer(dtype, a*b);
  dbuffer_t rhs = make_dbuffer(dtype, b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(elementwise, {lhs, rhs});

  
  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");


  cutensorHandle_t* handle;
  cutensorCreate(&handle);

  //printf("here\n");
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

  auto ceot = make_cutensor_elementwise_op(elementwise);  

  cutensor_elementwise_op_t op = *ceot;

  auto func = build_cutensor_elementwise(op);



  vector<void const*> inns;
  inns.push_back(rh);
  inns.push_back(lh);

  func(stream, handle, ou, inns);

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL13:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL13 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << " of Kernel 13" <<std::endl;
  }
  //DOUT(out_ref);
  //DOUT(out);

}


int main(){
  kernel3(dtype_t::f32);
  kernel9(dtype_t::f16);
  kernel9(dtype_t::f32);
  kernel11(dtype_t::f16);
  kernel11(dtype_t::f32);
  kernel13(dtype_t::f16);
  kernel13(dtype_t::f32);
  kernel21(dtype_t::f16);
  kernel21(dtype_t::f32);
  kernel22(dtype_t::f16);
  kernel22(dtype_t::f32);
}