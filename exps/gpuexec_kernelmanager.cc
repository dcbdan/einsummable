#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/reference.h"
#include "../src/engine/gpu/kernels.h"
#include "../src/engine/gpu/gpu_kernel_manager.h"

#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"

#include <chrono>

#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

void kernel23(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 147;
  uint64_t b = 16;
  uint64_t c = 16;
  uint64_t d = 4;
  uint64_t e = 4;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a},
    { {0}, {0} },
    1,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a);
  dbuffer_t rhs = make_dbuffer(dtype, a);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();


    
  kernel_manager_t km(0);
  //auto const& [is_built,wsz] = km.build(matmul);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  //uint64_t size = wsz.value();

  

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  uint64_t size = workspace_info.value().value();
  
  void* work;
  cudaMalloc(&work, size);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);


  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.03f)) {\
    printf("KERNEL23:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL23 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << " of Kernel 23" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void kernel27(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 88;
  uint64_t b = 16;
  uint64_t c = 16;
  uint64_t d = 4;
  uint64_t e = 4;



  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a},
    { {0}, {0} },
    1,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a);
  dbuffer_t rhs = make_dbuffer(dtype, a);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();


    
  kernel_manager_t km(0);
  //auto const& [is_built,wsz] = km.build(matmul);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  //uint64_t size = wsz.value();

  

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  uint64_t size = workspace_info.value().value();
  
  void* work;
  cudaMalloc(&work, size);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.03f)) {\
    printf("KERNEL27:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL27 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << " of Kernel 27" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}

void kernel10(dtype_t dtype){
  uint64_t a = 2;
  uint64_t b = 4;
  uint64_t c = 4;
  uint64_t d = 2;
  
  auto scar = parse_with_ss<scalarop_t>("*[hole|c64@0,hole|c64@1]");
  


  einsummable_t einsummable = einsummable_t(
    {a,b,c,d},
    {{0, 1, 2, 3}, {1, 3}},
    4,
    scar,
    castable_t::add);


  scalarop_t op = einsummable.join;

  op = op.simplify();

  auto op_str = op.to_cppstr();

  //std::cout <<  op_str <<std::endl;

  
  auto scar2 = parse_with_ss<scalarop_t>("*[hole|f32@0,hole|f32@1]");
  


  einsummable_t einsummable2 = einsummable_t(
    {a,b*2,c,d},
    {{0, 1, 2, 3}, {1, 3}},
    4,
    scar2,
    castable_t::add);

 
  dbuffer_t lhs = make_dbuffer(dtype, a*b*c*d);
  dbuffer_t rhs = make_dbuffer(dtype, b*d);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(einsummable, {lhs, rhs});

  

  kernel_manager_t km(0);
  auto workspace_info = km.build(einsummable);

  //printf("hereyet?");

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();

  void* work;
  cudaMalloc(&work, size);

  dbuffer_t out = make_dbuffer(dtype,  a*b*c*d);
  out.zeros();

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  

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



  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(einsummable,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL10 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 10" <<std::endl;
  }
}











void kernel9(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 16;
  uint64_t c = 16;
  uint64_t d = 4;
  uint64_t e = 4;



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
  dbuffer_t rhs = make_dbuffer(dtype, c*b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);
  std::cout << "Kernel 9 built successfully" << std::endl; 

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  std::cout << "Kernel 9 start executing" << std::endl; 
  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.03f)) {\
    printf("KERNEL9:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL9 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 9" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}


void kernel24(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 16;
  uint64_t c = 64;
  uint64_t d = 4;
  uint64_t e = 4;



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
  dbuffer_t rhs = make_dbuffer(dtype, c*b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.1f)) {\
    printf("KERNEL24:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL24 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 24" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}


void kernel28(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 128;
  uint64_t c = 16;
  uint64_t d = 4;
  uint64_t e = 4;



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
  dbuffer_t rhs = make_dbuffer(dtype, c*b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.03f)) {\
    printf("KERNEL28:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL28 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of kernel 28" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}


void kernel29(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 32;
  uint64_t c = 64;
  uint64_t d = 4;
  uint64_t e = 4;



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
  dbuffer_t rhs = make_dbuffer(dtype, c*b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.1f)) {\
    printf("KERNEL29:\n");
    //DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    printf("KERNEL29 ARE NOT CLOSE!\n");
  }else{
     std::cout << "Contraction operation successful for dtype "<<dtype << " of Kernel 29" <<std::endl;
  }

  //DOUT(out_ref);
  //DOUT(out);

}




void kernel11(dtype_t dtype) {
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  
  uint64_t a = 4;
  uint64_t b = 16;
  uint64_t c = 8;
  uint64_t d = 8;
  uint64_t e = 16;
   
  /*
  uint64_t a = 1;
  uint64_t b = 32;
  uint64_t c = 1024;
  uint64_t d = 1024;
  uint64_t e = 128;
  */


  //einsummable_t matmul = einsummable_t(
  //  {b, i, k, j},
  //  { {0, 1, 3}, {3, 2} },
  //  3,
  //  scalarop_t::make_mul(dtype),
  //  castable_t::add);

  einsummable_t matmul = einsummable_t(
    {a,b,c,d,e},
    { {0, 2, 1, 4}, {0,3,1,4} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  //dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  //dbuffer_t rhs = make_dbuffer(dtype, j*k);

  dbuffer_t lhs = make_dbuffer(dtype, a*c*b*e);
  dbuffer_t rhs = make_dbuffer(dtype, a*d*e*b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b*c*d);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}
  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out, 0.03f)) {\
    printf("KERNEL11:\n");
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
  // adbe, cde -> abc
  // 0314, 234  012
  //uint64_t b = 3;
  //uint64_t i = 5;
  //uint64_t j = 6;
  //uint64_t k = 7;

  uint64_t a = 4;
  uint64_t b = 16;
  uint64_t c = 8;
  uint64_t d = 16;
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
  dbuffer_t rhs = make_dbuffer(dtype, a*b*d*e);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  //dbuffer_t out = make_dbuffer(dtype, b*i*k);
  dbuffer_t out = make_dbuffer(dtype, a*b*c*d);
  out.zeros();

  //matmul.merge_adjacent_dims();

  //printf("here\n");

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out, 0.03f)) {\
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

  //cutensorContractionDescriptor_t desc;

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  //auto startTime1 = std::chrono::high_resolution_clock::now();




    
  kernel_manager_t km(0);
  auto workspace_info = km.build(matmul);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

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

  //execute_contraction(stream,handle,&desc,ou,lh,rh,dtype);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  km(matmul,stream,ou,inns,workspace);



  //auto endTime2 = std::chrono::high_resolution_clock::now();
  //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - startTime2).count();
  //std::cout << "Execution time for execute_contraction: " << duration2 << " microseconds" << std::endl;



  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);


  //auto f = build_einsummable(1, matmul);
  //f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out, 0.03f)) {\
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

void kernel15(dtype_t dtype){
  uint64_t b = 128;
  uint64_t i = 4;
  

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

  //auto func = build_cutensor_reduction(inn_modes,inn_shape,out_modes,out_shape,castable_t::add,dtype);

  kernel_manager_t km(0);
  auto workspace_info = km.build(reduction);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  void* work;
  cudaMalloc(&work, size);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(reduction,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL15 ARE NOT CLOSE!");
  }else{
     std::cout << "Reduction operation successful for dtype "<<dtype << "for kernel 15" <<std::endl;
  }
}


void kernel18(dtype_t dtype){
  uint64_t b = 128;
  uint64_t i = 4;
  

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

  //auto func = build_cutensor_reduction(inn_modes,inn_shape,out_modes,out_shape,castable_t::add,dtype);

  kernel_manager_t km(0);
  auto workspace_info = km.build(reduction);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  void* work;
  cudaMalloc(&work, size);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(reduction,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL18 ARE NOT CLOSE!");
  }else{
     std::cout << "Reduction operation successful for dtype "<<dtype << "for kernel 18" <<std::endl;
  }
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

  //auto func = build_cutensor_reduction(inn_modes,inn_shape,out_modes,out_shape,castable_t::add,dtype);

  kernel_manager_t km(0);
  auto workspace_info = km.build(reduction);

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  void* work;
  cudaMalloc(&work, size);

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(reduction,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL3 ARE NOT CLOSE!");
  }else{
     std::cout << "Reduction operation successful for dtype "<<dtype << "for kernel 3" <<std::endl;
  }
}



void kernel2(dtype_t dtype){
  uint64_t b = 32;
  
  

  einsummable_t power_2 = einsummable_t(
    {b},
    {{0}},
    1,
    scalarop_t::make_square(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(power_2, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(power_2);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(power_2,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL3 ARE NOT CLOSE!");
  }else{
     std::cout << "Power operation successful for dtype "<<dtype << "for kernel 2" <<std::endl;
  }
}


void kernel5(dtype_t dtype){
  uint64_t b = 32;
  
  

  einsummable_t power_2 = einsummable_t(
    {b},
    {{0}},
    1,
    scalarop_t::make_inverse_sqrt(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(power_2, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(power_2);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(power_2,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL5 ARE NOT CLOSE!");
  }else{
     std::cout << "Power operation successful for dtype "<<dtype << "for kernel 5" <<std::endl;
  }
}

void kernel1(dtype_t src, dtype_t dst){
  uint64_t b = 8192;
  
  
  einsummable_t convert = einsummable_t(
    {b},
    { {0}},
    1,
    scalarop_t::make_convert_dtype(src, dst),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(src, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(convert, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(convert);

  dbuffer_t out = make_dbuffer(dst, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(convert,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL1 ARE NOT CLOSE!\n");
    //DOUT(out_ref);
    //DOUT(out);
  }else{
     std::cout << "Conversion operation successful for dtype "<<dst << "for kernel 1" <<std::endl;
  }
}


void kernel7(dtype_t src, dtype_t dst){
  uint64_t b = 8192;
  
  

  
  einsummable_t convert = einsummable_t(
    {b},
    { {0}},
    1,
    scalarop_t::make_convert_dtype(src, dst),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(src, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(convert, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(convert);

  dbuffer_t out = make_dbuffer(dst, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(convert,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL1 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Conversion operation successful for dtype "<<dst << "for kernel 7" <<std::endl;
  }
}


void kernel14(dtype_t src, dtype_t dst){
  uint64_t b = 512;
  
  

  
  einsummable_t convert = einsummable_t(
    {b},
    { {0}},
    1,
    scalarop_t::make_convert_dtype(src, dst),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(src, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(convert, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(convert);

  dbuffer_t out = make_dbuffer(dst, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(convert,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL1 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Conversion operation successful for dtype "<<dst << "for kernel 14" <<std::endl;
  }
}


void kernel20(dtype_t src, dtype_t dst){
  uint64_t b = 512;
  
  

  
  einsummable_t convert = einsummable_t(
    {b},
    { {0}},
    1,
    scalarop_t::make_convert_dtype(src, dst),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(src, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(convert, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(convert);

  dbuffer_t out = make_dbuffer(dst, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(convert,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL1 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Conversion operation successful for dtype "<<dst << "for kernel 20" <<std::endl;
  }
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


  cutensorHandle_t handle;
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

  kernel_manager_t km(0);
  auto workspace_info = km.build(elementwise);


  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //func(stream, handle, ou, inns);
  km(elementwise,stream,ou,inns);

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


void kernel4(dtype_t dtype){
  uint64_t b = 16;
  
  auto scar = parse_with_ss<scalarop_t>("+[*[constant{f32|0.000244141},hole|f32@0],constant{f32|1e-06}]");
  


  einsummable_t increment = einsummable_t(
    {b},
    {{0}},
    1,
    scar,
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();


  dbuffer_t out_ref = reference_einsummable(increment, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(increment);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(increment,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL4 ARE NOT CLOSE!");
  }else{
     std::cout << "Increment operation successful for dtype "<<dtype << "for kernel 4" <<std::endl;
  }
}

void kernel19(dtype_t dtype){
  uint64_t b = 128;
  uint64_t i = 4;
  
  auto scar = parse_with_ss<scalarop_t>("*[hole|f32@0,power{-1}[hole|f32@1]]");
  


  einsummable_t einsummable = einsummable_t(
    {b, i},
    {{0, 1}, {0}},
    2,
    scar,
    castable_t::add);

 
  dbuffer_t lhs = make_dbuffer(dtype, b*i);
  dbuffer_t rhs = make_dbuffer(dtype, b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(einsummable, {lhs, rhs});

  
  
  
  //printf("herenot?");

  kernel_manager_t km(0);
  auto workspace_info = km.build(einsummable);

  //printf("hereyet?");

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

  dbuffer_t out = make_dbuffer(dtype, b*i);
  out.zeros();

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  

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



  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(einsummable,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL19 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Power_elementwise operation successful for dtype "<<dtype << "for kernel 19" <<std::endl;
  }
}



void kernel26(dtype_t dtype){
  uint64_t b = 16;
  
  auto scar = parse_with_ss<scalarop_t>("*[hole|f16@0,power{-1}[+[constant{f16|1},exp[*[constant{f16|-1},hole|f16@0]]]]]");
  


  einsummable_t increment = einsummable_t(
    {b},
    {{0}},
    1,
    scar,
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();
  //lhs.iota(1);


  dbuffer_t out_ref = reference_einsummable(increment, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(increment);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(increment,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL 26 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Weird custom operation successful for dtype "<<dtype << "for kernel 26" <<std::endl;
  }
}

void kernel17(dtype_t dtype){
  uint64_t b = 16;
  
  auto scar = parse_with_ss<scalarop_t>("exp[hole|f32@0]");
  


  einsummable_t increment = einsummable_t(
    {b},
    {{0}},
    1,
    scar,
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();
  //lhs.iota(1);


  dbuffer_t out_ref = reference_einsummable(increment, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(increment);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(increment,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL 17 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 17" <<std::endl;
  }
}

void kernel12(dtype_t dtype){
  uint64_t b = 16;
  
  auto scar = parse_with_ss<scalarop_t>("*[constant{f16|0.0883789},hole|f16@0]");
  


  einsummable_t increment = einsummable_t(
    {b},
    {{0}},
    1,
    scar,
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b);

  lhs.random();
  //lhs.iota(1);

 
  dbuffer_t out_ref = reference_einsummable(increment, {lhs});


  kernel_manager_t km(0);
  auto workspace_info = km.build(increment);

  dbuffer_t out = make_dbuffer(dtype, b);
  out.zeros();

  cutensorHandle_t handle;
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

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  km(increment,stream,ou,inns);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL 12 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 12" <<std::endl;
  }
}



void kernel6(dtype_t dtype){
  uint64_t b = 32;
  uint64_t i = 256;
  
  auto scar = parse_with_ss<scalarop_t>("*[hole|f32@0,hole|f32@1]");
  


  einsummable_t einsummable = einsummable_t(
    {b, i},
    {{0, 1}, {0}},
    2,
    scar,
    castable_t::add);

 
  dbuffer_t lhs = make_dbuffer(dtype, b*i);
  dbuffer_t rhs = make_dbuffer(dtype, b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(einsummable, {lhs, rhs});

  
  printf("herenot?");

  kernel_manager_t km(0);
  auto workspace_info = km.build(einsummable);

  std::cout << "Log message7" << std::endl;

  printf("hereyet?");

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

  dbuffer_t out = make_dbuffer(dtype, b*i);
  out.zeros();

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  

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



  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(einsummable,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL6 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 6" <<std::endl;
  }
}

void kernel16(dtype_t dtype){
  uint64_t b = 128;
  uint64_t i = 4;
  
  auto scar = parse_with_ss<scalarop_t>("+[hole|f32@0,*[constant{f32|-1},hole|f32@1]]");
  


  einsummable_t einsummable = einsummable_t(
    {b, i},
    {{0, 1}, {0}},
    2,
    scar,
    castable_t::add);

 
  dbuffer_t lhs = make_dbuffer(dtype, b*i);
  dbuffer_t rhs = make_dbuffer(dtype, b);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(einsummable, {lhs, rhs});

  
  
  
  //printf("herenot?");

  kernel_manager_t km(0);
  auto workspace_info = km.build(einsummable);

  //printf("hereyet?");

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();

  void* work;
  cudaMalloc(&work, size);

  dbuffer_t out = make_dbuffer(dtype, b*i);
  out.zeros();

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  

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



  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(einsummable,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out)) {
    printf("KERNEL16 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 16" <<std::endl;
  }
}

void kernel8(dtype_t dtype){
  uint64_t b = 128;
  uint64_t i = 4;
  
  auto scar = parse_with_ss<scalarop_t>("*[hole|f16@0,hole|f16@1]");
  


  einsummable_t einsummable = einsummable_t(
    {b, i},
    {{0, 1}, {1}},
    2,
    scar,
    castable_t::add);

 
  dbuffer_t lhs = make_dbuffer(dtype, b*i);
  dbuffer_t rhs = make_dbuffer(dtype, i);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(einsummable, {lhs, rhs});

  
  
  
  //printf("herenot?");

  kernel_manager_t km(0);
  auto workspace_info = km.build(einsummable);

  //printf("hereyet?");

  //if(!wsz){
  //  throw std::runtime_error("Invalid return!");
  //}

  uint64_t size = workspace_info.value().value();


  void* work;
  cudaMalloc(&work, size);

  dbuffer_t out = make_dbuffer(dtype, b*i);
  out.zeros();

  cutensorHandle_t handle;
  cutensorCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  

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



  vector<void const*> inns;
  inns.push_back(lh);
  inns.push_back(rh);

  //uint64_t size = km.workspace_size(reduction,ou,inns,handle);
  //void* work;
  //cudaMalloc(&work, size);

  //optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
  //  work, size };

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    work, size };

  km(einsummable,stream,ou,inns,workspace);

  //func(stream, handle, ou, inns);

  cudaStreamDestroy(stream);

  cudaMemcpy(out.ptr(), ou,sizeC, cudaMemcpyDeviceToHost);

  if(!is_close(out_ref, out,0.05f)) {
    printf("KERNEL8 ARE NOT CLOSE!\n");
    DOUT(out_ref);
    DOUT(out);
  }else{
     std::cout << "Elementwise operation successful for dtype "<<dtype << "for kernel 8" <<std::endl;
  }
}























int main(){

  // kernel1(dtype_t::f16, dtype_t::f32);
  
  // kernel2(dtype_t::f32); 
  
  // kernel3(dtype_t::f32);
  
  // kernel4(dtype_t::f32);

  // kernel5(dtype_t::f32);

  // kernel6(dtype_t::f32); 

  // kernel7(dtype_t::f32, dtype_t::f16);

  // kernel8(dtype_t::f16); 

  // kernel9(dtype_t::f16);

  kernel10(dtype_t::c64);
   
  // //complex contraction
  // kernel11(dtype_t::f16);  

  // kernel12(dtype_t::f16);

  // kernel13(dtype_t::f32);

  // kernel14(dtype_t::f16, dtype_t::f32);
   
  // kernel15(dtype_t::f32);

  // kernel16(dtype_t::f32);

  // kernel17(dtype_t::f32);

  // kernel18(dtype_t::f32);
 
  // kernel19(dtype_t::f32);  //power{-1}

  // kernel20(dtype_t::f32, dtype_t::f16);
  
  // kernel21(dtype_t::f16);

  // kernel22(dtype_t::f16);

  // kernel23(dtype_t::f16);

  // kernel24(dtype_t::f16);

  // printf("There is no kernel 25 due to counting error - kernel 25\n");

  // kernel26(dtype_t::f16);

  // kernel27(dtype_t::f16);

  // kernel28(dtype_t::f16);

  // kernel29(dtype_t::f16);

}