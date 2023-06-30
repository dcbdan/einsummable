#include "../src/einsummable/einsummable.h"

#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"

#include <chrono>

#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <stdio.h>


void func1(){
    vector<uint64_t> join_shape{2,2};
    cutensor_elementwise_op_t op;
    op.join_shape = join_shape;
    cutensor_elementwise_op_t::arg_t a0 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_SQRT,{0,1}};
    cutensor_elementwise_op_t::arg_t a1 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
    cutensor_elementwise_op_t::arg_t a2 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
    cutensorOperator_t op_01_2 = CUTENSOR_OP_ADD;
    cutensorOperator_t op_0_1 = CUTENSOR_OP_ADD;

    cutensor_elementwise_op_t::binary_t ter_op{
      op_01_2,
      a1,
      a2
    };

    op.op = ter_op;

    auto func = build_cutensor_elementwise(op);

    size_t sizeA = sizeof(float) * 4;
    size_t sizeB = sizeof(float) * 4;
    size_t sizeC = sizeof(float) * 4;

    float *A = (float*) malloc(sizeA);
    float *B = (float*) malloc(sizeA);
    float *C = (float*) malloc(sizeA);
    float *D = (float*) malloc(sizeA);

    for (int64_t i = 1; i < 5; i++)
        A[i-1] = i;
    for (int64_t i = 5; i < 9; i++)
        B[i-5] = i;
    for (int64_t i = 9; i < 13; i++)
        C[i-9] = i;
    for (int64_t i = 0; i < 4; i++)
        D[i] = 1;
    

    void* out;
    void* inn0;
    void* inn1;
    void* inn2;

    cudaMalloc(&inn0, sizeA);
    cudaMalloc(&inn1, sizeA);
    cudaMalloc(&inn2, sizeA);
    cudaMalloc(&out, sizeC);

    cudaMemcpy(out, D, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(inn0, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(inn1, B, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(inn2, C, sizeA, cudaMemcpyHostToDevice);

    std::vector<void const*> inns;
    inns.push_back(inn0);
    inns.push_back(inn1);
    inns.push_back(inn2);


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cutensorHandle_t* handle;
    cutensorCreate(&handle);

    func(stream,handle,out, inns);

    cudaStreamDestroy(stream);

    cudaMemcpy(D, out,sizeC, cudaMemcpyDeviceToHost);

    float* out1 = new float[4]();

    out1 = (float*)D;

    for (int i = 0; i < 4; ++i) {
       std::cout << "The element " << i << " for out 1 " << out1[i] << std::endl;
    }
}

int main(){
    vector<uint64_t> join_shape{2,2};
    cutensor_elementwise_op_t op;
    op.join_shape = join_shape;
    cutensor_elementwise_op_t::arg_t a0 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_SQRT,{0,1}};
    cutensor_elementwise_op_t::arg_t a1 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
    cutensor_elementwise_op_t::arg_t a2 {scalar_t::one(dtype_t::f32),CUTENSOR_OP_IDENTITY,{0,1}};
    cutensorOperator_t op_01_2 = CUTENSOR_OP_ADD;
    cutensorOperator_t op_0_1 = CUTENSOR_OP_ADD;

    cutensor_elementwise_op_t::ternary_t ter_op{
      op_01_2,
      op_0_1,
      a0,
      a1,
      a2
    };

    op.op = ter_op;

    auto func = build_cutensor_elementwise(op);

    size_t sizeA = sizeof(float) * 4;
    size_t sizeB = sizeof(float) * 4;
    size_t sizeC = sizeof(float) * 4;

    float *A = (float*) malloc(sizeA);
    float *B = (float*) malloc(sizeA);
    float *C = (float*) malloc(sizeA);
    float *D = (float*) malloc(sizeA);

    for (int64_t i = 1; i < 5; i++)
        A[i-1] = i;
    for (int64_t i = 5; i < 9; i++)
        B[i-5] = i;
    for (int64_t i = 9; i < 13; i++)
        C[i-9] = i;
    for (int64_t i = 0; i < 4; i++)
        D[i] = 0;
    

    void* out;
    void* inn0;
    void* inn1;
    void* inn2;

    cudaMalloc(&inn0, sizeA);
    cudaMalloc(&inn1, sizeA);
    cudaMalloc(&inn2, sizeA);
    cudaMalloc(&out, sizeC);

    cudaMemcpy(out, D, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(inn0, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(inn1, B, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(inn2, C, sizeA, cudaMemcpyHostToDevice);

    std::vector<void const*> inns;
    inns.push_back(inn0);
    inns.push_back(inn1);
    inns.push_back(inn2);


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cutensorHandle_t* handle;
    cutensorCreate(&handle);

    func(stream,handle,out, inns);

    cudaStreamDestroy(stream);

    cudaMemcpy(D, out,sizeC, cudaMemcpyDeviceToHost);

    float* out1 = new float[4]();

    out1 = (float*)D;

    for (int i = 0; i < 4; ++i) {
       std::cout << "The element " << i << " for out 1 " << out1[i] << std::endl;
    }


    func1();
}