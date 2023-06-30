#include "../src/einsummable/einsummable.h"

#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"
#include "../src/einsummable/scalarop.h"

#include <chrono>

#include <unordered_map>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err));  } \
};

bool checkArrays3D(float* arr1, float* arr2, int totalSize) {
    for (int i = 0; i < totalSize; ++i) {
        if (arr1[i] != arr2[i]) {
                return false; // Arrays are different
            }
    }
    return true; // Arrays are the same
}


int nonZero(float* arr1,  int totalSize) {
    int count = 0;
    for (int i = 0; i < totalSize; ++i) {
        if (arr1[i] != 0) {
                count++; // Arrays are different
            }
    }
    return count; // Arrays are the same
}

int main(){
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;



    castable_t castable = castable_t::add;
    std::vector<int32_t> modeA{'m','h','k','v'};
    std::vector<int32_t> modeC{'m','v'};
    int32_t nmodeA = modeA.size();
    int32_t nmodeC = modeC.size();

    std::unordered_map<int32_t, uint64_t> extent;
    extent['m'] = 196;
    extent['v'] = 64;
    extent['h'] = 256;
    extent['k'] = 64;

    std::vector<uint64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<uint64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    
    auto func = build_cutensor_reduction(modeA,extentA,modeC,extentC,castable,dtype_t::f32);

    printf("Return successfully\n");


    cutensorHandle_t* handle;
    HANDLE_ERROR( cutensorCreate(&handle) );

    /**********************
     * Allocating data
     *********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *C_d;
    cudaMalloc((void**)&A_d, sizeA);
    cudaMalloc((void**)&C_d, sizeC);

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == NULL || C == NULL)
    {
        printf("Error: Host allocation of A, B, or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice);

    void* out;
    float* inn0;

    cudaMalloc(&inn0, sizeA);
    cudaMalloc((void**)&out, sizeC);

    cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(inn0, A, sizeA, cudaMemcpyHostToDevice);
     


    //float* out = C;
    //std::vector<float const*> inns;
    std::vector<void const*> inns;
    
    inns.push_back(A_d);
    //inns.push_back(A);



    std::vector<int64_t> extent_A;
    extent_A.reserve(extentA.size());
    for (const auto& element : extentA) {
        extent_A.push_back(static_cast<int64_t>(element));
    }

    std::vector<int64_t> extent_C;
    extent_C.reserve(extentC.size());
    for (const auto& element : extentC) {
        extent_C.push_back(static_cast<int64_t>(element));
    }



    // Call execute_contraction
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //func(stream,handle,out, inns);
    //func(stream, handle, C, inns);
    func(stream, handle, out, inns);



    cudaStreamDestroy(stream);

    cudaMemcpy(C, out,sizeC, cudaMemcpyDeviceToHost);

    printf("Reduction Executed!\n");

    for (size_t i = 0; i < modeA.size(); i++) {
        std::cout << modeA.data()[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < modeC.size(); i++) {
            std::cout << modeC.data()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << nmodeA << std::endl;
    std::cout << nmodeC << std::endl;

    std::cout << sizeA << std::endl;
    std::cout << sizeC << std::endl;

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                 &descA,
                 nmodeA,
                 extent_A.data(),
                 NULL /* stride */,
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                 &descC,
                 nmodeC,
                 extent_C.data(),
                 NULL /* stride */,
                 typeC, CUTENSOR_OP_IDENTITY));

    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    /**********************
     * Querry workspace
     **********************/

    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorReductionGetWorkspaceSize(handle, 
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

    dtype_t type = dtype_t::f32;

    void* ptr1;
    void* ptr2;
    float16_t alpha1, beta1;
    float alpha2, beta2;
    double alpha3, beta3;
    std::complex<float> alpha4(1.0f, 0.0f);
    std::complex<float> beta4(0.0f, 0.0f);

    if(type == dtype_t::f16){
      alpha1 = float16_t(1.0f);
      ptr1 = static_cast<void*>(&alpha1);
      beta1 = float16_t(0.0f);
      ptr2 = static_cast<void*>(&beta1);
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

    void const* alphayi = ptr1; 
    void const* betayi = ptr2; 

    //void* A_xd = (void*)inns[0];

    std::vector<void const*> innst;
    innst.push_back(A_d);
    //std::vector<float const*> innst;
    //innst.push_back(inn0);
    //void const* angst = (void*)innst[0];

    cutensorStatus_t err;
    err = cutensorReduction(handle, 
                /*(const void*)&alpha*/alphayi, innst[0], &descA, modeA.data(),
                /*(const void*)&beta*/betayi,  C_d, &descC, modeC.data(), 
                                     C_d, &descC, modeC.data(), 
                opReduce, typeCompute, work, worksize, 0 /* stream */);

    if(err != CUTENSOR_STATUS_SUCCESS)
            printf("ERROR: %s\n", cutensorGetErrorString(err) );

    std::cout << "worksize" << worksize << std::endl;

    float* out1 = new float[elementsC]();
    float* out2 = new float[elementsC]();
    cudaMemcpy(out2, C_d,sizeC, cudaMemcpyDeviceToHost);

    out1 = C;

    for (size_t i = 0; i < 10; i++) {
      std::cout << out1[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 10; i++) {
      std::cout << out2[i] << " ";
    }
    std::cout << std::endl;

    std::cout << nonZero(out1,elementsC) << std::endl;
    std::cout << nonZero(out2,elementsC) << std::endl;



    

    bool areEqual = checkArrays3D(out1, out2, elementsC);

    // Whether the results from our kernels and using cutensor directly is the same:
    if (areEqual) {
        printf("The arrays are the same.\n");
    } else {
        printf("The arrays are different.\n");
    }
    for (int i = 0; i < 100; ++i) {
       std::cout << "The element " << i << " for out 1 " << out1[i] << " for out 2 " << out2[i] << std::endl;
    }

    return 0;



}