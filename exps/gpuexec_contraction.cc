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

bool checkArraysDouble(double* arr1, double* arr2, int totalSize) {
    for (int i = 0; i < totalSize; ++i) {
        if (arr1[i] != arr2[i]) {
                return false; // Arrays are different
            }
    }
    return true; // Arrays are the same
}

int main(){
    //Initialize Einsummable and other variables for testing build_contraction
    //einsummable_t eins = einsummable_t({96,96,96,64,64,64}, { {0,4,5,2},{1,5,3,4} }, 4, scalarop_t::make_mul(), castable_t::add);
    einsummable_t eins = einsummable_t({96,96,96,64,64,64}, { {0,4,5,2},{1,5,3,4} }, 4, scalarop_t::make_mul(dtype_t::f64), castable_t::add);
    
    
    
    cutensorContractionDescriptor_t desc;

    cutensorHandle_t* handle;
    HANDLE_ERROR( cutensorCreate(&handle) );

    auto startTime1 = std::chrono::high_resolution_clock::now();




    //Test build_contraction
    build_contraction(&desc,handle,eins);

    auto endTime1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - startTime1).count();
    std::cout << "Execution time for build_contraction: " << duration1 << " microseconds" << std::endl;



    //Initialize lhs, rhs, out for testing execute_contraction
    einsummable_t e = eins;

    std::vector<int> modeA = e.inns[0];
    std::vector<int> modeB = e.inns[1];
    std::vector<int> modeC;
    for (int i = 0; i < e.out_rank; i++) {
        modeC.push_back(i);
    }

    int nmodeA = e.inns[0].size();
    int nmodeB = e.inns[1].size();
    int nmodeC = e.out_rank;

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


    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= e.join_shape[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= e.join_shape[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= e.join_shape[mode];

    // f64 -> double
    //typedef float floatTypeA;
    //typedef float floatTypeB;
    //typedef float floatTypeC;
    //typedef float floatTypeCompute;

    typedef double floatTypeA;
    typedef double floatTypeB;
    typedef double floatTypeC;
    typedef double floatTypeCompute;

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    //float *lhs, *rhs, *out;
    void *lhs, *rhs, *out;
    cudaMalloc((void**)&lhs, sizeA);
    cudaMalloc((void**)&rhs, sizeB);
    cudaMalloc((void**)&out, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    //for(int64_t i = 0; i < elementsA; i++)
    //    A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    //for(int64_t i = 0; i < elementsB; i++)
    //    B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    //for(int64_t i = 0; i < elementsC; i++)
     //   C[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((double) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((double) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((double) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    cudaMemcpy(out, C, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(lhs, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs, B, sizeB, cudaMemcpyHostToDevice);




    double alpha1 = 1.0;


    // Call execute_contraction
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    execute_contraction(stream,handle,&desc,out,lhs,rhs,e.inn_dtype(0));

    cudaStreamDestroy(stream);



    HANDLE_ERROR( cutensorDestroy(handle) );






    /*Run cutensor contraction without einsummable and stuffs*/







    // CUDA types
    //cudaDataType_t typeA = CUDA_R_32F;
    //cudaDataType_t typeB = CUDA_R_32F;
    //cudaDataType_t typeC = CUDA_R_32F;
    //cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    cudaDataType_t typeA = CUDA_R_64F;
    cudaDataType_t typeB = CUDA_R_64F;
    cudaDataType_t typeC = CUDA_R_64F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;

    std::cout << typeCompute << std::endl;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)1.0f;



    modeC = {'m','u','n','v'};
    modeA = {'m','h','k','n'};
    modeB = {'u','k','v','h'};
    nmodeA = modeA.size();
    nmodeB = modeB.size();
    nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    //printf("Define modes and extents\n");



    // Allocate on device
    void *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, sizeA);
    cudaMalloc((void**)&B_d, sizeB);
    cudaMalloc((void**)&C_d, sizeC);



    // Copy to device
    cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice);

    //printf("Allocate, initialize and transfer tensors\n");

    /* ***************************** */

    // Initialize cuTENSOR library
    HANDLE_ERROR( cutensorCreate(&handle) );

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, CUTENSOR_OP_IDENTITY ) );

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, CUTENSOR_OP_IDENTITY ) );

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR( cutensorInitTensorDescriptor( handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, CUTENSOR_OP_IDENTITY ) );

    //printf("Initialize cuTENSOR and tensor descriptors\n");

    /* ***************************** */

    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                A_d,
                &descA,
                &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                B_d,
                &descB,
                &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( cutensorGetAlignmentRequirement( handle,
                C_d,
                &descC,
                &alignmentRequirementC) );

    //printf("Query best alignment requirement for our pointers\n");

    std::cout << " A " << alignmentRequirementA << " B " << alignmentRequirementB << " C " << alignmentRequirementC << std::endl;

    /* ***************************** */

    // Create the Contraction Descriptor
    cutensorContractionDescriptor_t desc2;
    HANDLE_ERROR( cutensorInitContractionDescriptor( handle,
                &desc2,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    //printf("Initialize contraction descriptor\n");

    for (size_t i = 0; i < modeA.size(); i++) {
        std::cout << modeA.data()[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < modeB.size(); i++) {
            std::cout << modeB.data()[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < modeC.size(); i++) {
            std::cout << modeC.data()[i] << " ";
    }
    std::cout << std::endl;


    /* ***************************** */

    // Set the algorithm to use
    cutensorContractionFind_t find;
    HANDLE_ERROR( cutensorInitContractionFind(
                handle, &find,
                CUTENSOR_ALGO_DEFAULT) );

    //printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( cutensorContractionGetWorkspaceSize(handle,
                &desc2,
                &find,
                CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    //printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */

    // Create Contraction Plan
    cutensorContractionPlan_t plan;
    HANDLE_ERROR( cutensorInitContractionPlan(handle,
                                                &plan,
                                                &desc2,
                                                &find,
                                                worksize) );

    //printf("Create plan for contraction\n");

    /* ***************************** */

    cutensorStatus_t err;

    // Execute the tensor contraction
    err = cutensorContraction(handle,
                                &plan,
                        (void*)&alpha, A_d,
                                        B_d,
                        (void*)&beta,  C_d,
                                        C_d,
                                work, worksize, 0 /* stream */);
    cudaDeviceSynchronize();

    // Check for errors
    if(err != CUTENSOR_STATUS_SUCCESS)
    {
        printf("ERROR: %s\n", cutensorGetErrorString(err));
    }

    printf("Execute contraction from plan\n");


    //Check whether the results are the same
    //float* C_out = (float*)C_d;
    double* C_out0 = (double*)out;
    double* C_out = (double*)C_d;

    const int totalSize = static_cast<int>(sizeC);

    //float* out1 = new float[totalSize]();
    //float* out2 = new float[totalSize]();

    double* out1 = new double[totalSize]();
    double* out2 = new double[totalSize]();

    //cudaMemcpy(out1, out, sizeC,cudaMemcpyDeviceToHost);
    cudaMemcpy(out1, C_out0, sizeC,cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, C_out,sizeC, cudaMemcpyDeviceToHost);


    //bool areEqual = checkArrays3D(out1, out2, elementsC);
    bool areEqual = checkArraysDouble(out1, out2, elementsC);

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
