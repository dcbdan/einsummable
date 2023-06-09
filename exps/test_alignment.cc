#include "../src/einsummable/einsummable.h"

#include "../src/execution/gpu/kernels.h"

#include "../src/einsummable/reference.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

int main(){
    typedef float floatTypeA;

    cudaDataType_t typeA = CUDA_R_32F;

    std::vector<int> modeA{'w','h','c','n'};
    int nmodeA = modeA.size();

    std::unordered_map<int, int64_t> extent;
    extent['h'] = 128;
    extent['w'] = 32;
    extent['c'] = 128;
    extent['n'] = 128;

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    
    size_t sizeA = sizeof(floatTypeA) * elementsA;

    void *A_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));

    floatTypeA *A;
    HANDLE_CUDA_ERROR(cudaMallocHost((void**) &A, sizeof(floatTypeA) * elementsA));

    for (size_t i = 0; i < elementsA; i++)
    {
        A[i] = (((float) rand())/RAND_MAX)*100;
    }

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));

    

    cutensorStatus_t err;
    cutensorHandle_t* handle;
    HANDLE_ERROR( cutensorCreate(&handle) );

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY));

    

    int diffs[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const int iterations = 100; // Specify the desired number of iterations here

    for (int diffIndex = 0; diffIndex < 8; ++diffIndex) {
        int diff = diffs[diffIndex];
        for (int i = 0; i < iterations; ++i) {
            size_t offset = i * diff;
            uint32_t alignmentRequirementA;
            char* offsetPtr = static_cast<char*>(A_d);
            offsetPtr += offset;
            void* newPtr = static_cast<void*>(offsetPtr);
            HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                  newPtr,
                  &descA,
                  &alignmentRequirementA));
            std::cout << "Alignment requirement at offset " << offset << " with diff " << diff << ": " << alignmentRequirementA << std::endl;
            
        }
    }

    if (A) free(A);

    if (A_d) cudaFree(A_d);

    return 0;

}