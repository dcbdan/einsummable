#include "utility.h"
#include <cuda_runtime_api.h>

cudaStream_t cuda_create_stream()
{
    cudaStream_t ret;
    auto         cudaError = cudaStreamCreate(&ret);
    if (cudaError != cudaSuccess) {
        throw std::runtime_error("cuda_create_stream... " + string(cudaGetErrorString(cudaError)));
    }
    return ret;
}

void const* offset_increment(void const* ptr, uint64_t offset)
{
    return static_cast<void const*>(static_cast<uint8_t const*>(ptr) + offset);
}
void* offset_increment(void* ptr, uint64_t offset)
{
    return static_cast<void*>(static_cast<uint8_t*>(ptr) + offset);
}

// prints float starting from ptr with count number of elements
void printFloatCPU(const float* cpu_ptr, int count)
{
    for (int i = 0; i < count; ++i) {
        // if cpu_ptr[i] is 0 then print 0 else print the value
        if (cpu_ptr[i] == 0) {
            printf("0 ");
        } else {
            printf("%.4f ", cpu_ptr[i]);
        }
    }
    printf("\n");
}
void printFloatGPU(const void* gpu_ptr, int count)
{
    float* cpu_ptr = (float*)malloc(count * sizeof(float));
    cudaMemcpy(cpu_ptr, gpu_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    printFloatCPU(cpu_ptr, count);
    free(cpu_ptr);
}

// calling cuda malloc to allocate memory for a given size
void* gpu_allocate_memory(size_t size, int device_id)
{
    cudaSetDevice(device_id);
    void* ret;
    auto  cudaError = cudaMalloc(&ret, size);
    if (cudaError != cudaSuccess) {
        // print an error message and the error code
        fprintf(stderr, "cudaStreamCreate failed with error: %s\n", cudaGetErrorString(cudaError));
        // printing the size and device id
        fprintf(stderr, "size: %zu, device_id: %d\n", size, device_id);
        throw std::runtime_error("cuda_malloc");
    }
    return ret;
}

void init_value(float* ptr, int count, float value)
{
    // malloc memory on cpu and cudamemcpy to gpu
    float* tmp = (float*)malloc(count * sizeof(float));
    // float *check = (float *)malloc(count * sizeof(float));
    for (int i = 0; i < count; ++i) {
        tmp[i] = value;
    }
    cudaMemcpy(ptr, tmp, count * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(tmp, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(check, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void handle_cutensor_error(cutensorStatus_t error, string msg)
{
    if (error != CUTENSOR_STATUS_SUCCESS) {
        if (msg == "") {
            msg = "handle_cutensor_error";
        }
        throw std::runtime_error(msg + ": " + string(cutensorGetErrorString(error)));
    }
}

void handle_cuda_error(cudaError_t error, string msg)
{
    // DOUT("error is " << error << " ... msg: " << msg);
    if (error != cudaSuccess) {
        if (msg == "") {
            msg = "handle_cuda_error";
        }
        throw std::runtime_error(msg + ": " + string(cudaGetErrorString(error)));
    }
}

void handle_cublas_error(cublasStatus_t error, string msg)
{
    if (error != CUBLAS_STATUS_SUCCESS) {
        if (msg == "") {
            msg = "handle_cublas_error";
        }
        throw std::runtime_error(msg + ": " + write_with_ss(error));
    }
}
