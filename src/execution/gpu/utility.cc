#include "utility.h"

// we need to define HANDLE_ERROR properly since it's not included in the header
// file defined:
// (https://docs.nvidia.com/cuda/cutensor/getting_started.html#determine-algorithm-and-workspace)
#define HANDLE_ERROR(x)                                                        \
{                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
}

cudaStream_t cuda_create_stream() {
  cudaStream_t ret;
  auto cudaError = cudaStreamCreate(&ret);
  if (cudaError != cudaSuccess) {
    // print error message and error code
    fprintf(stderr, "cudaStreamCreate failed with error: %s\n", cudaGetErrorString(cudaError));
    throw std::runtime_error("cuda_create_stream");
  }
  return ret;
}

// increment the pointer by the byte offset
// ONLY USE IF THE UNIT OF OFFSET IS BYTE
void* offset_increment(const void *ptr, int offset) {
  return (void *)((char *)ptr + offset);
}

// prints float starting from ptr with count number of elements
void printFloatCPU(const float *cpu_ptr, int count) {
  for (int i = 0; i < count; ++i) {
    printf("%.2f ", cpu_ptr[i]);
  }
  printf("\n");
}
void printFloatGPU(const void *gpu_ptr, int count) {
  float *cpu_ptr = (float *)malloc(count * sizeof(float));
  cudaMemcpy(cpu_ptr, gpu_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  printFloatCPU(cpu_ptr, count);
  free(cpu_ptr);
}

// calling cuda malloc to allocate memory for a given size
void* gpu_allocate_memory(size_t size, int device_id) {
  cudaSetDevice(device_id);
  void* ret;
  auto cudaError = cudaMalloc(&ret, size);
  if (cudaError != cudaSuccess) {
    // print an error message and the error code
    fprintf(stderr, "cudaStreamCreate failed with error: %s\n", cudaGetErrorString(cudaError));
    // printing the size and device id
    fprintf(stderr, "size: %zu, device_id: %d\n", size, device_id);
    throw std::runtime_error("cuda_malloc");
  }
  return ret;
}

void init_value(float *ptr, int count, float value) {
  // malloc memory on cpu and cudamemcpy to gpu
  float *tmp = (float *)malloc(count * sizeof(float));
  float *check = (float *)malloc(count * sizeof(float));
  for (int i = 0; i < count; ++i) {
    tmp[i] = value;
  }
  cudaMemcpy(ptr, tmp, count * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(tmp, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(check, ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void checkAlignment(cutensorHandle_t *handle, float *ptr,
                    cutensorTensorDescriptor_t desc) {
  uint32_t alignmentRequirement;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, ptr, &desc,
                                               &alignmentRequirement));

  if (alignmentRequirement != 16) {
    // print the alignment requirement
    std::cout << "*** Alignment requirement mismatch; alignment: "
              << alignmentRequirement << std::endl;
  }
}