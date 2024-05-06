#include "workspace.h"
#include <driver_types.h>

gpu_workspace_manager_t::gpu_workspace_manager_t() {
  // on each device, preallocate 2 16 MB and 2 32 MB blocks
  // NOTE: this is hard-coded and we assume that we have enough memory for this
  auto num_gpus = 4;
  data.resize(num_gpus);
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    handle_cuda_error(cudaSetDevice(gpu), "gpu_workspace_manager_t. set device");
    // for (int i = 0; i < 2; ++i) {
    //   void* mem;
    //   handle_cuda_error(cudaMalloc(&mem, 16 * 1024 * 1024), "gpu_workspace_manager_t. cuda malloc");
    //   data[gpu].emplace_back(mem, 16 * 1024 * 1024);
    // }
    // for (int i = 0; i < num_gpus; ++i) {
    //   void* mem;
    //   handle_cuda_error(cudaMalloc(&mem, 32 * 1024 * 1024), "gpu_workspace_manager_t. cuda malloc");
    //   data[gpu].emplace_back(mem, 32 * 1024 * 1024);
    // }
  }
}

gpu_workspace_manager_t::~gpu_workspace_manager_t() {
  int flag = 0;
  for(int gpu = 0; gpu != data.size(); ++gpu) {
    // DOUT("The number of workspaces for device " << gpu << " is " << data[gpu].size() << "\n");
    handle_cuda_error(cudaSetDevice(gpu), "~gpu_workspace_manager_t. set device");
    for(auto const& [mem, size]: data[gpu]) {
      // TODO: for ffnn graph (ffnn_specific), 
      // there's error what():  ~workspace_maanger_t. cuda free: invalid argument
      cudaError_t error = cudaFree(mem);
      if(error != cudaSuccess) {
        std::cout << "error: " << cudaGetErrorString(error) << std::endl;
        // print out the mem, size
        std::cout << "mem: " << mem << std::endl;
        std::cout << "size: " << size << std::endl;
        flag = 1;
      }
    }
  }
  // TODO: currently not setting the device back to wtvr it was
}

optional<gpu_workspace_resource_t>
gpu_workspace_manager_t::try_to_acquire_impl(
    gpu_workspace_desc_t const& desc)
{
  std::unique_lock lk(m);

  auto const& [device, size] = desc;

  if(data.size() < device + 1) {
    data.resize(device+1);
  }
  auto& data_here = data[device];

  for(int i = 0; i != data_here.size(); ++i) {
  auto const& [mem, size_] = data_here[i];
   if(size_ >= size) {
     gpu_workspace_resource_t rsrc { 
        .device = device,
        .ptr = mem,
        .size = size_
     };
     data_here.erase(data_here.begin() + i);
     return rsrc;
   }
}

  handle_cuda_error(cudaSetDevice(device));

  void* mem;
  // DOUT("Allocating " << size / 1024 / 1024 << " MB on device " << device);
  handle_cuda_error(cudaMalloc(&mem, size));
  // TODO: currently not setting device back to wtvr it was

  return gpu_workspace_resource_t {
    .device = device,
    .ptr = mem,
    .size = size
  };
}

void gpu_workspace_manager_t::release_impl(
  gpu_workspace_resource_t const& resource)
{
  std::unique_lock lk(m);
  auto const& [device, ptr, size] = resource;
  // DOUT("Releasing " << size / 1024 / 1024 << " MB on device " << device)
  data[device].emplace_back(ptr, size);
}

