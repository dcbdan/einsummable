#include "workspace.h"

gpu_workspace_manager_t::~gpu_workspace_manager_t() {
  for(int gpu = 0; gpu != data.size(); ++gpu) {
    handle_cuda_error(cudaSetDevice(gpu), "~gpu_workspace_manager_t. set device");
    for(auto const& [mem, size]: data[gpu]) {
      handle_cuda_error(cudaFree(mem), "~workspace_maanger_t. cuda free");
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

  for(auto iter = data_here.begin(); iter != data_here.end(); ++iter) {
    auto const& [mem,size_] = *iter;
    if(size_ >= size) {
      data_here.erase(iter);
      return gpu_workspace_resource_t {
        .device = device,
        .ptr = mem,
        .size = size_
      };
    }
  }

  handle_cuda_error(cudaSetDevice(device));

  void* mem;
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
  auto const& [device, ptr, size];
  data[device].emplace_back(mem, sz);
}

