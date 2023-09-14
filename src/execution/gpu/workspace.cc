#include "workspace.h"

workspace_manager_t::~workspace_manager_t() {
  for(int gpu = 0; gpu != data.size(); ++gpu) {
    handle_cuda_error(cudaSetDevice(gpu), "~workspace_manager_t. set device");  
    for(auto const& [mem, size]: data[gpu]) {      
      handle_cuda_error(cudaFree(mem), "~workspace_maanger_t. cuda free");
    }
  }
  // TODO: currently not setting the device back to wtvr it was
}

tuple<void*, uint64_t> 
workspace_manager_t::borrow_workspace(int gpu, uint64_t size) {
  std::unique_lock lk(m);

  if(data.size() < gpu + 1) {
    data.resize(gpu+1);
  }
  auto& data_here = data[gpu];

  for(auto iter = data_here.begin(); iter != data_here.end(); ++iter) {
    auto workspace = *iter; 
    auto const& [mem,size_] = workspace;
    if(size_ >= size) {
      data_here.erase(iter);
      return workspace;
    }
  }

  handle_cuda_error(cudaSetDevice(gpu)); 

  void* mem;
  handle_cuda_error(cudaMalloc(&mem, size));

  // TODO: currently not setting device back to wtvr it was

  return {mem,size};
}

void workspace_manager_t::return_workspace(int gpu, void* mem, uint64_t sz) 
{
  std::unique_lock lk(m);
  data[gpu].emplace_back(mem, sz);
}

