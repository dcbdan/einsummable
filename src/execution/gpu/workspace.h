#pragma once
#include "../../base/setup.h"

#include "utility.h"

#include <cuda_runtime.h>

#include <mutex>

struct workspace_manager_t {
  ~workspace_manager_t();
 
  tuple<void*, uint64_t> borrow_workspace(int gpu, uint64_t size);

  void return_workspace(int gpu, void* mem, uint64_t sz);

  std::mutex m;

  vector<vector<tuple<void*, uint64_t>>> data;
};
