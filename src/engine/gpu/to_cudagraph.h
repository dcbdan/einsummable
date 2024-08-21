#pragma once

#include "../../einsummable/memgraph.h"
#include "gpu_kernel_manager.h"
#include "utility.h"

// TODO: still does not support evict and load
cudaGraph_t compile_cuda_graph(
  memgraph_t const& memgraph,
  vector<kernel_manager_t>& kms,
  vector<void*> mems,
  map<string, scalar_t> const& scalar_vars);

