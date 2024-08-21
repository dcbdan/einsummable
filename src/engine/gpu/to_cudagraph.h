#pragma once

#include "../../einsummable/memgraph.h"
#include "gpu_kernel_manager.h"
#include "utility.h"

struct cg_event_t {
  cudaEvent_t beg;
  cudaEvent_t end;

  cg_event_t();
  cg_event_t(cudaEvent_t beg, cudaEvent_t end);
  ~cg_event_t(); 
  float elapsed_time() const; 
};

// TODO: still does not support evict and load
tuple<
  cudaGraph_t,
  map<int, cg_event_t>>
compile_cuda_graph(
  memgraph_t const& memgraph,
  vector<kernel_manager_t>& kms,
  vector<void*> mems,
  map<string, scalar_t> const& scalar_vars);

