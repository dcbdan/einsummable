#pragma once
#include "executetg.h"

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute_taskgraph_2(
  taskgraph_t const& taskgraph,
  execute_taskgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the taskgraph must be single-node
  map<int, buffer_t>& tensors);
