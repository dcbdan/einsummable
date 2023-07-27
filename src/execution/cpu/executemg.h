#pragma once
#include "../../base/setup.h"

#include "../../einsummable/memgraph.h"

#include "mpi_class.h"
#include "kernels.h"
#include "storage.h"

#include <thread>

struct execute_memgraph_settings_t {
  int num_apply_runner;
  int num_cache_runner;
  int num_send_runner;
  int num_recv_runner;

  static execute_memgraph_settings_t default_settings() {
    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    return execute_memgraph_settings_t {
      .num_apply_runner = num_threads,
      .num_cache_runner = 2,
      .num_send_runner = 2,
      .num_recv_runner = 2
    };
  }
};

void execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the memgraph must be single-node
  buffer_t memory);

// TODO
void execute_memgraph(
  memgraph_t const& memgraph,
  execute_memgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the memgraph must be single-node
  buffer_t memory,
  storage_t& storage);
