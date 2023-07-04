#pragma once
#include "../../base/setup.h"

#include "../../einsummable/taskgraph.h"
#include "../../einsummable/dbuffer.h"

#include "mpi_class.h"
#include "kernels.h"

#include <thread>

struct execute_taskgraph_settings_t {
  int num_apply_runner;
  int num_touch_runner;
  int num_send_runner;
  int num_recv_runner;
  int num_apply_kernel_threads; // for mkl threads and ew threads

  static execute_taskgraph_settings_t default_settings() {
    // standards says hardware_concurrency could return 0
    // if not computable or well defined
    int num_threads = std::max(1u, std::thread::hardware_concurrency());

    return execute_taskgraph_settings_t {
      .num_apply_runner = num_threads,
      .num_touch_runner = std::max(1, num_threads / 2),
      .num_send_runner  = 2,
      .num_recv_runner  = 2,
      .num_apply_kernel_threads = 1
    };
  }
  static execute_taskgraph_settings_t only_touch_settings() {
    // standards says hardware_concurrency could return 0
    // if not computable or well defined
    int num_threads = std::max(1u, std::thread::hardware_concurrency());

    return execute_taskgraph_settings_t {
      .num_apply_runner = 0,
      .num_touch_runner = std::max(1, num_threads / 2),
      .num_send_runner  = 2,
      .num_recv_runner  = 2,
      .num_apply_kernel_threads = 1
    };
  }
};

kernel_manager_t make_kernel_manager(taskgraph_t const& taskgraph);

void update_kernel_manager(kernel_manager_t& km, taskgraph_t const& taskgraph);

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute_taskgraph(
  taskgraph_t const& taskgraph,
  execute_taskgraph_settings_t const& settings,
  kernel_manager_t const& kernel_manager,
  mpi_t* mpi, // if this is nullptr, the taskgraph must be single-node
  map<int, buffer_t>& tensors);

