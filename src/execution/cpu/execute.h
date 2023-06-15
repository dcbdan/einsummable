#pragma once
#include "../../base/setup.h"

#include "../../einsummable/taskgraph.h"
#include "../../einsummable/dbuffer.h"

#include "mpi_class.h"

#include <thread>

struct settings_t {
  int num_apply_runner;
  int num_touch_runner;
  int num_send_runner;
  int num_recv_runner;
  int num_apply_kernel_threads; // for mkl threads and ew threads

  static settings_t default_settings() {
    // standards says hardware_concurrency could return 0
    // if not computable or well defined
    int num_threads = std::max(1u, std::thread::hardware_concurrency());

    return settings_t {
      .num_apply_runner = 1,
      .num_touch_runner = std::max(1, num_threads / 2),
      .num_send_runner  = 2,
      .num_recv_runner  = 2,
      .num_apply_kernel_threads = num_threads
    };
  }
};

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute(
  taskgraph_t const& taskgraph,
  settings_t const& settings,
  mpi_t& mpi,
  map<int, buffer_t>& tensors);

