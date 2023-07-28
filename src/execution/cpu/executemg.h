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

void update_kernel_manager(kernel_manager_t& km, memgraph_t const& memgraph);

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

// compile the taskgraph into a memgraph,
// execute the memgraph, and update
// data_locs, memory and storage accordingly.
//
// The server version should be called once and will do
// the extra coordinating. The client version should be
// called by all the other nodes.
void execute_taskgraph_as_memgraph_server(
  taskgraph_t const& memgraph,
  execute_memgraph_settings_t const& exec_settings,
  kernel_manager_t const& kernel_manager,
  allocator_settings_t const& alloc_settings,
  mpi_t* mpi,
  map<int, memsto_t>& data_locs,
  buffer_t memory,
  storage_t& storage);

void execute_taskgraph_as_memgraph_client(
  execute_memgraph_settings_t const& exec_settings,
  kernel_manager_t const& kernel_manager,
  int server_rank,
  mpi_t* mpi,
  map<int, memsto_t>& data_locs,
  buffer_t memory,
  storage_t& storage);

// These are common utilities for executing a taskgraph via a memgraph.
struct _tg_with_mg_helper_t {
  _tg_with_mg_helper_t(
    mpi_t* mpi,
    map<int, memsto_t>& data_locs,
    buffer_t& buffer,
    storage_t& storage,
    int server_rank);

  mpi_t* mpi;

  map<int, memsto_t>& data_locs;
  buffer_t& mem;
  storage_t& storage;

  int this_rank;
  int world_size;
  int server_rank;

  vector<uint64_t> recv_mem_sizes();
  void send_mem_size();

  map<int, memstoloc_t> recv_full_data_locs();
  void send_data_locs();

  void storage_remap_server(
    vector<vector<std::array<int, 2>>> const& storage_remaps);
  void storage_remap_client();

  void broadcast_memgraph(memgraph_t const& mg);
  memgraph_t recv_memgraph();

  void rewrite_data_locs_server(map<int, memstoloc_t> const& new_data_locs);
  void rewrite_data_locs_client();

  vector<vector<std::array<int, 2>>>
  create_storage_remaps(
    map<int, memstoloc_t> const& full_data_locs,
    map<int, memstoloc_t> const& inn_tg_to_loc);

  struct id_memsto_t {
    int id;
    memsto_t memsto;
  };
};

