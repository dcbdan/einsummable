#pragma once
#include "../../base/setup.h"

#include "../base.h"

#include "../../engine/gpu/storage_manager.h"
#include "../../engine/gpu/gpu_kernel_manager.h"
#include "../../engine/gpu/workspace.h"
#include "../../engine/gpu/stream_pool.h"

struct gpu_mg_server_t : server_mg_base_t
{
  gpu_mg_server_t(
    communicator_t& c,
    // one buffer per gpu
    vector<uint64_t> buffer_sizes);

  gpu_mg_server_t(
    vector<uint64_t> buffer_sizes, uint64_t storage_size);

  gpu_mg_server_t(
    communicator_t& c,
    // one buffer per gpu
    vector<uint64_t> buffer_sizes,
    uint64_t storage_size);

  ~gpu_mg_server_t();

  bool has_storage() const;

  void execute_memgraph(
    memgraph_t const& memgraph,
    bool for_remap,
    map<string, scalar_t> const& scalar_vars);

  // server, client pairs {{{
  make_mg_info_t recv_make_mg_info();
  void           send_make_mg_info();

  void storage_remap_server(
    vector<vector<std::array<int, 2>>> const& remaps);
  void storage_remap_client();

  void rewrite_data_locs_server(map<int, memstoloc_t> const& out_tg_to_loc);
  void rewrite_data_locs_client();
  // }}}

  int local_get_max_tid() const;
  int local_candidate_location() const;
  int loc_to_compute_node(int loc) const;

  buffer_t local_copy_data(int tid);

  void local_insert_tensors(map<int, tuple<int, buffer_t>> data);
  void local_erase_tensors(vector<int> const& tids);

  bool is_local_gpu(int global_loc) const;

  void debug_mem(int device, uint64_t counts);

  static communicator_t null_comm;
  vector<void*> mems;

private:
// location is to the local gpu, not global
  map<int, memstoloc_t> data_locs;



  // Example:
  //   compute-node 0: 4 gpus,
  //   compute-node 1: 3 gpus,
  //   compute-node 2: 2 gpus,
  //   compute-node 3: 5 gpus
  // num_gpus_per_node = {4,3,2,5}
  // start_gpus_per_node = {0,4,7,9}
  vector<int> num_gpus_per_node;
  vector<int> start_gpus_per_node;

  vector<uint64_t> all_mem_sizes;

  int which_local_gpu(int loc) const;
  int get_num_gpus() const;
  vector<int> get_which_storage() const;

  vector<kernel_manager_t> kernel_managers;

  std::shared_ptr<gpu_storage_t> storage;

  // NOTE: Change the streams per device as needed
  // 5 is enough in initial experiments
  int num_streams_per_device = 5;

  streampool_t stream_pool;
};
