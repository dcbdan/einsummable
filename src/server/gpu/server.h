#pragma once
#include "../../base/setup.h"

#include "../base.h"

struct gpu_mg_server_t : server_mg_base_t
{
  gpu_mg_server_t(
    communicator_t& c,
    // one buffer per gpu
    vector<uint64_t> buffer_sizes);

  void execute_memgraph(memgraph_t const& memgraph, bool for_remap);

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

private:
  map<int, memstoloc_t> data_locs;

  vector<buffer_t> mems;

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
};

