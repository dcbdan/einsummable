#pragma once
#include "../../base/setup.h"

#include "../base.h"

#include "../../engine/threadpool.h"

#include "../../engine/cpu/kernel_executor.h"
#include "../../engine/cpu/storage.h"

//#include <sys/mman.h>

struct cpu_mg_server_t : server_mg_base_t
{
  cpu_mg_server_t(
    communicator_t& c,
    uint64_t buffer_size,
    int num_threads)
    : server_mg_base_t(c), mem(make_buffer(buffer_size)), threadpool(num_threads)
  {
    //if(mlock(mem->data, mem->size) != 0) {
    //  //DOUT(strerror(errno));
    //  throw std::runtime_error("could not lock memory");
    //}
  }

  void execute_memgraph(memgraph_t const& memgraph, bool for_remap);

  buffer_t local_copy_data(int tid);

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

  void local_insert_tensors(map<int, tuple<int, buffer_t>> data);
  void local_erase_tensors(vector<int> const& tids);

  // for debugging
  void print();
private:
  map<int, memsto_t> data_locs;

  buffer_t mem;
  cpu_storage_t storage;

  cpu_kernel_executor_t kernel_executor;

  threadpool_t threadpool;
};

