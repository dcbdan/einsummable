#pragma once
#include "../../base/setup.h"

#include "../base.h"

#include "../../engine/threadpool.h"
#include "../../engine/exec_state.h"

#include "../../engine/cpu/kernel_executor.h"
#include "../../engine/cpu/mg/storage.h"

//#include <sys/mman.h>

struct cpu_mg_server_t : server_mg_base_t
{
  cpu_mg_server_t(
    communicator_t& c_,
    uint64_t buffer_size,
    int num_threads,
    int num_channels_per_move = 1,
    exec_state_t::priority_t priority_type = exec_state_t::priority_t::given)
    : server_mg_base_t(c_), mem(make_buffer(buffer_size)),
      threadpool(num_threads),
      num_channels_per_move(num_channels_per_move),
      priority_type(priority_type)
  {
    //if(mlock(mem->data, mem->size) != 0) {
    //  //DOUT(strerror(errno));
    //  throw std::runtime_error("could not lock memory");
    //}

    if(num_channels_per_move > num_threads) {
      throw std::runtime_error("invalid num channels per move");
    }

    // TODO: verify that num_channels_per_move is the same on all nodes
  }

  int get_num_threads() const {
    return threadpool.num_runners();
  }

  threadpool_t* get_cpu_threadpool() { return &threadpool; }

  void execute_memgraph(
    memgraph_t const& memgraph,
    bool for_remap,
    map<string, scalar_t> const& scalar_vars);

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

  int num_channels_per_move;

  exec_state_t::priority_t priority_type;
};



// // This is a server_base object that executes all taskgraphs by first
// // converting them into memgraphs, but without any distribution (runs locally without communicator)
// struct server_cpu_mg_local_t : server_base_t {
//   server_cpu_mg_local_t(
//     uint64_t buffer_size,
//     int num_threads,
//     int num_channels_per_move = 1,
//     exec_state_t::priority_t priority_type = exec_state_t::priority_t::given
//     allocator_settings_t s = allocator_settings_t::default_settings())
//     : server_dist_base_t(c), alloc_settings(s),
//       make_parallel_partialize_groups_(false),
//       use_storage_(true),
//       split_off_inputs_(true),
//       mem(make_buffer(buffer_size)),
//       threadpool(num_threads),
//       num_channels_per_move(num_channels_per_move),
//       priority_type(priority_type)
//   {}

//   void set_parallel_partialize(bool new_val) {
//     this->make_parallel_partialize_groups_ = new_val;
//   }

//   void set_use_storage(bool new_val) {
//     this->use_storage_ = new_val;
//   }

//   void set_split_off_inputs(bool new_val) {
//     this->split_off_inputs_ = new_val;
//   }

//   // this must be called from all locations
//   virtual void execute_memgraph(
//     memgraph_t const& memgraph,
//     bool for_remap,
//     map<string, scalar_t> const& scalar_vars = {}) = 0;
//   // the for_remap should be true if the computation is just a remap instead of
//   // a general taskgraph

//   // // server, client pairs {{{
//   // void execute_tg_server(
//   //   taskgraph_t const& taskgraph,
//   //   map<string, scalar_t> const& scalar_vars);
//   // void execute_tg_client();

//   void remap_server(remap_relations_t const& remap_relations);
//   // void remap_client();

//   struct make_mg_info_t {
//     vector<uint64_t> mem_sizes;
//     map<int, memstoloc_t> data_locs;
//     vector<int> which_storage;
//   };
//   virtual make_mg_info_t recv_make_mg_info() = 0;
//   virtual void           send_make_mg_info() = 0;

//   void storage_remap_server(
//     vector<vector<std::array<int, 2>>> const& remaps) = 0;
//   virtual void storage_remap_client() = 0;

//   void rewrite_data_locs_server(map<int, memstoloc_t> const& out_tg_to_loc) = 0;
//   virtual void rewrite_data_locs_client() = 0;
//   // }}}

//   static 
//   vector<vector<std::array<int, 2>>>
//   create_storage_remaps(
//   map<int, memstoloc_t> const& full_data_locs,
//   map<int, memstoloc_t> const& inn_tg_to_loc)

//   bool make_parallel_partialize_groups() const {
//     return make_parallel_partialize_groups_;
//   }

// public:
//   allocator_settings_t alloc_settings;

//   bool make_parallel_partialize_groups_;
//   bool use_storage_;
//   bool split_off_inputs_;

// //Got all the private fields from cpu_mg_server_t
// private:
//   map<int, memsto_t> data_locs;

//   buffer_t mem;
//   cpu_storage_t storage;

//   cpu_kernel_executor_t kernel_executor;

//   threadpool_t threadpool;

//   exec_state_t::priority_t priority_type;
// };
