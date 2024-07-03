#include "server.h"

#include "../../engine/exec_graph.h"
#include "../../engine/exec_state.h"
#include "../../engine/managers.h"
#include <cuda_profiler_api.h>

gpu_mg_server_t::gpu_mg_server_t(
  communicator_t& c,
  // one buffer per gpu
  vector<uint64_t> buffer_sizes)
  : server_mg_base_t(c, allocator_settings_t::gpu_alignment_settings())
{
  int this_rank = comm.get_this_rank();
  int world_size = comm.get_world_size();
  all_mem_sizes.reserve(world_size);
  if(this_rank == 0) {
    num_gpus_per_node.push_back(buffer_sizes.size());
    all_mem_sizes = buffer_sizes;

    for(int rank = 1; rank != world_size; ++rank) {
      vector<uint64_t> other_buffer_sizes = comm.recv_vector<uint64_t>(rank);

      num_gpus_per_node.push_back(other_buffer_sizes.size());
      vector_concatenate_into(all_mem_sizes, other_buffer_sizes);
    }
    comm.broadcast_vector(num_gpus_per_node);
    comm.broadcast_vector(all_mem_sizes);
  } else {
    comm.send_vector(0, buffer_sizes);
    num_gpus_per_node = comm.recv_vector<int>(0);
    all_mem_sizes = comm.recv_vector<uint64_t>(0);
  }

  start_gpus_per_node.resize(world_size);
  std::exclusive_scan(
    num_gpus_per_node.begin(), num_gpus_per_node.end(),
    start_gpus_per_node.begin(),
    0);

  // allocate memory on each gpu into mems
  int num_gpus_here = num_gpus_per_node[this_rank];
  kernel_managers.reserve(num_gpus_here);
  for (int i = 0; i < num_gpus_here; i++) {
    mems.push_back(gpu_allocate_memory(buffer_sizes[i], i));
    kernel_managers.emplace_back(i);
  }

  // print all mems
  // for (auto i = 0; i < mems.size(); i++){
  //   DOUT("mems GPU[" << i << "]: " << mems[i]);
  // }

  // initialize the stream pool now that we have num_gpus_per_node
  stream_pool.initialize(num_streams_per_device, num_gpus_per_node[this_rank]);

  // When creating the gpu server, also enable peer access to have best transfer performance
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    for (int j = 0; j < deviceCount; ++j) {
      if (i != j) {
        cudaSetDevice(i);
        // enable p2p access
        cudaDeviceEnablePeerAccess(j, 0);
        // enable host memory mapping access by cudaHostAlloc
        cudaSetDeviceFlags(cudaDeviceMapHost);
      }
    }
  }

  // check if the peer access is really enabled
  for (int i = 0; i < deviceCount; ++i) {
    for (int j = 0; j < deviceCount; ++j) {
      if (i != j) {
        int canAccessPeer;
        cudaSetDevice(i);
        cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
        if (canAccessPeer != 1){
          throw std::runtime_error("Peer access is not enabled");
        }
      }
    }
  }
}

gpu_mg_server_t::gpu_mg_server_t(
  communicator_t& c,
  vector<uint64_t> buffer_sizes,
  uint64_t storage_size)
  : gpu_mg_server_t(c, buffer_sizes)
{
  storage = std::make_shared<gpu_storage_t>(storage_size);
}

bool gpu_mg_server_t::has_storage() const {
  return bool(storage);
}

void gpu_mg_server_t::execute_memgraph(
  memgraph_t const& memgraph,
  bool for_remap,
  map<string, scalar_t> const& scalar_vars)
{
  // 1. make the exec graph
  // 2. create the resource manager
  // 3. create the exec state and call the event loop
  auto initial = std::chrono::high_resolution_clock::now();
  // DOUT("Making exec graph...");
  // Note: the kernel_manager must outlive the exec graph
  exec_graph_t graph =
    exec_graph_t::make_gpu_exec_graph(
      memgraph, comm.get_this_rank(), kernel_managers,
      num_gpus_per_node[comm.get_this_rank()], mems,
      scalar_vars);
  // DOUT("Finished making exec graph...");

  vector<rm_ptr_t> rms {
    rm_ptr_t(new gpu_workspace_manager_t()),
    rm_ptr_t(new group_manager_t()),
    rm_ptr_t(new global_buffers_t(mems)),
    rm_ptr_t(new streampool_manager_t(stream_pool))
  };
  if(storage) {
    rms.push_back(rm_ptr_t(new gpu_storage_manager_t(storage.get())));
  }
  rm_ptr_t resource_manager(new resource_manager_t(rms));

  // exec_state_t state(graph, resource_manager, exec_state_t::priority_t::dfs);
  exec_state_t state(graph, resource_manager);

  // DOUT("Executing...");
  // print the execution time of event_loop()
  if (!for_remap){
    cudaProfilerStart();
  }
  auto start = std::chrono::high_resolution_clock::now();
  state.event_loop();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  // print the duration in milliseconds with 4 decimal places
  if (!for_remap){
    DOUT("Event Loop finished. Time: " << duration.count() << " ms");
    cudaProfilerStop();
  }
  auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end-initial);
  if (duration2.count() - duration.count() > 10 && !for_remap){
    DOUT("Execute memgraph finished. Time: " << duration2.count() << " ms");
  }
}

// memstoloc_t is not a contiguous data structure,
// so we have this contiguous data structure that is basically
// holding the contiguous analogue of tuple<int, memstoloc_t>.
struct id_memstoloc_t {
  id_memstoloc_t() {}

  id_memstoloc_t(int id, memstoloc_t const& memstoloc)
    : id(id)
  {
    if(memstoloc.is_memloc()) {
      is_memloc = true;
      data.memloc = memstoloc.get_memloc();
    } else {
      is_memloc = false;
      data.stoloc = memstoloc.get_stoloc();
    }
  }

  int id;
  bool is_memloc;
  union {
    memloc_t memloc;
    stoloc_t stoloc;
  } data;

  memstoloc_t as_memstoloc() const {
    if(is_memloc) {
      return memstoloc_t(data.memloc);
    } else {
      return memstoloc_t(data.stoloc);
    }
  }
};

server_mg_base_t::make_mg_info_t
gpu_mg_server_t::recv_make_mg_info()
{
  make_mg_info_t ret {
    .mem_sizes = all_mem_sizes,
    .data_locs = data_locs,
    .which_storage = get_which_storage()
  };
  // Note: ret.data_locs is with respect to global gpu locations
  // Since this is rank 0, they should be the same 
  auto& all_data_locs = ret.data_locs;

  int world_size = comm.get_world_size();
  for(int rank = 1; rank != world_size; ++rank) {
    // info is local
    for(auto const& info: comm.recv_vector<id_memstoloc_t>(rank)) {
      // TODO: need to convert each location from local to global
      throw std::runtime_error("not implemented");
      all_data_locs.insert({info.id, info.as_memstoloc()});
    }
  }

  return ret;
}

void gpu_mg_server_t::send_make_mg_info()
{
  vector<id_memstoloc_t> ds;
  ds.reserve(data_locs.size());
  for(auto const& [id, memstoloc]: data_locs) {
    ds.emplace_back(id, memstoloc);
  }
  comm.send_vector(0, ds);
}

void gpu_mg_server_t::storage_remap_server(
  vector<vector<std::array<int, 2>>> const& remaps)
{
  if(!bool(storage)) {
    throw std::runtime_error("storage not initialized");
  }

  int world_size = comm.get_world_size();

  storage->remap(remaps[0]);

  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_vector(dst, remaps[dst]);
  }
}

void gpu_mg_server_t::storage_remap_client()
{
  if(!bool(storage)) {
    throw std::runtime_error("storage not initialized");
  }

  auto remap = comm.recv_vector<std::array<int, 2>>(0);
  storage->remap(remap);
}

void gpu_mg_server_t::rewrite_data_locs_server(
  map<int, memstoloc_t> const& new_data_locs)
{
  auto get_compute_node = [&](memstoloc_t const& x) {
    if(x.is_memloc()) {
      return loc_to_compute_node(x.get_memloc().loc);
    } else {
      // stoloc_t::loc is a storage location, but we are
      // assuming that storage loc == compute-node
      return x.get_stoloc().loc;
    }
  };

  int world_size = comm.get_world_size();
  vector<vector<id_memstoloc_t>> items(world_size);
  data_locs.clear();
  for(auto const& [id, memstoloc]: new_data_locs) {
    int compute_node = get_compute_node(memstoloc);
    if(compute_node == 0) {
      data_locs.insert({ id, memstoloc });
    } else {
      items[compute_node].emplace_back(id, memstoloc);
    }
  }

  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_vector(dst, items[dst]);
  }
}

void gpu_mg_server_t::rewrite_data_locs_client() {
  data_locs.clear();
  auto new_data_locs = comm.recv_vector<id_memstoloc_t>(0);
  for(auto const& info: new_data_locs) {
    data_locs.insert({info.id, info.as_memstoloc()});
  }
}

int gpu_mg_server_t::local_get_max_tid() const {
  return data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;
}

int gpu_mg_server_t::local_candidate_location() const {
  return start_gpus_per_node[comm.get_this_rank()];
}

int gpu_mg_server_t::loc_to_compute_node(int loc) const {
  auto iter = std::upper_bound(
    start_gpus_per_node.begin(),
    start_gpus_per_node.end(),
    loc);
  return std::distance(start_gpus_per_node.begin(), iter) - 1;
}

buffer_t gpu_mg_server_t::local_copy_data(int tid) {
  int this_rank = comm.get_this_rank();
  auto const& d = data_locs.at(tid);
  if(d.is_memloc()) {
    auto const& [offset, size, global_gpu] = d.get_memloc();
    int local_gpu = which_local_gpu(global_gpu);

    // copy from gpu at mems[local_gpu] from [offset, offset+size) into
    // buffer at this location and return
    auto ret_buffer = make_buffer(size);
    cudaError_t error = cudaMemcpy(
      ret_buffer->data,
      increment_void_ptr(mems[local_gpu], offset),
      ret_buffer->size,
      cudaMemcpyDeviceToHost);

    if(error != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed");
    }

    // DLINEOUT(dbuffer_t(dtype_t::f32, ret_buffer));

    return ret_buffer;
  } else if(d.is_stoloc()) {
    auto const& [sto_loc, sto_id] = d.get_stoloc();
    if(sto_loc != this_rank) {
      throw std::runtime_error("invalid storage location");
    }
    return storage->read(sto_id);
  } else {
    throw std::runtime_error("local_copy_data should not reach");
  }
}

void gpu_mg_server_t::local_insert_tensors(map<int, tuple<int, buffer_t>> data) {
  int this_rank = comm.get_this_rank();
  int const& num_gpus_here = num_gpus_per_node[this_rank];

  // create some allocator and fill them with current data
  vector<allocator_t> allocators;
  allocators.reserve(num_gpus_here);
  for(int local_gpu = 0; local_gpu != num_gpus_here; ++local_gpu) {
    allocators.emplace_back(
      // mems[local_gpu]->size,
      all_mem_sizes[start_gpus_per_node[this_rank] + local_gpu],
      alloc_settings);
  }

  for(auto const& [tid, memstoloc]: data_locs) {
    if(memstoloc.is_memloc()) {
      auto const& [offset, size, global_gpu] = memstoloc.get_memloc();
      int local_gpu = which_local_gpu(global_gpu);
      auto& allocator = allocators[local_gpu];
      if(!allocator.allocate_at_without_deps(offset, size)) {
        throw std::runtime_error("could not setup allocator");
      }
    }
  }

  for(auto const& [tid, loc_tensor]: data) {
    auto const& [global_gpu, tensor] = loc_tensor;
    int local_gpu = which_local_gpu(global_gpu);
    auto& allocator = allocators[local_gpu];
    auto maybe_offset = allocator.allocate_without_deps(tensor->size);

    memstoloc_t memstoloc;
    if(maybe_offset) {
      auto const& offset = maybe_offset.value();

      // copy tensor onto gpu memory at mems[local_gpu] + offset
      cudaMemcpy(
        increment_void_ptr(mems[local_gpu], offset),
        tensor->data,
        tensor->size,
        cudaMemcpyHostToDevice);

      memloc_t memloc {
        .offset = offset,
        .size = tensor->size,
        .loc = global_gpu
      };

      memstoloc = memstoloc_t(memloc);
    } else {
      if (!has_storage()){
        throw std::runtime_error("could not allocate memory; not using storage");
      }
      int id = 1 + storage->get_max_id();

      // DOUT("Inserting into storage... id: " << id << " size: " << tensor->size);
      storage->write(tensor, id);

      stoloc_t stoloc {
        .loc = comm.get_this_rank(),
        .id = id
      };

      memstoloc = memstoloc_t(stoloc);
    }

    auto [_, did_insert] = data_locs.insert({tid, memstoloc});
    if(!did_insert) {
      throw std::runtime_error("this tid is already in data locs");
    }
  }
}

void gpu_mg_server_t::local_erase_tensors(vector<int> const& tids) {
  for(auto const& tid: tids) {
    auto iter = data_locs.find(tid);
    if(iter == data_locs.end()) {
      throw std::runtime_error("no tid to delete");
    }
    memstoloc_t const& memstoloc = iter->second;
    if(memstoloc.is_memloc()) {
      // nothing to do
    } else {
      auto const& [_,id] = memstoloc.get_stoloc();
      storage->remove(id);
    }
    data_locs.erase(iter);
  }
}

int gpu_mg_server_t::which_local_gpu(int loc) const {
  int this_rank = comm.get_this_rank();
  return loc - start_gpus_per_node[this_rank];
}

int gpu_mg_server_t::get_num_gpus() const {
  int ret = 0;
  for(auto const& n: num_gpus_per_node) {
    ret += n;
  }
  return ret;
}

bool gpu_mg_server_t::is_local_gpu(int global_loc) const {
  int my_rank = comm.get_this_rank();
  int my_start = start_gpus_per_node[my_rank];
  int my_end = my_start + num_gpus_per_node[my_rank];
  return my_start <= global_loc && global_loc < my_end;
}

vector<int> gpu_mg_server_t::get_which_storage() const {
  int num_gpus = get_num_gpus();
  vector<int> ret;
  ret.reserve(num_gpus);
  for(int loc = 0; loc != num_gpus; ++loc) {
    ret.push_back(loc_to_compute_node(loc));
  }
  // Note: storage loc == compute node
  return ret;
}

gpu_mg_server_t::~gpu_mg_server_t(){
  // free all gpu memories
  for (auto mem : mems){
    cudaFree(mem);
  }
}

void gpu_mg_server_t::debug_mem(int device, uint64_t counts){
  DOUT("Debugging memory on device: " << device);
  DOUT("Counts: " << counts);
  printFloatGPU(mems[device], counts);
}

