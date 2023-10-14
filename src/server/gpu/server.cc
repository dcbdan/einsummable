#include "server.h"

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

  // TODO: allocate memory on each gpu into mems
}

void gpu_mg_server_t::execute_memgraph(memgraph_t const& memgraph)
{
  // TODO:
  // 1. make the exec graph
  // 2. create the resource manager
  // 3. create the exec state and call the event loop
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
  auto& all_data_locs = ret.data_locs;

  int world_size = comm.get_world_size();
  for(int rank = 1; rank != world_size; ++rank) {
    for(auto const& info: comm.recv_vector<id_memstoloc_t>(rank)) {
      data_locs.insert({info.id, info.as_memstoloc()});
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
  int world_size = comm.get_world_size();

  // TODO: remap storage here with remaps[0]

  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_vector(dst, remaps[dst]);
  }
}

void gpu_mg_server_t::storage_remap_client()
{
  auto remap = comm.recv_vector<std::array<int, 2>>(0);
  // TODO: remap storage here with remap
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
    //
    // TODO: copy from gpu at mems[local_gpu] from [offset, offset+size) into
    //       buffer at this location and return
    return make_buffer(1);
  } else if(d.is_stoloc()) {
    auto const& [sto_loc, sto_id] = d.get_stoloc();
    if(sto_loc != this_rank) {
      throw std::runtime_error("invalid storage location");
    }
    // TODO: copy data from the storage object at sto_id into a new buffer
    return make_buffer(1);
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
      mems[local_gpu]->size,
      alloc_settings);
  }

  for(auto const& [tid, memstoloc]: data_locs) {
    if(memstoloc.is_memloc()) {
      auto const& [offset, size, global_gpu] = memstoloc.get_memloc();
      int local_gpu = which_local_gpu(global_gpu);
      auto& allocator = allocators[local_gpu];
      allocator.allocate_at_without_deps(offset, size);
    }
  }

  for(auto const& [tid, loc_tensor]: data) {
    auto const& [global_gpu, tensor] = loc_tensor;
    int local_gpu = which_local_gpu(global_gpu);
    auto& allocator = allocators[local_gpu];
    auto maybe_offset = allocator.try_to_allocate_without_deps(tensor->size);

    memstoloc_t memstoloc;
    if(maybe_offset) {
      auto const& offset = maybe_offset.value();

      // TODO: copy tensor onto gpu memory at mems[local_gpu] + offset

      memloc_t memloc {
        .offset = offset,
        .size = tensor->size,
        .loc = global_gpu
      };

      memstoloc = memstoloc_t(memloc);
    } else {
      // TODO: create an id and insert copy tensor into the storage object
      int id = 99999;

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
      // TODO: tell storage to remove id
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

