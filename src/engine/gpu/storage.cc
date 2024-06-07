#include "storage.h"
#include <stdlib.h>

static uint64_t storage_size = 4lu*1024lu*1024lu*1024lu;

host_buffer_holder_t::host_buffer_holder_t(uint64_t size)
  : size(size)
{
  // cudaHostAllocPortable allows for the buffer to be used on any device
  // cudaHostAllocWriteCombined allows for faster writes to the buffer
  handle_cuda_error(
    cudaHostAlloc(&data, size, 
      cudaHostAllocPortable|cudaHostAllocWriteCombined),
    "host buffer construction");
}

host_buffer_holder_t::~host_buffer_holder_t()
{
  handle_cuda_error(
    cudaFreeHost(data),
    "host buffer destructor");
}

host_buffer_t make_host_buffer(uint64_t size) {
  return std::make_shared<host_buffer_holder_t>(size);
}

gpu_storage_t::gpu_storage_t()
  : allocator(
      storage_size,
      allocator_settings_t {
        .strat = allocator_strat_t::first,
        .alignment_power = 0
      })
{
  DOUT("CudaMallocHost for host_data; size in GB: " << storage_size/(1024*1024*1024));
  host_data = make_host_buffer(storage_size);
  DOUT("CudaMallocHost for host_data done");
}

buffer_t gpu_storage_t::alloc(uint64_t size, int id) {
  std::unique_lock lk(m);

  auto maybe = allocator.allocate_without_deps(size);
  if(!maybe) {
    throw std::runtime_error("ran out of gpu_storage blob space");
  }
  uint64_t const& offset = maybe.value();

  mem_t mem { .offset = offset, .size = size };
  auto [_, did_insert] = this->info.insert({id, mem});
  if(!did_insert) {
    throw std::runtime_error("id already in gpu storage");
  }

  return make_reference(offset, size);
}

void gpu_storage_t::write(buffer_t d, int id) {
  buffer_t ret = alloc(d->size, id);
  std::copy(d->data, d->data + d->size, ret->data);
}

buffer_t gpu_storage_t::_read(int id) {
  auto iter = info.find(id);
  if(iter == info.end()) {
    throw std::runtime_error("id not in storage.");
  }

  buffer_t d = make_reference(iter->second);
  buffer_t ret = make_buffer(d->size);
  std::copy(d->data, d->data + d->size, ret->data);

  return ret;
}

buffer_t gpu_storage_t::read(int id) {
  std::unique_lock lk(m);
  return _read(id);
}

void gpu_storage_t::_remove(int id) {
  auto iter = info.find(id);
  if(iter == info.end()) {
    throw std::runtime_error("id not in storage.");
  }

  {
    uint64_t const& offset = iter->second.offset;
    allocator.free(offset, -1);
  }

  info.erase(iter);
}

void gpu_storage_t::remove(int id) {
  std::unique_lock lk(m);
  return _remove(id);
}

buffer_t gpu_storage_t::load(int id) {
  std::unique_lock lk(m);
  buffer_t ret = _read(id);
  _remove(id);
  return ret;
}

void gpu_storage_t::remap(vector<std::array<int, 2>> const& old_to_new_stoids) {
  map<int, int> items;
  for(auto const& old_new: old_to_new_stoids) {
    items.insert({old_new[0], old_new[1]});
  }
  remap(items);
}

void gpu_storage_t::remap(map<int, int> const& rmap) {
  std::unique_lock lk(m);

  map<int, mem_t> new_info;

  for(auto const& [old_id,m]: info) {
    int const& new_id = rmap.at(old_id);
    new_info.insert({new_id, m});
  }

  info = new_info;
}

int gpu_storage_t::get_max_id() const {
  if(info.size() == 0) {
    return -1;
  }
  return info.rbegin()->first;
}

buffer_t gpu_storage_t::make_reference(uint64_t offset, uint64_t size)
{
  return make_buffer_reference(host_data->as_uint8() + offset, size);
}

buffer_t gpu_storage_t::make_reference(mem_t const& mem) 
{
  return make_reference(mem.offset, mem.size);
}

buffer_t gpu_storage_t::reference(int id) 
{
  std::unique_lock lk(m);
  auto iter = info.find(id);
  if(iter == info.end()) {
    throw std::runtime_error("id not in storage.");
  }

  return make_reference(iter->second);
}

