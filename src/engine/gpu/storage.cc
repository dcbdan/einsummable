#include "storage.h"
#include <stdlib.h>

host_buffer_holder_t::host_buffer_holder_t(uint64_t size)
  : size(size)
{
  handle_cuda_error(
    cudaMallocHost(&data, size),
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

gpu_storage_t::gpu_storage_t() {}

void gpu_storage_t::write(host_buffer_t buffer, int id) {
  host_buffer_t ret = make_host_buffer(buffer->size);
  std::copy(buffer->as_uint8(), buffer->as_uint8() + buffer->size, ret->as_uint8());
  insert(id, ret);
}
void gpu_storage_t::write(buffer_t buffer, int id) {
  host_buffer_t ret = make_host_buffer(buffer->size);
  std::copy(buffer->data, buffer->data + buffer->size, ret->as_uint8());
  insert(id, ret);
}

void gpu_storage_t::insert(int id, host_buffer_t buffer) {
  std::unique_lock lk(m);

  auto [_, did_insert] = this->data.insert({id, buffer});
  if(!did_insert) {
    throw std::runtime_error("id already in gpu storage");
  }
}

void gpu_storage_t::remove(int id) {
  load(id);
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

  map<int, host_buffer_t> new_data;

  for(auto const& [old_id,buffer]: data) {
    int const& new_id = rmap.at(old_id);
    new_data.insert({new_id, buffer});
  }

  data = new_data;
}

int gpu_storage_t::get_max_id() const {
  if(data.size() == 0) {
    return -1;
  }
  return data.rbegin()->first;
}

host_buffer_t gpu_storage_t::load(int id) {
  std::unique_lock lk(m);
  auto iter = data.find(id);
  if(iter == data.end()) {
    throw std::runtime_error("id not in storage.");
  }
  host_buffer_t data = iter->second;
  this->data.erase(iter);
  return data;
}

host_buffer_t gpu_storage_t::read(int id) {
  std::unique_lock lk(m);
  auto iter = data.find(id);
  if(iter == data.end()) {
    throw std::runtime_error("id not in storage.");
  }
  host_buffer_t d = iter->second;
  host_buffer_t ret = make_host_buffer(d->size);
  std::copy(d->as_uint8(), d->as_uint8() + d->size, ret->as_uint8());
  return ret;
}

buffer_t gpu_storage_t::read_to_buffer(int id) {
  std::unique_lock lk(m);
  auto iter = data.find(id);
  if(iter == data.end()) {
    throw std::runtime_error("id not in storage.");
  }
  host_buffer_t d = iter->second;
  buffer_t ret = make_buffer(d->size);
  std::copy(d->as_uint8(), d->as_uint8() + d->size, ret->data);
  return ret;
}

