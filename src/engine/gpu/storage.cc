#include "storage.h"
#include <stdlib.h>

gpu_storage_t::gpu_storage_t() {}

void gpu_storage_t::write(buffer_t data, int id) {
  buffer_t ret = make_buffer(data->size);
  std::copy(data->data, data->data + data->size, ret->data);
  insert(id, ret);
}

void gpu_storage_t::insert(int id, buffer_t data) {
  std::unique_lock lk(m);

  auto [_, did_insert] = this->data.insert({id, data});
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

  map<int, buffer_t> new_data;

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

buffer_t gpu_storage_t::load(int id) {
  std::unique_lock lk(m);
  auto iter = data.find(id);
  if(iter == data.end()) {
    throw std::runtime_error("id not in storage.");
  }
  buffer_t data = iter->second;
  this->data.erase(iter);
  return data;
}

buffer_t gpu_storage_t::read(int id) {
  std::unique_lock lk(m);
  auto iter = data.find(id);
  if(iter == data.end()) {
    throw std::runtime_error("id not in storage.");
  }
  buffer_t data = iter->second;
  buffer_t ret = make_buffer(data->size);
  std::copy(data->data, data->data + data->size, ret->data);
  return ret;
}

