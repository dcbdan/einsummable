#include "storage.h"
#include <stdlib.h>

gpu_storage_t::stoalloc_t::stoalloc_t()
  : allocator(
      std::numeric_limits<uint64_t>::max(),
      allocator_settings_t {
        .strat = allocator_strat_t::first,
        .alignment_power = 0
      })
{}

uint64_t gpu_storage_t::stoalloc_t::allocate(uint64_t sz) {
  return std::get<0>(allocator.allocate(sz));
}

void gpu_storage_t::stoalloc_t::free(uint64_t offset) {
  allocator.free(offset, 0);
}

uint64_t gpu_storage_t::stoalloc_t::get_size_at(uint64_t offset) const {
  auto maybe = allocator.get_allocated_region(offset);
  if(!maybe) {
    throw std::runtime_error("get_size_at: this memory not allocated");
  }
  auto const& [beg,end] = maybe.value();
  if(beg != offset) {
    throw std::runtime_error("get_size_at: offset provided in middle of allocated region");
  }
  return end-beg;
}

void gpu_storage_t::write(void* gpu_offseted_ptr, int size, int id, cuda_stream_t stream){
  std::unique_lock lk(m);

  auto iter = id_to_cpu_offset.find(id);
  if(iter != id_to_cpu_offset.end()) {
    throw std::runtime_error("id already in storage; try removing first");
  }

  uint64_t offset = allocator.allocate(size);
  id_to_cpu_offset.insert({id, offset});

  // TODO: GET THE CORRECT CPU_PTR WITH THE CORRECT OFFSET
  // HOW TO ACCESS CPU BASE PTR?
  void* cpu_ptr;

  cudaMemcpyAsync(cpu_ptr, gpu_offseted_ptr, 
    size, cudaMemcpyDeviceToHost, stream.stream);

}

void gpu_storage_t::read(void* gpu_offseted_ptr, int id, cuda_stream_t stream) {
  std::unique_lock lk(m);

  auto iter = id_to_cpu_offset.find(id);
  if(iter == id_to_cpu_offset.end()) {
    throw std::runtime_error("read: id not in storage");
  }
  uint64_t const& offset = iter->second;
  void* cpu_ptr;
  // IS THE ALLOCATOR ACCESSING THE DISK FILE OR CPU RAM?
  uint64_t sz = allocator.get_size_at(offset);
  // TODO: GET THE CORRECT CPU_PTR WITH THE CORRECT OFFSET
  // HOW TO ACCESS CPU BASE PTR?
  cudaMemcpyAsync(gpu_offseted_ptr, cpu_ptr, 
    sz, cudaMemcpyHostToDevice, stream.stream);
}
// // ---- if use a map of (int id, buffer_t buffer_on_cpu) ---

// void gpu_storage_t::write(void* gpu_offseted_ptr, int size, int id, cuda_stream_t stream){
//   std::unique_lock lk(m);

//   auto iter = id_to_cpu_offset.find(id);
//   if(iter != id_to_cpu_offset.end()) {
//     throw std::runtime_error("id already in storage; try removing first");
//   }

//   buffer_t cpu_buffer = make_buffer(size);
//   id_to_cpu_offset.insert({id, cpu_buffer});

//   cudaMemcpyAsync(cpu_buffer->raw(), gpu_offseted_ptr, 
//     size, cudaMemcpyDeviceToHost, stream.stream);
// }

// void gpu_storage_t::read(void* gpu_offseted_ptr, int id, cuda_stream_t stream) {
//   std::unique_lock lk(m);
//   auto iter = id_to_cpu_offset.find(id);
//   if(iter == id_to_cpu_offset.end()) {
//     throw std::runtime_error("read: id not in storage");
//   }

//   buffer_t cpu_buffer = iter->second;
//   cudaMemcpyAsync(gpu_offseted_ptr, cpu_buffer->raw(), 
//     cpu_buffer->size, cudaMemcpyHostToDevice, stream.stream);
// }

void gpu_storage_t::remove(int id) {
  std::unique_lock lk(m);
  _remove(id);
}

void gpu_storage_t::_remove(int id) {
  auto iter = id_to_cpu_offset.find(id);
  if(iter == id_to_cpu_offset.end()) {
    throw std::runtime_error("storage remove: id not in storage");
  }
  uint64_t const& offset = iter->second;
  allocator.free(offset);
  id_to_cpu_offset.erase(iter);
}


void gpu_storage_t::remap(vector<std::array<int, 2>> const& old_to_new_stoids) {
  map<int, int> items;
  for(auto const& old_new: old_to_new_stoids) {
    items.insert({old_new[0], old_new[1]});
  }
  remap(items);
}

void gpu_storage_t::remap(map<int, int> const& mm) {
  std::unique_lock lk(m);

  map<int, uint64_t> new_offsets;
  set<int> will_remove;

  for(auto const& [id, cpu_offset]: id_to_cpu_offset) {
    auto iter = mm.find(id);
    if(iter == mm.end()) {
      will_remove.insert(id);
    } else {
      new_offsets.insert({iter->second, cpu_offset});
    }
  }

  for(int const& id: will_remove) {
    _remove(id);
  }

  id_to_cpu_offset = new_offsets;
}

int gpu_storage_t::get_max_id() const {
  if(id_to_cpu_offset.size() == 0) {
    return -1;
  }
  return id_to_cpu_offset.rbegin()->first;
}



