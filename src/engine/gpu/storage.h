#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

// for allocator_t
#include "../../einsummable/memgraph.h"
#include "utility.h"

#include <mutex>
#include <fstream>

struct gpu_storage_t
{
  gpu_storage_t();

	void write(void* gpu_offseted_ptr, int size, int id, cuda_stream_t stream);

	void read(void* gpu_offseted_ptr, int id, cuda_stream_t stream);

	void remove(int id);

  // for each [old,new] in old_to_new_stoids,
  //   convert the storage id at old to new
  // If an old isn't a storage id here, throw an error
  // If a storage id isn't an old, delete it
  void remap(vector<std::array<int, 2>> const& old_to_new_stoids);
  void remap(map<int, int> const& old_to_new_stoids);
  // before: 3,5,60,9,10
  // remap: {3,8}, {5,60}
  // after: 8,60
  // (the before 60,9,10 would get removed)

  int get_max_id() const;

private:
  // allocator_t from the memgraph does everything
  // we need, except that it also has this dependency mechanism.
  // This wrapper gets rid of that and makes the allocator_t
  // settings appropriate for this uue case.
  struct stoalloc_t {
    stoalloc_t();
    uint64_t allocate(uint64_t sz);
    void free(uint64_t offset);

    uint64_t get_size_at(uint64_t offset) const;

  private:
    allocator_t allocator;
  };
  
  void _remove(int id);
private:
  std::mutex m;

  map<int, uint64_t> id_to_cpu_offset;

  stoalloc_t allocator;
};

