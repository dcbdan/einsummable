#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

#include "utility.h"

struct host_buffer_holder_t {
  host_buffer_holder_t(uint64_t size);
  ~host_buffer_holder_t();

  void*       raw()       { return data; }
  void const* raw() const { return data; }

  uint8_t*       as_uint8()       { return reinterpret_cast<uint8_t*>(      data); }
  uint8_t const* as_uint8() const { return reinterpret_cast<uint8_t const*>(data); }

  uint64_t size;
  void* data;
};

using host_buffer_t = std::shared_ptr<host_buffer_holder_t>;

host_buffer_t make_host_buffer(uint64_t size);

struct gpu_storage_t
{
  gpu_storage_t();

  // inserts a copy of d into the storage
  void write(host_buffer_t d, int id);
  void write(buffer_t d, int id);

  // inserts id, d into storage; does not copy the
  // d object
  void insert(int id, host_buffer_t d);

  // gets a copy of the bytes at id
  host_buffer_t read(int id);
  buffer_t read_to_buffer(int id);

  // same as read but removes the id from storage
  host_buffer_t load(int id);

  // removes the id
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
  std::mutex m;

  map<int, host_buffer_t> data;
};

