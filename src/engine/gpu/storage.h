#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

// for allocator_t
#include "../../einsummable/memgraph.h"
#include "utility.h"

struct gpu_storage_t
{
  gpu_storage_t();

  // inserts a copy of d into the storage
  void write(buffer_t d, int id);

  // inserts id, d into storage; does not copy the
  // d object
  void insert(int id, buffer_t d);

  // gets a copy of the bytes at id
  buffer_t read(int id);

  // same as read but removes the id from storage
  buffer_t load(int id);

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

  map<int, buffer_t> data;
};

