#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

// TODO: this is just a placeholder implementation
struct storage_t
{
	void write(buffer_t buffer, int id) {}
	void read(buffer_t& buffer, int id) {}

  buffer_t read(int id) { return buffer_t(); }

  // same as read but removes the id from the
  // storage
  buffer_t load(int id) {
    buffer_t ret = read(id);
    remove(id);
    return ret;
  }

  void load(buffer_t& ret, int id) {
    read(ret, id);
    remove(id);
  }

	void remove(int id) {}

  // for each [old,new] in old_to_new_stoids,
  //   convert the storage id at old to new
  // If an old isn't a storage id here, throw an error
  // If a storage id isn't an old, delete it
  void remap(vector<std::array<int, 2>> const& old_to_new_stoids);
  // before: 3,5,60,9,10
  // remap: {3,8}, {5,60}
  // after: 8,60
  // (the before 60,9,10 would get removed)

  storage_t() {}
};

