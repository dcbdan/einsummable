#pragma once
#include "../base/setup.h"

#include "memgraph.h"

// allocator_t contains a vector of blocks that either
// have been (1) deleted, or (2) are currently occupied
struct allocator_t {
  allocator_t() = delete;

  allocator_t(
    uint64_t memsize_t,
    allocator_settings_t settings = allocator_settings_t::default_settings());

  // Allocate this much memory if possible and return the offset and all dependents.
  // If there is not free memory of this size, none is returned.
  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate(uint64_t size);

  // tuple<uint64_t, vector<int>>
  // allocate(uint64_t size);

  optional<tuple<uint64_t, set<int>>> allocate(uint64_t sz);

  optional<tuple<vector<uint64_t>, vector<set<int>>>> allocate_multiple(vector<uint64_t> sizes);
  // This function is specifically for allocating without any dependencies.
  // It will try to allocate a block without any deps and on failure returns none.
  optional<uint64_t>
  try_to_allocate_without_deps(uint64_t size);

  void allocate_at_without_deps(uint64_t offset, uint64_t size);

  void set_strategy(allocator_strat_t s) { strat = s; };
  allocator_strat_t get_strategy() const { return strat; }

  // delete this memory, storing the delete dependent
  // for future use of this memory block
  void free(uint64_t offset, int del);

  void print() const;

  // chcek that no memory is being taken up
  bool is_empty() const;

  // If the memory at offset is occupied, return the corresponding
  // occupied interval. Else return None.
  optional<tuple<uint64_t, uint64_t>>
  get_allocated_region(uint64_t offset) const;

  // Remove all dependencies from available memory
  void clear_dependencies();

private:
  struct block_t {
    uint64_t beg;
    uint64_t end;

    // dep is none:
    //   this memory is occupied
    // dep is < 0:
    //   this memory is free and can be used without
    //   adding a dependency
    // dep is >= 0:
    //   this memory is free and can only be used
    //   after dep id has been deleted
    optional<int> dep;

    uint64_t size() const { return end - beg; }
    bool occupied() const  { return !dep.has_value(); }
    bool available() const { return !occupied(); }
    void free(int dep);
  };

  vector<block_t> blocks;
  allocator_strat_t strat;
  uint64_t alignment_power;

  using iter_t = vector<block_t>::iterator;

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_lowest_dependency_available(uint64_t size);

  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate_impl(uint64_t size_without_rem, bool no_deps);

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_first_available(uint64_t size);
};

