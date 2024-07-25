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
  // If there is no free memory of this size, none is returned.
  optional<tuple<uint64_t, set<int>>> 
  allocate(uint64_t sz);

  optional<vector<tuple<uint64_t, set<int>>>>
  allocate_multiple(vector<uint64_t> sizes);

  // This function is specifically for allocating without any dependencies.
  // It will try to allocate a block without any deps and on failure returns none.
  optional<uint64_t>
  allocate_without_deps(uint64_t size);

  // Return whehter or not the allocate was successful
  [[nodiscard]] bool 
  allocate_at_without_deps(uint64_t offset, uint64_t size);

  // Returns a return code and a set of dependencies. If the return code is 
  // >=0, the return code is the block_id of the occupying block and the set
  // should be empty. If the return code is -1, the allocation was successful
  // and the set should contain the nodes that area must depend on.
  tuple<uint64_t, set<int>>
  allocate_at(uint64_t offset, uint64_t size);

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

  // For this offset, get the occupied block id at that offset
  int _get_block_id(uint64_t const& offset);

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

  struct enumerate_t {
    uint64_t size; // without rem
    iter_t first;
    allocator_t* self;
    
    optional<tuple<iter_t, iter_t, uint64_t>> operator()();
  };

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_lowest_dependency_available(uint64_t size);

  optional< tuple<uint64_t, set<int>> >
  allocate_impl(uint64_t size_without_rem, bool no_deps);

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_first_available(uint64_t size);


public:
  ///////////////////////////////////////////////////////////////////
  // A naive implementation to restore the
  // state of an allocator to before a bunch of changes

  // save is a maybe type of either blocks or nothing,
  // but the blocks cannot be changed since it's private
  struct save_t {
    save_t();

    save_t(vector<block_t> const& bs);

    explicit operator bool() const;

    vector<block_t> const& operator()() const;

  private:
    std::unique_ptr<vector<block_t>> blocks;
  };

  save_t checkpoint() const; 

  void reset(save_t const& save) {
    blocks = save(); 
  }

  // return the (best) set of block ids standing in the way from an allocation
  optional<set<int>> _find_best_evict_block_ids(
    // how much we would want to allocate
    uint64_t size,
    // a function from block_id to the score.
    // If score is < 0, the block cannot be used. Otherwise, higher scores are better.
    std::function<int(int)> f_score) const;
    // Example: We have blocks [3,4,5] with score [9,20,8]. Then the score for this range of blocks
    //          is min(9,20,8) = 8.



  double buffer_utilization() const;
  ///////////////////////////////////////////////////////////////////
};

