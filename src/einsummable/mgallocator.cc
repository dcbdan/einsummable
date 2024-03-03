#include "mgallocator.h"

allocator_t::allocator_t(uint64_t memsize, allocator_settings_t s)
  : strat(s.strat), alignment_power(s.alignment_power) 
{
  if(memsize == 0) {
    throw std::runtime_error("invalid memsize for allocator");
  }
  blocks.push_back(block_t{
    .beg = 0,
    .end = memsize,
    .dep = -1
  });
}

optional<vector<tuple<uint64_t, set<int>>>>
allocator_t::allocate_multiple(vector<uint64_t> sizes)
{
  // Note, it may be best to allocate the largest items first,
  // but we need to return the deps in order so we aren't bothering

  // Record the status of the blocks before anything
  // and on failure, just revert.
  vector<block_t> original_blocks = blocks;

  bool fail = false;
  vector<tuple<uint64_t, set<int>>> alloced_info;
  for(uint64_t const& size: sizes) {
    auto const& maybe_info = allocate(size);
    if(maybe_info) {
      auto const& [offset, deps_] = maybe_info.value();

      set<int> deps;
      for(int const& d: deps_) {
        if(d != -1) {
          deps.insert(d);
        }
      }

      alloced_info.emplace_back(offset, deps);
    } else {
      fail = true;
      break;
    }
  }

  if(fail) {
    blocks = original_blocks;
    return std::nullopt;
  } else {
    return alloced_info;
  }
}

optional<uint64_t>
allocator_t::allocate_without_deps(uint64_t size)
{
  auto const& maybe = allocate_impl(size, true);
  if(maybe)
  {
    auto const& [offset, d] = maybe.value();
    return offset;
  }
  else
  {
    return std::nullopt;
  }
}

void allocator_t::block_t::free(int d)
{
  if(!occupied())
  {
    throw std::runtime_error("cannot free unoccupied memory block");
  }
  dep = d;
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_first_available(uint64_t size)
{
  using return_t = tuple<iter_t, iter_t, uint64_t>;

  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter)
  {
    if(iter->available())
    {
      iter_t ret = iter;
      uint64_t sz = 0;
      uint64_t rem = align_to_power_of_two(iter->beg, alignment_power) - iter->beg;
      for(; iter != blocks.end() && iter->available(); ++iter)
      {
        sz += iter->size();
        if(rem != 0 && sz > rem)
        {
          rem = 0;
          sz -= rem;
        }
        if(rem == 0 && sz >= size)
        {
          return optional<return_t>({ret, iter + 1, sz});
        }
      }
    }
  }

  return std::nullopt;
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_lowest_dependency_available(uint64_t size)
{
  using return_t = tuple<iter_t, iter_t, uint64_t>;
  optional<return_t> return_block;
  int min_dep = std::numeric_limits<int>::max();
  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter)
  {
    if(iter->available())
    {
      iter_t ret = iter;
      uint64_t sz = 0;
      uint64_t rem = align_to_power_of_two(iter->beg, alignment_power) - iter->beg;
      int inner_max_dep = -1;
      for(iter_t inner_iter = iter;
           inner_iter != blocks.end() && inner_iter->available();
           ++inner_iter)
      {
        inner_max_dep = std::max(inner_max_dep, inner_iter->dep.value());
        if(inner_max_dep >= min_dep)
        {
          break;
        }
        sz += inner_iter->size();
        if(rem != 0 && sz > rem)
        {
          rem = 0;
          sz -= rem;
        }
        if(rem == 0 && sz >= size)
        {
          min_dep = inner_max_dep;
          return_block = {ret, inner_iter + 1, sz};
          break;
        }
      }
    }
  }
  return return_block;
}

optional<tuple<uint64_t, set<int>>>
allocator_t::allocate_impl(uint64_t size_without_rem, bool no_deps)
{
  using value_t = tuple<uint64_t, set<int>>;

  optional<tuple<iter_t, iter_t, uint64_t>> maybe_info;
  if(no_deps) {
    maybe_info = find_lowest_dependency_available(size_without_rem);
  } else {
    if(strat == allocator_strat_t::lowest_dependency)
    {
      maybe_info = find_lowest_dependency_available(size_without_rem);
    }
    else if(strat == allocator_strat_t::first)
    {
      maybe_info = find_first_available(size_without_rem);
    }
    else
    {
      throw std::runtime_error("should not reach");
    }
  }

  if(maybe_info)
  {
    auto const& [beg, end, sz] = maybe_info.value();

    // collect the output information
    uint64_t offset = beg->beg;
    uint64_t aligned_offset = align_to_power_of_two(beg->beg, alignment_power);

    uint64_t size = size_without_rem + (aligned_offset - offset);

    set<int> deps;
    for(auto iter = beg; iter != end; ++iter)
    {
      if(!iter->dep)
      {
        throw std::runtime_error("invalid find_available return");
      }
      int const& d = iter->dep.value();
      if(d >= 0)
      {
        deps.insert(d);
      }
    }

    if(no_deps && deps.size() > 0)
    {
      // if deps aren't allowed and there would be some, then fail here
      return std::nullopt;
    }

    // fix blocks
    block_t last_block_copy = *(end - 1);

    // // init the blocks that are in the list before, so that we know how to revert it
    // vector<block_t> before_blocks(beg, end);

    auto iter = blocks.erase(beg, end);
    auto occupied_iter = blocks.insert(iter, block_t {
      .beg = offset,
      .end = offset + size,
      .dep = optional<int>()
    });
    if(size != sz)
    {
      blocks.insert(occupied_iter + 1, block_t {
        .beg = offset + size,
        .end = last_block_copy.end,
        .dep = last_block_copy.dep
      });
    }
    return value_t{aligned_offset, deps};
  }
  else
  {
    return std::nullopt;
  }
}

optional<tuple<uint64_t, set<int>>>
allocator_t::allocate(uint64_t size_without_rem)
{
  return allocate_impl(size_without_rem, false);
}

bool allocator_t::allocate_at_without_deps(uint64_t offset, uint64_t size)
{
  auto beg = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk)
    {
      return blk.beg <= offset;
    }
  );
  // beg points to the last block that has the beg <= offset.

  auto last = binary_search_find(beg, blocks.end(),
    [&offset, &size](block_t const& blk)
    {
      return blk.beg < offset + size;
    }
  );

  // last points to the last block with an element in [offset,offset+size).
  if(last == blocks.end())
  {
    return false;
  }

  auto end = last + 1;

  for(auto iter = beg; iter != end; ++iter)
  {
    auto const& blk = *iter;
    if(blk.dep && blk.dep.value() < 0)
    {
      // this memory is available and has no dependencies
    } else {
      return false;
    }
  }

  uint64_t unused_begin = beg->beg;
  uint64_t unused_begin_size = offset - unused_begin;

  uint64_t unused_end = offset + size;
  uint64_t unused_end_size = last->end - unused_end;

  auto iter = blocks.erase(beg, end);

  // iter = the spot to insert before,
  // so insert the unused end segment,
  // the used segment and then the unused
  // begin segment.

  if(unused_end_size != 0)
  {
    iter = blocks.insert(iter, block_t {
      .beg = unused_end,
      .end = unused_end + unused_end_size,
      .dep = optional<int>(-1)
    });
  }

  iter = blocks.insert(iter, block_t{
    .beg = offset,
    .end = offset + size,
    .dep = std::nullopt
  });

  if(unused_begin_size != 0)
  {
    iter = blocks.insert(iter, block_t {
      .beg = unused_begin,
      .end = unused_begin + unused_begin_size,
      .dep = optional<int>(-1)}
    );
  }

  return true;
}

void allocator_t::free(uint64_t offset, int del)
{
  auto iter = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk)
    {
      return blk.beg <= offset;
    }
  );

  if(iter == blocks.end())
  {
    throw std::runtime_error("did not find a block");
  }

  block_t &block = *iter;
  block.free(del);
}

void allocator_t::print() const
{
  auto &out = std::cout;

  for(auto const&blk: blocks)
  {
    out << "[" << blk.beg << "," << blk.end << ")@";
    if(blk.dep)
    {
      int const& d = blk.dep.value();
      if(d < 0) {
        out << "neverassigned";
      } else {
        out << "free;dep" << blk.dep.value();
      }
    } else {
      out << "occupied";
    }
    out << std::endl;
  }
}

bool allocator_t::is_empty() const
{
  for(auto const& blk: blocks) {
    if(blk.occupied()) {
      return false;
    }
  }
  return true;
}

optional<tuple<uint64_t, uint64_t>>
allocator_t::get_allocated_region(uint64_t offset) const
{
  auto iter = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk)
    {
      return blk.beg <= offset;
    }
  );

  if(iter == blocks.end()) {
    return std::nullopt;
  }

  block_t const& block = *iter;
  if(!block.occupied()) {
    return std::nullopt;
  }

  return optional<tuple<uint64_t, uint64_t>>({block.beg, block.end});
}

void allocator_t::clear_dependencies()
{
  vector<block_t> new_blocks;
  optional<tuple<uint64_t, uint64_t>> next_interval;
  for(auto const& block: blocks)
  {
    if(block.dep) {
      if(next_interval) {
        auto &[_, end] = next_interval.value();
        end = block.end;
      } else {
        next_interval = tuple<uint64_t, uint64_t>{block.beg, block.end};
      }
    } else {
      if(next_interval) {
        auto const& [beg, end] = next_interval.value();
        new_blocks.push_back(block_t{
            .beg = beg,
            .end = end,
            .dep = -1});
        next_interval = std::nullopt;
      }
      new_blocks.push_back(block);
    }
  }
  if(next_interval) {
    auto const& [beg, end] = next_interval.value();
    new_blocks.push_back(block_t {
      .beg = beg,
      .end = end,
      .dep = -1
    });
  }

  blocks = new_blocks;
}

