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

  save_t save = checkpoint();
  vector<tuple<uint64_t, set<int>>> ret;
  for(uint64_t const& size: sizes) {
    auto maybe = allocate(size);
    if(maybe) {
      ret.emplace_back(maybe.value());
    } else {
      break;
    }
  }

  if(ret.size() != sizes.size()) {
    reset(save);
    return std::nullopt;
  } else {
    return ret;
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
allocator_t::enumerate_t::operator()()
{
  optional<tuple<iter_t, iter_t, uint64_t>> ret;

  for(; first != self->blocks.end() && !bool(ret);) {
    if(first->available()) {
      uint64_t rem = align_to_power_of_two(first->beg, self->alignment_power) - first->beg;

      iter_t last = first;

      uint64_t sz = 0;
      uint64_t size_with_rem = rem + size;

      bool success = false;
      for(; last != self->blocks.end() && last->available(); ++last) {
        sz += last->size();
        if(sz >= size_with_rem) {
          success = true;
          ret = {first, last+1, sz};
          break;
        }
      }

      if(success) {
        first++;
      } else {
        first = last;
      }
    } else {
      first++;
    }
  }

  return ret;
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_first_available(uint64_t size)
{
  enumerate_t enumerate { .size = size, .first = blocks.begin(), .self = this };
  return enumerate();
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_lowest_dependency_available(uint64_t size)
{
  enumerate_t enumerate { .size = size, .first = blocks.begin(), .self = this };
  optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>> ret = std::nullopt;
  int min_dep = -2;
  while(true) {
    auto maybe = enumerate();
    if(!maybe) {
      // we have tried em all, so return the best one if any was found
      return ret;
    } else {
      int dep = -1;      
      auto const& [beg,end,_] = maybe.value();
      for(iter_t iter = beg; iter != end; ++iter) {
        dep = std::max(dep, iter->dep.value());
      }
      if(!bool(ret) || dep < min_dep) {
        ret = maybe;
        min_dep = dep;
      }
    }
  }
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
  } else {
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

allocator_t::save_t allocator_t::checkpoint() const {
  return save_t(blocks);
}

allocator_t::save_t::save_t(): blocks(nullptr) {}

allocator_t::save_t::save_t(vector<block_t> const& bs)
  : blocks(std::make_unique<vector<block_t>>(bs))
{}

allocator_t::save_t::operator bool() const {
  return bool(blocks);
}

vector<allocator_t::block_t> const& allocator_t::save_t::operator()() const {
  if(!blocks) {
    throw std::runtime_error("invalid save_t state");
  }
  return *blocks;
}

optional<set<int>> allocator_t::_find_best_evict_block_ids(
  uint64_t size,
  std::function<int(int)> f_block_score) const
{
  optional<set<int>> ret;
  optional<int> best_score;
  for(auto first = blocks.begin(); first != blocks.end(); ++first) {
    uint64_t rem = align_to_power_of_two(first->beg, alignment_power) - first->beg;
    uint64_t size_with_rem = rem + size;

    uint64_t sz = 0;
    bool success = false;
    auto last = first;
    for(; last != blocks.end(); ++last) {
      sz += last->size();
      if(sz >= size_with_rem) {
        success = true;
        break;
      }
    }

    if(success) {
      // At this point, the interval we are evaluating is [first,last]

      // make sure that each occupied block is valid and get the score.
      int score = std::numeric_limits<int>::max();
      set<int> ret_blocks;
      for(auto iter = first; iter != last + 1; ++iter) {
        int blk_id = std::distance(blocks.begin(), iter);
        if(iter->occupied()) {
          int s = f_block_score(blk_id);
          if(s < 0) {
            // ok, we found a block we can't evict, so fail
            success = false;
            break;
          } else {
            score = std::min(score, s);
            ret_blocks.insert(blk_id);
          }
        }
      }
      if(success && (!best_score || best_score.value() < score)) {
        best_score = score;
        ret = ret_blocks;
      }
    }
  }

  return ret;
}

//throws exception if the block is not allocated at offset.
//Note: we are assuming an unchanged order on blocks, so between calls of this function,
//      we cannot merge or do anything to the blocks.
int allocator_t::_get_block_id(uint64_t const& offset)
{
  // for (auto block: blocks) {
  //   if (offset <= block.beg) {
  //     std::cout << "is the block occupied: " << block.occupied() << std::endl;
  //     break;
  //   }
  // }
  // this->print();
  //find the last block in allocator that has block.beg >= (input) offset
  auto ret = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk) {
      return offset >= blk.beg;
    });
  if(!ret->occupied()) {
    throw std::runtime_error("block not occupied at the offset provided!");
  }
  return std::distance(blocks.begin(), ret);
}

double allocator_t::buffer_utilization() const {
    uint64_t total_occupied = 0;
    uint64_t total_memory = blocks.empty() ? 0 : blocks.back().end; // Assuming blocks cover all memory.

    for (const auto& block : blocks) {
        if (block.occupied()) {
            total_occupied += block.size();
        }
    }

    double utilization = total_memory > 0 ? (100.0 * total_occupied / total_memory) : 0.0;
    std::cout << "total memory: " << total_memory << "block count: " << blocks.size() << "utilization: " << utilization <<std::endl;
    return utilization;
}


