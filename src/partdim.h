#pragma once
#include "setup.h"

struct partdim_t {
  static partdim_t from_sizes(vector<uint64_t> sizes) {
    vector<uint64_t> spans = sizes;
    uint64_t total = spans[0];
    for(int i = 1; i < spans.size(); ++i) {
      spans[i] += total;
      total = spans[i];
    }
    return partdim_t { .spans = spans };
  }
  static partdim_t repeat(int n_repeat, uint64_t sz) {
    return from_sizes(vector<uint64_t>(n_repeat, sz));
  }
  static partdim_t singleton(uint64_t sz) {
    return partdim_t { .spans = {sz} };
  }
  static partdim_t split(uint64_t total_size, int n_split) {
    return from_sizes(divide_evenly(n_split, total_size));
  }
  static partdim_t unions(vector<partdim_t> ps) {
    if(ps.size() == 0) {
      throw std::runtime_error("partdim_t::unions: must have nonempty input");
    }

    uint64_t d = ps[0].total();
    for(auto const& p: ps) {
      if(d != p.total()) {
        throw std::runtime_error("partdim_t::unions: all inputs must have the same size");
      }
    }

    // merge sort and remove the duplicates
    vector<vector<uint64_t>> _ps;
    ps.reserve(ps.size());
    for(auto const& p: ps) {
      _ps.push_back(p.spans);
    }
    vector<uint64_t> spans = vector_sorted_merges(_ps);
    vector_remove_duplicates(spans);

    partdim_t ret { .spans = spans };

    if(d != ret.total()) {
      throw std::runtime_error("pardim_t::unions implementation error");
    }

    return ret;
  }

  uint64_t total() const { return spans.back(); }

  vector<uint64_t> sizes() const {
    vector<uint64_t> ret = spans;
    for(int i = ret.size(); i > 0; --i) {
      ret[i] -= ret[i-1];
    }
    // spans = [10,20,30,35]
    // ret   = [10,10,10, 5]
    return ret;
  }

  uint64_t size_at(int i) const {
    if(i == 0) {
      return spans[0];
    } else if(i < spans.size()) {
      return spans[i] - spans[i-1];
    } else {
      throw std::runtime_error("partdim_t::size_at input error");
    }
  }

  int num_parts() const {
    return spans.size();
  }

  tuple<uint64_t, uint64_t> which_vals(int blk) const {
    if(blk == 0) {
      return {0, spans[0]};
    }
    return {spans[blk-1], spans[blk]};
  }

  int which_block(uint64_t val) const {
    // Example:
    //   spans =   {10,20,30,40,50}
    //   val in [0, 10) -> 0
    //          [10,20) -> 1
    //          [20,30) -> 2
    //          [30,40) -> 3
    //          [40,50) -> 4
    //          [50,inf) -> error

    // TODO: spans is sorted; a binary search should be faster
    for(int i = 0; i != spans.size(); ++i) {
      if(val < spans[i]) {
        return i;
      }
    }
    throw std::runtime_error("which_block: should not reach");
  }

  tuple<int,int> region(uint64_t beg, uint64_t end) const {
    // Example:
    //   spans =   {10,20,30,40,50}
    //   0,50  -> 0,5
    //   0,41  -> 0,5
    //   9,40  -> 0,4
    //   25,35 -> 2,4
    //   25,30 -> 2,3
    if(end <= beg) {
      throw std::runtime_error("region: end >= beg");
    }
    return {which_block(beg), which_block(end-1) + 1};
  }

  tuple<int,int> exact_region(uint64_t beg, uint64_t end) const {
    // Return the integer indexing spanning beg,end exactly
    // (that is, it is an error if beg and end are not 0 or in spans)
    //
    // Example:
    //   spans =   {10,20,30,40,50}
    //   0,50 -> 0,5
    //   0,51 -> error
    //   0,49 -> error
    //   1,50 -> error
    auto [bb,ee] = region(beg,end);
    if(
      (beg == 0 || spans[bb-1] == beg) &&
      spans[ee-1] == end)
    {
      return {bb,ee};
    } else {
      throw std::runtime_error("exact_region: not exact");
      return {0,0};
    }
  }

  vector<uint64_t> spans;
};

bool operator==(partdim_t const& lhs, partdim_t const& rhs);
bool operator!=(partdim_t const& lhs, partdim_t const& rhs);

std::ostream& operator<<(std::ostream& out, partdim_t const& partdim);
