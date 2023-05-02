#pragma once
#include "setup.h"

struct partdim_t {
  static partdim_t from_spans(vector<uint64_t> const& spans);
  static partdim_t from_sizes(vector<uint64_t> const& sizes);
  static partdim_t repeat(int n_repeat, uint64_t sz);
  static partdim_t singleton(uint64_t sz);
  static partdim_t split(uint64_t total_size, int n_split);
  static partdim_t unions(vector<partdim_t> const& ps);
  static partdim_t split_each(partdim_t const& p, int n_split_each);

  uint64_t total() const { return spans.back(); }

  vector<uint64_t> sizes() const;

  uint64_t size_at(int i) const;

  int num_parts() const { return spans.size(); }

  tuple<uint64_t, uint64_t> which_vals(int blk) const;

  int which_block(uint64_t val) const;

  tuple<int,int> region(uint64_t beg, uint64_t end) const;

  tuple<int,int> exact_region(uint64_t beg, uint64_t end) const;

  vector<uint64_t> spans;
};

bool operator==(partdim_t const& lhs, partdim_t const& rhs);
bool operator!=(partdim_t const& lhs, partdim_t const& rhs);

std::ostream& operator<<(std::ostream& out, partdim_t const& partdim);
