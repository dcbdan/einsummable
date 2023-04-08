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
  static partdim_t singleton(uint64_t shape) {
    return partdim_t { .spans = {shape} };
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

  int num_parts() const {
    return spans.size();
  }

  vector<uint64_t> spans;
};


