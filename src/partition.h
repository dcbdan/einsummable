#pragma once
#include "setup.h"

#include "partdim.h"

struct partition_t {
  partition_t(vector<partdim_t> const& p):
    partdims(p)
  {}

  static partition_t singleton(vector<uint64_t> shape) {
    vector<partdim_t> partdims(shape.size());
    for(auto const& sz: shape) {
      partdims.push_back(partdim_t::singleton(sz));
    }
    return partition_t(partdims);
  };

  vector<uint64_t> total_shape() const {
    vector<uint64_t> ret(partdims.size());
    for(auto const& pdim: partdims) {
      ret.push_back(pdim.total());
    }
    return ret;
  }

  int num_parts() const {
    return product(this->block_shape());
  }

  vector<int> block_shape() const {
    vector<int> ret;
    ret.reserve(partdims.size());
    for(auto const& p: partdims) {
      ret.push_back(p.num_parts());
    }
    return ret;
  }

  vector<partdim_t> partdims;
};

