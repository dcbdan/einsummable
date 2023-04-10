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

  // Get the hyper-rectanuglar set represnted by this index
  vector<tuple<uint64_t, uint64_t>>
  get_hrect(vector<int> const& idxs) const
  {
    if(idxs.size() != partdims.size()) {
      throw std::runtime_error("partition_t::get_hrect");
    }

    vector<tuple<uint64_t, uint64_t>> ret;
    ret.reserve(idxs.size());
    for(int i = 0; i != partdims.size(); ++i) {
      ret.push_back(partdims[i].which_vals(idxs[i]));
    }

    return ret;
  }

  // hrect:  hyper-rectangular subset of uint64s
  // region: hyper-rectangular subset of blocks
  vector<tuple<int,int> >
  get_exact_region(
    vector<tuple<uint64_t,uint64_t>> const& region) const
  {
    if(region.size() != partdims.size()) {
      throw std::runtime_error("partition_t::get_region");
    }
    vector<tuple<int,int> > ret;
    ret.reserve(region.size());
    for(int i = 0; i != partdims.size(); ++i) {
      auto const& [beg,end] = region[i];
      ret.push_back(partdims[i].exact_region(beg,end));
    }
    return ret;
  }

  vector<partdim_t> partdims;
};

