#pragma once
#include "setup.h"

#include "tensor.h"
#include "partition.h"

struct placement_t {
  placement_t(partition_t const& p, tensor_t<int> const& locs):
    partition(p), locations(locs)
  {}

  placement_t(partition_t const& p):
    placement_t(p, tensor_t<int>(p.block_shape()))
  {}

  static placement_t join_to_out(placement_t const& p_join, int out_rank);

  static placement_t random(partition_t const& partition, int nloc);
  static placement_t random(vector<partdim_t> const& partdims, int nloc);

  placement_t subset(vector<tuple<int, int>> const& region) const;

  // Args must be all ints or this won't compile
  template <typename... Args>
  int& operator()(Args... args) {
    return locations.operator()(args...);
  }

  template <typename... Args>
  int const& operator()(Args... args) const {
    return locations.operator()(args...);
  }

  int& at(vector<int> idxs) {
    return locations.at(idxs);
  }
  int const& at(vector<int> idxs) const {
    return locations.at(idxs);
  }


  vector<uint64_t> total_shape() const {
    return partition.total_shape();
  }

  vector<int> block_shape() const {
    return partition.block_shape();
  }

  int num_parts() const {
    return partition.num_parts();
  }

  partition_t partition;
  tensor_t<int> locations;
};


