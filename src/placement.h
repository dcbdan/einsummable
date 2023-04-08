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

  vector<uint64_t> total_shape() const {
    return partition.total_shape();
  }

  vector<int> block_shape() const {
    return partition.block_shape();
  }

  int num_parts() const {
    return partition.num_parts();
  }

  partition_t const partition;
  tensor_t<int> locations;
};


