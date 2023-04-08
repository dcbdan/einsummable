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

  partition_t const partition;
  tensor_t<int> locations;
};


