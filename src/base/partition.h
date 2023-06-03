#pragma once
#include "setup.h"

#include "partdim.h"

struct partition_t {
  partition_t(vector<partdim_t> const& p);

  static partition_t singleton(vector<uint64_t> shape);

  vector<uint64_t> total_shape() const;

  int num_parts() const;

  vector<int> block_shape() const;

  bool refines(partition_t const& other) const;

  vector<uint64_t> tensor_shape_at(vector<int> const& idxs) const;

  // Get the hyper-rectanuglar set represnted by this index
  vector<tuple<uint64_t, uint64_t>>
  get_hrect(vector<int> const& idxs) const;

  // hrect:  hyper-rectangular subset of uint64s
  // region: hyper-rectangular subset of blocks
  vector<tuple<int,int> >
  get_exact_region(
    vector<tuple<uint64_t,uint64_t>> const& region) const;

  vector<tuple<int,int> >
  get_region(
    vector<tuple<uint64_t,uint64_t>> const& region) const;

  vector<partdim_t> partdims;
};

bool operator==(partition_t const& lhs, partition_t const& rhs);
bool operator!=(partition_t const& lhs, partition_t const& rhs);

std::ostream& operator<<(std::ostream& out, partition_t const& p);

