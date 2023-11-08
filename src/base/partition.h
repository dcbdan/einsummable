#pragma once
#include "setup.h"

#include "partdim.h"
#include "vtensor.h"

#include "einsummable.pb.h"

struct partition_t {
  partition_t(vector<partdim_t> const& p);

  static partition_t singleton(vector<uint64_t> shape);

  static partition_t from_wire(string const& str);
  static partition_t from_proto(es_proto::Partition const& p);

  vector<uint64_t> total_shape() const;

  int num_parts() const;

  vector<int> block_shape() const;

  vector<int> from_bid(int const& bid) const;

  bool refines(partition_t const& other) const;

  vector<uint64_t> tensor_shape_at(vector<int> const& idxs) const;

  vtensor_t<uint64_t> all_block_sizes() const;

  uint64_t block_size_at_bid(int bid) const;

  partition_t subset(vector<tuple<int, int>> const& region) const;

  partition_t subset(vector<tuple<uint64_t, uint64_t>> const& hrect) const;

  // Get the hyper-rectanuglar set represnted by this index
  vector<tuple<uint64_t, uint64_t>>
  get_hrect(vector<int> const& idxs) const;

  // hrect:  hyper-rectangular subset of uint64s
  // region: hyper-rectangular subset of blocks
  vector<tuple<int,int> >
  get_exact_region(
    vector<tuple<uint64_t,uint64_t>> const& hrect) const;

  vector<tuple<int,int> >
  get_region(
    vector<tuple<uint64_t,uint64_t>> const& hrect) const;

  // If multiple index cover the area given by hrect,
  // throw an error
  vector<int> get_index_covering(
    vector<tuple<uint64_t,uint64_t>> const& hrect) const;

  string to_wire() const;
  void to_proto(es_proto::Partition& p) const;

  vector<partdim_t> partdims;
};

bool operator==(partition_t const& lhs, partition_t const& rhs);
bool operator!=(partition_t const& lhs, partition_t const& rhs);

std::ostream& operator<<(std::ostream& out, partition_t const& p);

// Putting inline methods of partition_t here
// (for what appears to be marginal performance improvements)
#include "partition_.h"


