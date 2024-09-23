#pragma once
#include "setup.h"

#include "vtensor.h"
#include "partition.h"

struct placement_t {
  placement_t(partition_t const& p, vtensor_t<int> const& locs):
    partition(p), locations(locs)
  {}

  placement_t(partition_t const& p):
    placement_t(p, vtensor_t<int>(p.block_shape()))
  {}

  static placement_t join_to_out(placement_t const& p_join, int out_rank);

  static placement_t random(partition_t const& partition, int nloc);
  static placement_t random(vector<partdim_t> const& partdims, int nloc);

  static placement_t from_wire(string const& str);
  static placement_t from_proto(es_proto::Placement const& p);

  placement_t refine(partition_t const& refined_partition) const;

  placement_t subset(vector<tuple<uint64_t, uint64_t>> const& hrect) const;
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

  string to_wire() const;
  void to_proto(es_proto::Placement& p) const;

  partition_t partition;
  vtensor_t<int> locations;
};

bool operator==(placement_t const& lhs, placement_t const& rhs);
bool operator!=(placement_t const& lhs, placement_t const& rhs);

vector<placement_t> from_proto_placement_list(es_proto::PlacementList const& pl);
es_proto::PlacementList to_proto_placement_list(vector<placement_t> const& pl);

vector<placement_t> from_wire_placement_list(string const& str);
string to_wire_placement_list(vector<placement_t> const& pl);

