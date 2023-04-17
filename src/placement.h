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

  static placement_t join_to_out(placement_t const& p_join, int out_rank) {
    // There is nothing to do if p_join has the rank
    int p_join_rank = p_join.total_shape().size();
    if(p_join_rank == out_rank) {
      return p_join;
    }
    if(p_join_rank > out_rank) {
      throw std::runtime_error("invalid value for out_rank in placement_t::join_to_out");
    }

    partition_t partition_ret(vector<partdim_t>(
      p_join.partition.partdims.begin(),
      p_join.partition.partdims.begin() + out_rank));

    placement_t ret(partition_ret);

    vector<int> join_shape = p_join.block_shape();
    vector<int> join_idxs(join_shape.size(), 0);

    tensor_t<map<int, int>> counts(ret.block_shape());
    do {
      vector<int> idxs(join_idxs.begin(), join_idxs.begin() + out_rank);
      counts.at(idxs)[p_join.at(join_idxs)]++;
    } while(increment_idxs(join_shape, join_idxs));

    auto get_max_loc = [](map<int, int> const& items) {
      int ret  = items.begin()->first;
      int best = items.begin()->second;
      for(auto const& [loc,score]: items) {
        if(score > best) {
          ret = loc;
          best = score;
        }
      }
      return best;
    };

    // TODO set em here
    vector<int> out_shape = ret.block_shape();
    vector<int> idxs(out_rank, 0);
    do {
      ret.at(idxs) = get_max_loc(counts.at(idxs));
    } while(increment_idxs(out_shape, idxs));

    return ret;
  }

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

  partition_t const partition;
  tensor_t<int> locations;
};


