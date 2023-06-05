#include "placement.h"

placement_t placement_t::join_to_out(placement_t const& p_join, int out_rank)
{
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
    return ret;
  };

  // TODO set em here
  vector<int> out_shape = ret.block_shape();
  vector<int> idxs(out_rank, 0);
  do {
    ret.at(idxs) = get_max_loc(counts.at(idxs));
  } while(increment_idxs(out_shape, idxs));

  return ret;
}

placement_t placement_t::random(partition_t const& partition, int nloc) {
  placement_t ret(partition);
  for(int& loc: ret.locations.get()) {
    loc = runif(nloc);
  }
  return ret;
}

placement_t placement_t::random(vector<partdim_t> const& partdims, int nloc) {
  return random(partition_t(partdims), nloc);
}

placement_t placement_t::refine(partition_t const& refined_partition) const {
  if(!refined_partition.refines(partition)) {
    throw std::runtime_error(
      "placement_t refine: must be given refining partition");
  }

  auto refi_block_shape = refined_partition.block_shape();

  tensor_t<int> ret(refi_block_shape, -1);

  vector<int> refi_index(refi_block_shape.size(), 0);
  do {
    auto hrect = refined_partition.get_hrect(refi_index);
    vector<int> index = partition.get_index_covering(hrect);
    ret.at(refi_index) = locations.at(index);
  } while(increment_idxs(refi_block_shape, refi_index));

  return placement_t(refined_partition, ret);
}

placement_t placement_t::subset(vector<tuple<int, int>> const& region) const {
  return placement_t(
    partition.subset(region),
    locations.subset(region));
}

