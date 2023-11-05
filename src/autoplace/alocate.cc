#include "alocate.h"
#include "buildtrees.h"

#include "../base/permute.h"

struct structured_placement_t {
  structured_placement_t(
    int nlocs,
    vector<uint64_t> const& size,
    vector<int> const& splits);

  structured_placement_t(
    int nlocs,
    vector<uint64_t> const& size,
    vector<int> const& splits,
    vector<int> const& priority);

  struct dim_t {
    uint64_t size;
    int split;
    int priority;
  };

  int nlocs;
  vector<dim_t> dims;

  vector<int> get_priorities() const {
    return vector_from_each_member(dims, int, priority);
  }
  vector<int> get_block_shape() const {
    return vector_from_each_member(dims, int, split);
  }
  vector<uint64_t> get_total_shape()  const {
    return vector_from_each_member(dims, uint64_t, size);
  }

  // Suppose the priorities are {1,2,0}.
  //                           di dj dk
  // Then the last dk moves the fastest, followed by di, then dj.
  // So we create a "tensor" of shape (dk,di,dj)
  // with locations 0,1,2,...,nlocs and then we permute
  // the locations kij->ijk.
  // Thats how to build the location tensor
  placement_t as_placement() const {
    vector<partdim_t> pds;
    for(int rank = 0; rank != dims.size(); ++rank) {
      pds.push_back(partdim_t::split(dims[rank].size, dims[rank].split));
    }

    auto out_block_shape = get_block_shape();
    int n_blocks = product(out_block_shape);

    vector<int> locs(n_blocks);
    {
      // get the permutation "kij->ijk"
      vector<int> out_perm;
      {
        vector<int> iota_(dims.size());
        std::iota(iota_.begin(), iota_.end(), 0);
        out_perm = as_out_perm(
          get_priorities(),
          iota_);
      }

      // get the input shape
      vector<uint64_t> inn_block_shape;
      {
        // convert out_block_shape to uint64_t
        vector<uint64_t> out_block_shape_;
        out_block_shape_.reserve(out_block_shape.size());
        for(int const& x: out_block_shape) {
          out_block_shape_.push_back(x);
        }

        inn_block_shape = backward_permute(out_perm, out_block_shape_);
      }

      // the input locations
      vector<int> inn_locs(n_blocks);
      std::iota(inn_locs.begin(), inn_locs.end(), 0);
      for(int& loc: inn_locs) {
        loc = loc % nlocs;
      }

      // now permute directly into locs
      permute_t permuter(1024);
      permuter(inn_block_shape, out_perm, locs.data(), inn_locs.data());
    }

    return placement_t(
      partition_t(pds),
      vtensor_t<int>(out_block_shape, locs));
  }

  // TODO: is this needed?
  placement_t as_placement_chopped(int new_rank) const {
    int full_rank = dims.size();

    if(new_rank > full_rank) {
      throw std::runtime_error("invalid rank to chop to");
    }

    if(new_rank == full_rank) {
      return as_placement();
    }

    placement_t full_pl = as_placement();

    partition_t new_partition = [&] {
      auto const& full_pds = full_pl.partition.partdims;
      vector<partdim_t> pds(
        full_pds.begin(),
        full_pds.begin() + new_rank);
      return partition_t(pds);
    }();

    vector<int> new_locs;
    {
      auto full_block_shape = full_pl.block_shape();

      vector<tuple<int,int>> region;
      region.reserve(full_rank);
      for(int r = 0; r != new_rank; ++r) {
        region.emplace_back(0, full_block_shape[r]);
      }
      for(int r = new_rank; r != full_rank; ++r) {
        region.emplace_back(0, 1);
      }

      new_locs = full_pl.locations.subset(region).get();
    }

    return placement_t(
      new_partition,
      vtensor_t<int>(new_partition.block_shape(), new_locs));
  }
};

vector<placement_t> autolocate(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs)
{
  // 1. build a graph of trees
  // 2. for each tree in graph order,
  //    solve for the best locations for that tree
  // 3. build the vector of placements
  return {};
}
