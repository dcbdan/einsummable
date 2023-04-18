#pragma once
#include "setup.h"

#include "indexer.h"
#include "partition.h"

// Given
// 1. an output partition
// 2. an input partition of the same overall shape
// 3. an input index,
// Iterate through all "ops" that will write
// the input subtensor at index into the output.
//
// Here, an op is the specification that would write
// some subtensor of the input tensor into some other
// subtensor of the output tensor.
//
// Example
//   Out partition           Inn partition
//   |        |        |     |   |             |
//   |        |        |     |   |             |
//   |--------|--------|     |   |             |
//   |        |        |     |---|-------------|
//   |        |        |     |   | Index points|
//   |--------|--------|     |   | here        |
//   |        |        |     |---|-------------|
//   |        |        |     |   |             |
//
//   Out partition again
//   |        |        |
//   |        |        |
//   |--------|--------|
//   |        |        |
//   |   11111222222222|
//   |---33333444444444|
//   |   33333444444444|   <- need to write into the
//   |        |        |      these four blocks with
//                            four separate ops
//
// What the code should roughly look like to
// do the computation:
//
//   copy_index_ops_t indexer(out, inn, index);
//   do {
//     out index  = [x.idx        for x in indexer.info]
//     offset inn = [x.offset_inn for x in indexer.info]
//     offset out = [x.offset_out for x in indexer.info]
//     copy shape = [x.size       for x in indexer.info]
//
//     out buffer = get out buffer from out index
//     out shape  = get out shape from out index and out partition
//
//     do the copy from inn buffer to out buffer using
//       out shape, inn shape, copy shape, offset inn, offset out
//   } while(indexer.increment());
struct copyregion_t {
  struct diminfo_t {
    int idx;
    uint64_t offset_inn;
    uint64_t offset_out;
    uint64_t size;
  };

  copyregion_t(
    partition_t const& out,
    partition_t const& inn,
    vector<int> const& inn_index)
    : out(out), inn(inn), inn_index(inn_index)
  {
    if(!vector_equal(out.total_shape(), inn.total_shape())) {
      throw std::runtime_error("total shapes do not match");
    }

    inn_hrect = inn.get_hrect(inn_index);
    region = out.get_region(inn_hrect);
    index = vector_mapfst(region);

    info = vector<diminfo_t>(index.size());

    set_info();
  }

  vector<diminfo_t> info;

  bool increment() {
    if(increment_idxs_region(region, index)) {
      set_info();
      return true;
    }
    return false;
  }

private:
  partition_t const& out;
  partition_t const& inn;
  vector<int> const& inn_index;

  vector<tuple<uint64_t, uint64_t>> inn_hrect;
  vector<tuple<uint64_t, uint64_t>> out_hrect;
  vector<tuple<int,int>> region;
  vector<int> index;

  // use index to set the op info
  void set_info() {
    out_hrect = out.get_hrect(index);

    for(int i = 0; i != index.size(); ++i) {
      auto const& [o_beg, o_end] = out_hrect[i];
      auto const& [i_beg, i_end] = inn_hrect[i];

      uint64_t beg = std::max(o_beg, i_beg);
      uint64_t end = std::min(o_end, i_end);

      info[i].idx = index[i];
      info[i].offset_inn = beg - i_beg;
      info[i].offset_out = beg - o_beg;
      info[i].size = end - beg;
    }
  }
};


