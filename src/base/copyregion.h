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
//   |   33333444444444|   <- need to write into
//   |        |        |      these four blocks with
//                            four separate ops
//
// What the code should roughly look like to
// do the computation:
//
//   copy_region_t indexer(out, inn, index);
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
    vector<int> const& inn_index);

  vector<diminfo_t> info;

  bool increment();

private:
  partition_t const& out;
  partition_t const& inn;
  vector<int> const& inn_index;

  vector<tuple<uint64_t, uint64_t>> inn_hrect;
  vector<tuple<uint64_t, uint64_t>> out_hrect;
  vector<tuple<int,int>> region;
  vector<int> index;

  // use index to set the op info
  void set_info();
};

// This is like copy region except doesn't take
// an inn_index. It is also more efficient.
//
// Given two partitions, get the refinement of those two
// partitions and iterate through all blocks in the refinement,
// updating the idx / index / offsets.
//
// (Here, the offset is with respect to the refined block,
//  not with respect to the full relation / partition aa or bb)
struct copyregion_full_t {
  copyregion_full_t(
    partition_t const& aa,
    partition_t const& bb);

  bool increment();

  int idx_aa;
  vector<int> index_aa;
  vector<uint64_t> offset_aa;

  int idx_bb;
  vector<int> index_bb;
  vector<uint64_t> offset_bb;

  vector<uint64_t> size;

private:
  int idx_rr;
  vector<int> index_rr;
  vector<int> block_shape_rr;

  vector<int> strides_aa;
  vector<vector<int>> breaks_aa;
  vector<int> rem_idx_aa;
  vector<int> rem_aa;

  vector<int> strides_bb;
  vector<vector<int>> breaks_bb;
  vector<int> rem_idx_bb;
  vector<int> rem_bb;

  partition_t const& aa;
  partition_t const& bb;
  partition_t        rr;
};
