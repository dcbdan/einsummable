#pragma once
#include "../../einsummable/setup.h"

// This is an out of place tensor permute that attempts to be cache-oblivious by
// generalizing the cache-oblivious tensor transpose algorithm.
// The algorihtm is to recursively split the input tensor along the largest
// dimension until the resulting tensor block falls below a threshold size. Then a
// naive permute is applied to the small tensor.

// It only supports up to rank-5 tensors.

// TODO: make this multithreaded with a num_threads parameter

struct permute_t {
  permute_t(uint64_t min_block_size);

  // out_perm = {2,0,1} implies we have ijk->kij
  void operator()(
    vector<uint64_t> inn_dims,
    vector<int> out_perm,
    float* out,
    float const* inn) const;

private:
  inline void recurse(
    vector<tuple<uint64_t,uint64_t>>& rngs,
    vector<uint64_t> const& str_inn,
    vector<uint64_t> const& str_out,
    float* out, float const* inn) const;

  static
  tuple<vector<uint64_t>, vector<uint64_t>>
  build_strides(
    vector<uint64_t> const& dims,
    vector<int> const& perm);

  bool has_fuse(vector<uint64_t>& dims, vector<int>& perm) const;

  bool has_singleton(vector<uint64_t>& dims, vector<int>& perm) const;

  void remove(int i, vector<uint64_t>& dims, vector<int>& perm) const;

private:
  uint64_t min_block_size;
};
