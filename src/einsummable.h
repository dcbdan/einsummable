#pragma once
#include "setup.h"

enum class castable_t { add, mul, min, max };

enum class scalar_join_t { mul };

struct einsummable_t {
  vector<uint64_t> join_shape;

  vector<vector<int>> inns;
  int out_rank;

  scalar_join_t join;
  castable_t castable;

  // ij,jk->ik
  // 02 21  01
  static einsummable_t from_matmul(uint64_t di, uint64_t dj, uint64_t dk);

  vector<uint64_t> out_shape() const;

  vector<vector<uint64_t>> inn_shapes() const;
};

