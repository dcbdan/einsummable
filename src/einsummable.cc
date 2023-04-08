#include "einsummable.h"

einsummable_t einsummable_t::from_matmul(uint64_t di, uint64_t dj, uint64_t dk) {
  // ij,jk->ik
  // 02 21  01
  return einsummable_t {
    .join_shape = {di, dk, dj},
    .inns = { {0, 2}, {2, 1} },
    .out_rank = 2,
    .join = scalar_join_t::mul,
    .castable = castable_t::add
  };
}

vector<uint64_t> einsummable_t::out_shape() const {
  return vector<uint64_t>(
    join_shape.begin(),
    join_shape.begin() + out_rank);
}

vector<vector<uint64_t>> einsummable_t::inn_shapes() const {
  vector<vector<uint64_t>> ret(inns.size());
  for(int i = 0; i != inns.size(); ++i) {
    ret[i].reserve(inns[i].size());
    for(int j: inns[i]) {
      ret[i].push_back(join_shape[j]);
    }
  }
  return ret;
}

