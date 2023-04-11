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

einsummable_t einsummable_t::with_new_shape(
  einsummable_t const& e,
  vector<uint64_t> const& new_join_shape)
{
  if(e.join_shape.size() != new_join_shape.size()) {
    throw std::runtime_error("einsummable_t::with_new_shape");
  }
  return einsummable_t {
    .join_shape = new_join_shape,
    .inns       = e.inns,
    .out_rank   = e.out_rank,
    .join       = e.join,
    .castable   = e.castable
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

vector<vector<int>>
einsummable_t::input_idxs(vector<int> const& join_idx) const
{
  if(join_shape.size() != join_idx.size()) {
    throw std::runtime_error("einsummable_t::input_idxs");
  }

  vector<vector<int>> ret(join_idx.size());
  for(int i = 0; i != ret.size(); ++i) {
    ret[i].reserve(inns.size());
    for(auto const& j: inns[i]) {
      ret[i].push_back(join_idx[j]);
    }
  }

  return ret;
}


