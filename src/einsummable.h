#pragma once
#include "setup.h"

enum class castable_t { add, mul, min, max };

enum class scalar_join_t { add, sub, mul, relu, negate, min, max };

struct einsummable_t {
  vector<uint64_t> join_shape;

  vector<vector<int>> inns;
  int out_rank;

  scalar_join_t join;
  castable_t castable;

  // ij,jk->ik
  // 02 21  01
  static einsummable_t from_matmul(uint64_t di, uint64_t dj, uint64_t dk);
  static einsummable_t with_new_shape(
    einsummable_t const& e, vector<uint64_t> const& new_join_shape);

  vector<uint64_t> out_shape() const;

  vector<vector<uint64_t>> inn_shapes() const;

  vector<vector<int>> input_idxs(vector<int> const& join_idx) const;

  template <typename T>
  vector<T> get_input_from_join(vector<T> const& join_ts, int which_inn) const
  {
    if(join_shape.size() != join_ts.size()) {
      throw std::runtime_error("einsummable_t::input_idxs");
    }

    vector<T> ret;
    ret.reserve(inns[which_inn].size());
    for(auto const& i: inns[which_inn]) {
      ret.push_back(join_ts[i]);
    }

    return ret;
  }

  template <typename T>
  vector<vector<T>> get_inputs_from_join(vector<T> const& join_ts) const
  {
    if(join_shape.size() != join_ts.size()) {
      throw std::runtime_error("einsummable_t::input_idxs");
    }

    vector<vector<T>> ret(inns.size());
    for(int i = 0; i != inns.size(); ++i) {
      ret[i].reserve(inns.size());
      for(auto const& j: inns[i]) {
        ret[i].push_back(join_ts[j]);
      }
    }

    return ret;
  }
};

