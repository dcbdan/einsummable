#pragma once
#include "setup.h"

#include "scalarop.h"

struct einsummable_t {
  vector<uint64_t> join_shape;

  vector<vector<int>> inns;
  int out_rank;

  scalarop_t join;

  // may be none when out_rank == join_rank
  optional<castable_t> castable;

  // Consider batched matrix multiply into
  // the same output:
  //   bij,bjk->ik
  // There are two possible einsummable representations:
  //   203 231->01 and
  //   302 321->01
  // TODO: should this form of indeterminism be
  //       allowed? should there be a simplification?
  // Another simplification that could be made:
  //   ijkl,klmn->ijmn is the same as
  //   a b, b c ->a c

  // ij,jk->ik
  // 02 21  01
  static einsummable_t from_matmul(uint64_t di, uint64_t dj, uint64_t dk);

  static einsummable_t from_matmul_ss(uint64_t di, uint64_t dj, uint64_t dk); // ij,jk->ik
  static einsummable_t from_matmul_ts(uint64_t di, uint64_t dj, uint64_t dk); // ji,jk->ik
  static einsummable_t from_matmul_st(uint64_t di, uint64_t dj, uint64_t dk); // ij,kj->ik
  static einsummable_t from_matmul_tt(uint64_t di, uint64_t dj, uint64_t dk); // ji,kj->ik

  static einsummable_t with_new_shape(
    einsummable_t const& e, vector<uint64_t> const& new_join_shape);

  vector<uint64_t> out_shape() const;

  vector<vector<uint64_t>> inn_shapes() const;

  vector<vector<int>> input_idxs(vector<int> const& join_idx) const;

  string str() const;

  // Note on straight-elementwise vs elementwise in this context:
  // Here (ij->ij) and (ijk,ijk->ijk) are straight_elementwise,
  // but               (ikj,ijk->ijk) is elementwise but not straight
  // as a transposition happens and
  //                   (ijk,ijk->ij) is not elementwise since an aggregation
  //                   happens.
  bool is_straight_elementwise() const;

  bool has_aggregation() const;

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

std::ostream& operator<<(std::ostream& out, einsummable_t const& e);
std::ostream& operator<<(std::ostream& out, castable_t const& c);
std::ostream& operator<<(std::ostream& out, optional<castable_t> const& maybe_c);

