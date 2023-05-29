#pragma once
#include "../base/setup.h"

#include "scalarop.h"

#include "einsummable.pb.h"

struct einsummable_t {
  vector<uint64_t> join_shape;

  vector<vector<int>> inns;
  int out_rank;

  scalarop_t join;

  // may be none when out_rank == join_rank
  optional<castable_t> castable;

  // Note: this sort of simplification can be made at the
  // kernel level:
  //   ijkl,klmn->ijmn is the same as
  //   a b, b c ->a c

  // Consider batched matrix multiply into
  // the same output:
  //   bij,bjk->ik
  // There are two possible einsummable representations:
  //   203 231->01 and
  //   302 321->01
  // Both parse_str and this constructor maintains
  // the first option.
  einsummable_t(
    vector<uint64_t> join_shape,
    vector<vector<int>> inns,
    int out_rank,
    scalarop_t join,
    optional<castable_t> castable = std::nullopt);

  string to_wire() const;
  static einsummable_t from_wire(string const& str);

  void to_proto(es_proto::Einsummable& e) const;
  static einsummable_t from_proto(es_proto::Einsummable const& e);

  // ij,jk->ik
  // 02 21  01
  static einsummable_t from_matmul(uint64_t di, uint64_t dj, uint64_t dk);

  static einsummable_t from_matmul_ss(uint64_t di, uint64_t dj, uint64_t dk); // ij,jk->ik
  static einsummable_t from_matmul_ts(uint64_t di, uint64_t dj, uint64_t dk); // ji,jk->ik
  static einsummable_t from_matmul_st(uint64_t di, uint64_t dj, uint64_t dk); // ij,kj->ik
  static einsummable_t from_matmul_tt(uint64_t di, uint64_t dj, uint64_t dk); // ji,kj->ik

  static einsummable_t with_new_shape(
    einsummable_t const& e, vector<uint64_t> const& new_join_shape);

  static tuple<vector<vector<int>>, int>
  parse_str(string einsummable_str);

  static bool valid_inns_out(
    vector<vector<int>> const& inns,
    int out_rank);

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
  // TODO: what is ij,i->ij ? That is, the left input can be donated but
  //                          this returns false

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

std::istream& operator>>(std::istream& inn, castable_t& c);

