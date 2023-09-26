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

  // Note: this sort of simplification can be made at the
  // kernel level:
  //   ijkl,klmn->ijmn is the same as
  //   a b, b c ->a c
  einsummable_t merge_adjacent_dims() const;

  string to_wire() const;
  static einsummable_t from_wire(string const& str);

  void to_proto(es_proto::Einsummable& e) const;
  static einsummable_t from_proto(es_proto::Einsummable const& e);

  // ij,jk->ik
  // 02 21  01
  static einsummable_t from_matmul(
    uint64_t di, uint64_t dj, uint64_t dk,
    dtype_t dtype = default_dtype());

  static einsummable_t from_matmul_ss( // ij,jk->ik
    uint64_t di, uint64_t dj, uint64_t dk,
    dtype_t dtype = default_dtype());
  static einsummable_t from_matmul_ts( // ji,jk->ik
    uint64_t di, uint64_t dj, uint64_t dk,
    dtype_t dtype = default_dtype());
  static einsummable_t from_matmul_st( // ij,kj->ik
    uint64_t di, uint64_t dj, uint64_t dk,
    dtype_t dtype = default_dtype());
  static einsummable_t from_matmul_tt( // ji,kj->ik
    uint64_t di, uint64_t dj, uint64_t dk,
    dtype_t dtype = default_dtype());

  static einsummable_t with_new_shape(
    einsummable_t const& e, vector<uint64_t> const& new_join_shape);

  static tuple<vector<vector<int>>, int>
  parse_str(string einsummable_str);

  string create_contraction_vjp_string(int which_inn);

  static string create_unary_vjp_string(vector<int> inn, int rank);

  static string create_batch_matmul_string(int lhs_rank, int rhs_rank, bool t_lhs, bool t_rhs);

  static string create_binary_vjp_string(vector<int> argument_shape, vector<int> other_shape);

  string create_reduction_vjp_string();

  static tuple<string, vector<string>>
  make_str_terms(vector<vector<int>> const& inns, int out_rank);

  static string make_str(vector<vector<int>> const& inns, int out_rank);

  static string normalize_str(string const& str);

  static bool valid_inns_out(
    vector<vector<int>> const& inns,
    int out_rank);

  // will return None if has_broadcast
  static optional<vector<uint64_t>> construct_join_shape(
    vector<vector<int>> const& inns,
    vector<vector<uint64_t>> const& inn_shapes);

  dtype_t out_dtype() const;

  uint64_t out_size() const;

  uint64_t out_nelem() const;

  vector<uint64_t> out_shape() const;

  vector<dtype_t> inn_dtypes() const;

  dtype_t inn_dtype(int which_inn) const;

  vector<vector<uint64_t>> inn_shapes() const;

  vector<vector<int>> input_idxs(vector<int> const& join_idx) const;

  string str() const;

  tuple<string, vector<string>> str_terms() const;

  std::size_t hash() const;

  // A straight elementwise operation can be computed by
  //   for(i = 0, i != product(join_shape); ++i) {
  //     out[i] = join(inn1[i], ..., innN[i]);
  //   }
  // Note on straight-elementwise vs elementwise in this context:
  // Here (ij->ij) and (ijk,ijk->ijk) are straight_elementwise,
  // but               (ikj,ijk->ijk) is elementwise but not straight
  // as a transposition happens and
  //                   (ijk,ijk->ij) is not elementwise since an aggregation
  //                   happens.
  bool is_straight_elementwise() const;

  bool is_permutation() const;

  bool has_aggregation() const;

  bool is_contraction() const;

  // is broadcast means this op is just doing a broadcast on a single
  // element and nothing else
  bool is_broadcast() const;

  // is_broadcast includes permutation style broadcasts like
  // ijk->kjil. straight broadcast must include all broadcast
  // dims followed by the input without any permutation:
  //   so 123->0123 is a straight broadcast but
  //      012->0123 is not
  bool is_straight_broadcast() const;

  // has broadcast is true if there are any broadcast modes
  bool has_broadcast() const;

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
  static vector<vector<T>> get_inputs_from_join_(
    vector<vector<int>> const& inns,
    vector<T> const& join_ts)
  {
    vector<vector<T>> ret(inns.size());
    for(int i = 0; i != inns.size(); ++i) {
      ret[i].reserve(inns.size());
      for(auto const& j: inns[i]) {
        ret[i].push_back(join_ts[j]);
      }
    }

    return ret;
  }

  template <typename T>
  vector<vector<T>> get_inputs_from_join(vector<T> const& join_ts) const
  {
    if(join_shape.size() != join_ts.size()) {
      throw std::runtime_error("einsummable_t::input_idxs");
    }

    return get_inputs_from_join_(inns, join_ts);
  }

  template <typename T, typename F>
  static optional<vector<T>>
  construct_join_shape_(
    vector<vector<int>> const& inns,
    vector<vector<T>> const& inn_shapes,
    T const& unassigned,
    F equals)
  {
    if(inns.size() != inn_shapes.size()) {
      std::cout << "inns.size() != inn_shapes.size()" << std::endl;
      return std::nullopt;
    }

    vector<T> join_shape;
    for(int which_inn = 0; which_inn != inns.size(); ++which_inn) {
      auto const& shape = inn_shapes[which_inn];
      auto const& is = inns[which_inn];
      for(int inn_i = 0; inn_i != is.size(); ++inn_i) {
        int const& out_i = is[inn_i];
        T const& inn_sz = shape[inn_i];

        if(out_i >= join_shape.size()) {
          join_shape.resize(out_i+1, unassigned);
        }
        T& out_sz = join_shape[out_i];
        if(equals(out_sz, unassigned)) {
          out_sz = inn_sz;
        } else {
          if(!equals(out_sz, inn_sz)) {
            std::cout << "Something in the for loop...." << std::endl;
            return std::nullopt;
          }
        }
      }
    }

    if(join_shape.size() == 0) {
      std::cout << "join_shape.size()...." << std::endl;
      return std::nullopt;
    }

    auto iter = std::find(join_shape.begin(), join_shape.end(), unassigned);
    if(iter != join_shape.end()) {
      std::cout << "iter != join_shape.end()..." << std::endl;
      return std::nullopt;
    }

    return join_shape;
  }

  optional<vector<einsummable_t>> derivative(int which_inn);
  bool deri_depends_on(int which_inn);

private:
  optional<vector<einsummable_t>> ewu_derivative(int which_inn);
  optional<vector<einsummable_t>> ewb_derivative(int which_inn);
};

std::ostream& operator<<(std::ostream& out, einsummable_t const& e);

template <> struct std::hash<einsummable_t> {
  inline std::size_t operator()(einsummable_t const& e) const
  {
    return e.hash();
  }
};

bool operator==(einsummable_t const& lhs, einsummable_t const& rhs);
bool operator!=(einsummable_t const& lhs, einsummable_t const& rhs);
