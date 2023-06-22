#pragma once
#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

struct contraction_t {
private:
  struct permute_info_t {
    vector<uint64_t> inn_shape;
    vector<int> out_perm;

    static permute_info_t from_inn_shape(
      vector<uint64_t> inn_shape,
      vector<int> inn_ord,
      vector<int> out_ord)
    {
      return permute_info_t { inn_shape, as_out_perm(inn_ord, out_ord) };
    }

    static permute_info_t from_out_shape(
      vector<uint64_t> out_shape,
      vector<int> inn_ord,
      vector<int> out_ord)
    {
      auto out_perm = as_out_perm(inn_ord, out_ord);
      auto inn_shape = backward_permute(out_perm, out_shape);
      return permute_info_t { inn_shape, out_perm };
    }

    permute_info_t drop_leading(int n) const;

    // If the permutation is ijk->ikj,
    // then we can think of this as |i|
    // jk->kj transposes
    //   num_leading_modes = 1,
    //   leading_nelem = |i|
    //   perm_nelem = |j|*|k|
    int num_leading_modes() const;

    bool is_no_op() const {
      return num_leading_modes() == out_perm.size();
    }

    uint64_t leading_nelem() const;
  };

public:
  dtype_t dtype;

  uint64_t lhs_work_offset;
  uint64_t rhs_work_offset;
  uint64_t out_work_offset;

  uint64_t workspace_size;

  uint64_t stride_lhs;
  uint64_t stride_rhs;
  uint64_t stride_out;

  uint64_t nb;
  uint64_t ni;

  optional<permute_info_t> lhs_p;
  optional<permute_info_t> rhs_p;
  optional<permute_info_t> out_p;

  struct {
    uint64_t nb;
    uint64_t ni;
    uint64_t nj;
    uint64_t nk;

    bool lhs_t;
    bool rhs_t;
  } inner;

  void operator()(
    void* workspace,
    void* out,
    void const* lhs, void const* rhs);

  static bool can_make(
    vector<int> const& lhs_inn_modes,
    vector<int> const& rhs_inn_modes,
    int out_rank);

  static contraction_t make(
    dtype_t dtype,
    vector<uint64_t> const& shape,
    vector<int> const& lhs_inn_modes,
    vector<int> const& rhs_inn_modes,
    int out_rank);

private:
  struct batching_t {
    // bij,bjk->bik
    // bij,bkj->bik
    // bji,bjk->bik
    // bji,bkj->bik
    vector<int> bs;
    vector<int> is;
    vector<int> js;
    vector<int> ks;
    bool lhs_t;
    bool rhs_t;

    vector<int> modes_lhs() const {
      return vector_concatenate(bs,
        lhs_t                      ?
        vector_concatenate(js, is) :
        vector_concatenate(is, js));
    }
    vector<int> modes_rhs() const {
      return vector_concatenate(bs,
        rhs_t                      ?
        vector_concatenate(ks, js) :
        vector_concatenate(js, ks));
    }
    vector<int> modes_out() const {
      return vector_concatenate(bs, vector_concatenate(is, ks));
    }
  };

  static contraction_t make(
    dtype_t dtype,
    vector<uint64_t> const& shape,
    vector<int> const& lhs_inn_modes,
    vector<int> const& rhs_inn_modes,
    int out_rank,
    batching_t const& plan);

  static optional<tuple<
    vector<int>, vector<int>, vector<int>, vector<int> >>
  make_bs_is_js_ks(
    vector<int> const& lhs_inn_modes,
    vector<int> const& rhs_inn_modes,
    int out_rank);

};


