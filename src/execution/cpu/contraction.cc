#include "contraction.h"
#include "kernels.h"

void contraction_t::operator()(
  void* workspace,
  void* out,
  void const* lhs, void const* rhs)
{
  void* lhs_work_ = (char*)workspace + lhs_work_offset;
  void* rhs_work_ = (char*)workspace + rhs_work_offset;
  void* out_work_ = (char*)workspace + out_work_offset;

  // bij,bjk->bik
  for(uint64_t b = 0; b != nb; ++b) {
  for(uint64_t i = 0; i != ni; ++i) {
    void* out_data = (char*)out + stride_out*(b*ni + i);
    void* lhs_data = (char*)lhs + stride_lhs*(b*ni + i);
    void* rhs_data = (char*)rhs + stride_rhs*b;

    void* lhs_work;
    if(lhs_p) {
      lhs_work = lhs_work_;
      permute_kernel(dtype, 1024,
        lhs_p.value().inn_shape,
        lhs_p.value().out_perm,
        lhs_work, lhs_data);
    } else {
      lhs_work = lhs_data;
    }

    void* rhs_work;
    if(rhs_p) {
      rhs_work = rhs_work_;
      permute_kernel(dtype, 1024,
        rhs_p.value().inn_shape,
        rhs_p.value().out_perm,
        rhs_work, rhs_data);
    } else {
      rhs_work = rhs_data;
    }

    void* out_work;
    if(out_p) {
      out_work = out_work_;
    } else {
      out_work = out_data;
    }

    batch_matrix_multiply(
      dtype,
      inner.nb,
      true, true, true, // have batching
      inner.ni, inner.nj, inner.nk,
      inner.lhs_t, inner.rhs_t,
      out_work, lhs_work, rhs_work);

    if(out_p) {
      permute_kernel(dtype, 1024,
        out_p.value().inn_shape,
        out_p.value().out_perm,
        out_data, out_work);
    }
  }}
}

bool contraction_t::can_make(
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank)
{
  return bool(make_bs_is_js_ks(lhs_inn_modes, rhs_inn_modes, out_rank));
}

optional<tuple<
  vector<int>, vector<int>, vector<int>, vector<int> >>
contraction_t::make_bs_is_js_ks(
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank)
{
  vector<int> bs;
  vector<int> is;
  vector<int> js;
  vector<int> ks;

  int join_rank = 1 + std::max(
    *std::max_element(lhs_inn_modes.begin(), lhs_inn_modes.end()),
    *std::max_element(rhs_inn_modes.begin(), rhs_inn_modes.end()));

  for(int i = 0; i != join_rank; ++i) {
    bool in_lhs = std::find(
      lhs_inn_modes.begin(), lhs_inn_modes.end(), i) != lhs_inn_modes.end();
    bool in_rhs = std::find(
      rhs_inn_modes.begin(), rhs_inn_modes.end(), i) != rhs_inn_modes.end();
    bool in_out = i < out_rank;

    if(in_lhs && in_rhs && in_out) {
      bs.push_back(i);
    } else if(in_lhs && in_rhs && !in_out) {
      js.push_back(i);
    } else if(in_lhs && !in_rhs && in_out) {
      is.push_back(i);
    } else if(!in_lhs && in_rhs && in_out) {
      ks.push_back(i);
    } else {
      return std::nullopt;
    }
  }

  using ret_t = tuple<vector<int>, vector<int>, vector<int>, vector<int> >;
  return ret_t{bs,is,js,ks};
}

contraction_t contraction_t::make(
  dtype_t dtype,
  vector<uint64_t> const& shape,
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank)
{
  auto maybe = make_bs_is_js_ks(lhs_inn_modes, rhs_inn_modes, out_rank);
  if(!maybe) {
    throw std::runtime_error(
      "one-sided aggs like k in ijk,ij->i aren't supported "
      "nor are broadcasting outs like z in ij,jk->ikz");
  }

  auto& [bs,is,js,ks] = maybe.value();

  optional<contraction_t> ret;

  DOUT(lhs_inn_modes << " " << rhs_inn_modes);

  do { do { do { do {
  for(bool lhs_t: {false, true}) { // do false first to avoid
  for(bool rhs_t: {false, true}) { // transpositions in the event of ties
    auto plan = make(
      dtype, shape, lhs_inn_modes, rhs_inn_modes, out_rank,
      batching_t { bs, is, js, ks, lhs_t, rhs_t });
    if(ret) {
      if(plan.workspace_size < ret.value().workspace_size) {
        batching_t b { bs, is, js, ks, lhs_t, rhs_t };
        DOUT("BEST " << b.modes_lhs() << " " << b.modes_rhs() << " ");
        ret = plan;
      }
    } else {
      batching_t b { bs, is, js, ks, lhs_t, rhs_t };
      DOUT("SET " << b.modes_lhs() << " " << b.modes_rhs() << " ");
      ret = plan;
    }
  }}
  } while(std::next_permutation(bs.begin(), bs.end()));
  } while(std::next_permutation(is.begin(), is.end()));
  } while(std::next_permutation(js.begin(), js.end()));
  } while(std::next_permutation(ks.begin(), ks.end()));

  return ret.value();
}

contraction_t contraction_t::make(
  dtype_t dtype,
  vector<uint64_t> const& shape,
  vector<int> const& lhs_inn_modes,
  vector<int> const& rhs_inn_modes,
  int out_rank,
  contraction_t::batching_t const& plan)
{
  // TODO: It is possible that the majority
  //       of the work could happen in nb and ni,
  //       which would be very goofy.
  // (note: this function just converts from batch matmul plan + info)

  contraction_t ret;
  ret.dtype = dtype;

  vector<uint64_t> lhs_shape;
  for(auto const& i: lhs_inn_modes) {
    lhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> rhs_shape;
  for(auto const& i: rhs_inn_modes) {
    rhs_shape.push_back(shape[i]);
  }
  vector<uint64_t> out_shape(shape.begin(), shape.begin() + out_rank);

  vector<int> out_modes(out_rank);
  std::iota(out_modes.begin(), out_modes.end(), 0);

  auto perm_lhs =
    permute_info_t::from_inn_shape(lhs_shape, lhs_inn_modes, plan.modes_lhs());
  auto perm_rhs =
    permute_info_t::from_inn_shape(rhs_shape, rhs_inn_modes, plan.modes_rhs());
  auto perm_out =
    permute_info_t::from_out_shape(out_shape, plan.modes_out(), out_modes    );

  int nb_plan = plan.bs.size();
  int ni_plan = plan.is.size();

  int max_leading_lhsout = nb_plan + (plan.lhs_t ? 0 : ni_plan);
  int max_leading_rhs = nb_plan;

  int leading_lhs = std::min(max_leading_lhsout, perm_lhs.num_leading_modes());
  int leading_rhs = std::min(max_leading_rhs,    perm_rhs.num_leading_modes());
  int leading_out = std::min(max_leading_lhsout, perm_out.num_leading_modes());

  int leading_b = std::min(leading_lhs, std::min(leading_rhs, leading_out));
  int leading_i;
  if(leading_b == nb_plan) {
    leading_i = std::min(leading_lhs, leading_out) - leading_b;
  } else {
    leading_i = 0;
  }

  ret.nb = 1;
  for(int i = 0; i != leading_b; ++i) {
    ret.nb *= shape[i];
  }
  ret.ni = 1;
  for(int i = leading_b; i != leading_b + leading_i; ++i) {
    ret.ni *= shape[lhs_inn_modes[i]];
  }

  auto dsz = dtype_size(dtype);

  ret.stride_lhs = dsz * (product(perm_lhs.inn_shape) / (ret.nb*ret.ni));
  ret.stride_rhs = dsz * (product(perm_rhs.inn_shape) /  ret.nb        );
  ret.stride_out = dsz * (product(perm_out.inn_shape) / (ret.nb*ret.ni));

  uint64_t ws = 0;
  ret.lhs_work_offset = ws;
  if(!perm_lhs.is_no_op()) {
    ret.lhs_p = perm_lhs.drop_leading(leading_b + leading_i);
    ws += dsz * product(ret.lhs_p.value().inn_shape);
  }

  ret.rhs_work_offset = ws;
  if(!perm_rhs.is_no_op()) {
    ret.rhs_p = perm_rhs.drop_leading(leading_b);
    ws += dsz * product(ret.rhs_p.value().inn_shape);
  }

  ret.out_work_offset = ws;
  if(!perm_out.is_no_op()) {
    ret.out_p = perm_out.drop_leading(leading_b + leading_i);
    ws += dsz * product(ret.out_p.value().inn_shape);
  }

  ret.workspace_size = ws;

  uint64_t total_b_elem = 1;
  for(auto const& x: plan.bs) { total_b_elem *= shape[x]; }
  uint64_t total_i_elem = 1;
  for(auto const& x: plan.is) { total_i_elem *= shape[x]; }
  uint64_t total_j_elem = 1;
  for(auto const& x: plan.js) { total_j_elem *= shape[x]; }
  uint64_t total_k_elem = 1;
  for(auto const& x: plan.ks) { total_k_elem *= shape[x]; }

  ret.inner.nb = total_b_elem / ret.nb ;
  ret.inner.ni = total_i_elem / ret.ni ;
  ret.inner.nj = total_j_elem          ;
  ret.inner.nk = total_k_elem          ;

  ret.inner.lhs_t = plan.lhs_t;
  ret.inner.rhs_t = plan.rhs_t;

  return ret;
}

contraction_t::permute_info_t
contraction_t::permute_info_t::drop_leading(int n) const {
  if(num_leading_modes() < n) {
    throw std::runtime_error("can't drop!");
  }

  vector<int> inn_modes(inn_shape.size() - n);
  std::iota(inn_modes.begin(), inn_modes.end(), n);

  vector<int> out_modes(out_perm.begin() + n, out_perm.end());

  return from_inn_shape(
    vector<uint64_t>(inn_shape.begin() + n, inn_shape.end()),
    inn_modes,
    out_modes);
}

int contraction_t::permute_info_t::num_leading_modes() const {
  int ret = 0;
  auto iter = out_perm.begin();
  while(iter != out_perm.end() && ret == *iter) {
    ret++;
    iter++;
  }
  return ret;
}

uint64_t
contraction_t::permute_info_t::leading_nelem() const
{
  return product(vector<uint64_t>(
    inn_shape.begin(),
    inn_shape.begin() + num_leading_modes()));
}


