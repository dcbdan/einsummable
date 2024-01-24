#include "kernel_executor.h"

#include <mkl_cblas.h>
#include <mkl.h>

#include "../../base/permute.h"

timetracker_t cpu_kernel_timetracker;

timetracker_t& get_cpu_kernel_timetracker() {
  return cpu_kernel_timetracker;
}

cpu_kernel_executor_t::cpu_kernel_executor_t()
{
  auto fix = einsummable_t::normalize_str;

  binfos = {
   { fix("ij,jk->ik"), { false,false, false,false,false } },
   { fix("ij,kj->ik"), { false, true, false,false,false } },
   { fix("ji,jk->ik"), {  true,false, false,false,false } },
   { fix("ji,kj->ik"), {  true, true, false,false,false } },

   { fix("bij,jk->ik"), { false,false, true,false,false } },
   { fix("bij,kj->ik"), { false, true, true,false,false } },
   { fix("bji,jk->ik"), {  true,false, true,false,false } },
   { fix("bji,kj->ik"), {  true, true, true,false,false } },

   { fix("ij,bjk->ik"), { false,false, false,true,false } },
   { fix("ij,bkj->ik"), { false, true, false,true,false } },
   { fix("ji,bjk->ik"), {  true,false, false,true,false } },
   { fix("ji,bkj->ik"), {  true, true, false,true,false } },

   { fix("bij,bjk->ik"), { false,false, true,true,false } },
   { fix("bij,bkj->ik"), { false, true, true,true,false } },
   { fix("bji,bjk->ik"), {  true,false, true,true,false } },
   { fix("bji,bkj->ik"), {  true, true, true,true,false } },

   { fix("bij,jk->bik"), { false,false, true,false,true } },
   { fix("bij,kj->bik"), { false, true, true,false,true } },
   { fix("bji,jk->bik"), {  true,false, true,false,true } },
   { fix("bji,kj->bik"), {  true, true, true,false,true } },

   { fix("ij,bjk->bik"), { false,false, false,true,true } },
   { fix("ij,bkj->bik"), { false, true, false,true,true } },
   { fix("ji,bjk->bik"), {  true,false, false,true,true } },
   { fix("ji,bkj->bik"), {  true, true, false,true,true } },

   { fix("bij,bjk->bik"), { false,false, true,true,true } },
   { fix("bij,bkj->bik"), { false, true, true,true,true } },
   { fix("bji,bjk->bik"), {  true,false, true,true,true } },
   { fix("bji,bkj->bik"), {  true, true, true,true,true } }
  };
}

optional<uint64_t> cpu_kernel_executor_t::build(einsummable_t const& e_)
{
  if(e_.join.has_variables()) {
    return std::nullopt;
  }

  auto einsummable = e_.merge_adjacent_dims();

  if(kernels.count(einsummable) > 0) {
    return workspace_size(einsummable);
  }

  if(einsummable.is_permutation()) {
    auto const& inn_modes = einsummable.inns[0];

    vector<int> out_modes(inn_modes.size());
    std::iota(out_modes.begin(), out_modes.end(), 0);

    kernels.insert({einsummable,
      tensor_permute_t {
        .dtype = einsummable.out_dtype(),
        .inn_shape = einsummable.inn_shapes()[0],
        .out_perm = as_out_perm(inn_modes, out_modes)
      }
    });
    return 0;
  }

  if(einsummable.is_straight_elementwise()) {
    int n = einsummable.inns.size();
    if(n == 1) {
      auto maybe = lookup_unary_straight_ew_kernel(einsummable.join);
      if(maybe) {
        auto const& [data, f] = maybe.value();
        kernels.insert({einsummable,
          unary_straight_ew_t {
            .n = product(einsummable.join_shape),
            .data = data,
            .f = f
          }
        });
        return 0;
      } else {
        return std::nullopt;
      }
    } else if(n == 2) {
      auto maybe = lookup_binary_straight_ew_kernel(einsummable.join);
      if(maybe) {
        auto const& [data, f] = maybe.value();
        kernels.insert({einsummable,
          binary_straight_ew_t {
            .n = product(einsummable.join_shape),
            .data = data,
            .f = f
          }
        });
        return 0;
      } else {
        return std::nullopt;
      }
    } else if(n == 3) {
      auto maybe = lookup_ternary_straight_ew_kernel(einsummable.join);
      if(maybe) {
        auto const& [data, f] = maybe.value();
        kernels.insert({einsummable,
          ternary_straight_ew_t {
            .n = product(einsummable.join_shape),
            .data = data,
            .f = f
          }
        });
        return 0;
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }

  auto estr = einsummable.str();

  if(estr == "ab,a->ab" || estr == "ab,b->ab" ||
     estr == "a,ab->ab" || estr == "b,ab->ab")
  {
    bool is_ab_a   = estr == "ab,a->ab" || estr == "a,ab->ab";
    bool swap_args = estr == "a,ab->ab" || estr == "b,ab->ab";
    auto maybe = lookup_binary_212_ew_kernel(einsummable.join, is_ab_a);
    if(maybe) {
      auto const& [data,f] = maybe.value();
      kernels.insert({einsummable,
        binary_212_ew_t {
          .na = einsummable.join_shape[0],
          .nb = einsummable.join_shape[1],
          .data = data,
          .f = f,
          .swapargs = swap_args
        }
      });
      return 0;
    } else {
      return std::nullopt;
    }
  }

  if(estr == "ab,a,a->ab") {
    auto maybe = lookup_ternary_2112_ew_kernel(einsummable.join);
    if(maybe) {
      auto const& [data,f] = maybe.value();
      kernels.insert({einsummable,
        ternary_2112_ew_t {
          .na = einsummable.join_shape[0],
          .nb = einsummable.join_shape[1],
          .data = data,
          .f = f
        }
      });
      return 0;
    } else {
      return std::nullopt;
    }
  }

  if(estr == "ab->a") {
    optional<scalar_t> maybe_scale = einsummable.join.get_scale_from_scale();
    if(maybe_scale && einsummable.castable.value() == castable_t::add) {
      scalar_t const& value = maybe_scale.value();
      if(dtype_is_complex(value.dtype)) {
        return std::nullopt;
      }
      kernels.insert({einsummable,
        sum_then_scale_ab_a_t {
          .na = einsummable.join_shape[0],
          .nb = einsummable.join_shape[1],
          .value = value
        }
      });
      return 0;
    }

    if(!einsummable.join.is_identity()) {
      return std::nullopt;
    }

    auto reduction = build_ab_a_reduction_kernel(
      einsummable.out_dtype(),
      einsummable.castable.value());

    kernels.insert({einsummable,
      reduction_ab_a_t {
        .na = einsummable.join_shape[0],
        .nb = einsummable.join_shape[1],
        .f = reduction
      }
    });
    return 0;
  }

  if(estr == "b->ab") {
    if(!einsummable.join.is_identity()) {
      return std::nullopt;
    }

    uint64_t dsz = dtype_size(einsummable.out_dtype());

    kernels.insert({einsummable,
      broadcast_b_ab_t {
        .nelem_a =       einsummable.join_shape[0],
        .sz_b    = dsz * einsummable.join_shape[1],
      }
    });
    return 0;
  }

  if(estr == "a->ab") {
    if(!einsummable.join.is_identity()) {
      return std::nullopt;
    }

    kernels.insert({einsummable,
      broadcast_a_ab_t {
        .dtype = einsummable.out_dtype(),
        .nelem_a = einsummable.join_shape[0],
        .nelem_b = einsummable.join_shape[1],
      }
    });
    return 0;
  }

  if(estr == "abcd,bd->abcd" && einsummable.join.out_dtype() == dtype_t::c64)
  {
    scalarop_t mulconj = scalarop_t::combine(
      scalarop_t::make_mul(dtype_t::c64),
      {
        scalarop_t::make_identity(dtype_t::c64),
        scalarop_t::make_conjugate(dtype_t::c64)
      });

    if(einsummable.join.is_mul())
    {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      uint64_t nc = einsummable.join_shape[2];
      uint64_t nd = einsummable.join_shape[3];
      kernel_t kernel = [na,nb,nc,nd](void* out, vector<void const*> inns) {
        using T = std::complex<float>;
        return c64_mul_abcd_bd_to_abcd(
          na,nb,nc,nd,
          reinterpret_cast<T*>(out),
          reinterpret_cast<T const*>(inns[0]),
          reinterpret_cast<T const*>(inns[1]));
      };
      kernels.insert({einsummable, kernel});
      return 0;
    } else if(einsummable.join == mulconj) {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      uint64_t nc = einsummable.join_shape[2];
      uint64_t nd = einsummable.join_shape[3];
      kernel_t kernel = [na,nb,nc,nd](void* out, vector<void const*> inns) {
        using T = std::complex<float>;
        return c64_mulconj_abcd_bd_to_abcd(
          na,nb,nc,nd,
          reinterpret_cast<T*>(out),
          reinterpret_cast<T const*>(inns[0]),
          reinterpret_cast<T const*>(inns[1]));
      };
      kernels.insert({einsummable, kernel});
      return 0;
    } else {
      return std::nullopt;
    }
  }

  if(estr == "ab,ab,a->a") {
    scalarop_t f = parse_with_ss<scalarop_t>(
      "*[hole|f32@0,*[hole|f32@1,*[constant{f32|-1},power{-2}[hole|f32@2]]]]");
    if(f == einsummable.join) {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      kernel_t kernel = [na,nb](void* out, vector<void const*> inns) {
        return custom01_float_ab_ab_a_to_a(
          na,nb,
          reinterpret_cast<float*>(out),
          reinterpret_cast<float const*>(inns[0]),
          reinterpret_cast<float const*>(inns[1]),
          reinterpret_cast<float const*>(inns[2]));
      };
      kernels.insert({einsummable, kernel});
      return 0;
    }
  }

  auto maybe_batch_matmul = make_batch_matmul(einsummable);
  if(maybe_batch_matmul) {
    kernels.insert({einsummable, maybe_batch_matmul.value()});
    return 0;
  }

  if(einsummable.is_contraction()) {
    if(!contraction_t::can_make(
      einsummable.inns[0],
      einsummable.inns[1],
      einsummable.out_rank))
    {
      return std::nullopt;
    }

    auto c = contraction_t::make(
      einsummable.out_dtype(),
      einsummable.join_shape,
      einsummable.inns[0],
      einsummable.inns[1],
      einsummable.out_rank);
    kernels.insert({einsummable, c});

    return c.workspace_size;
  }

  return std::nullopt;
}

// get the workspace size
// (throw an error if e has not been built)
uint64_t cpu_kernel_executor_t::workspace_size(einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  if(std::holds_alternative<contraction_t>(kernel)) {
    return std::get<contraction_t>(kernel).workspace_size;
  } else {
    return 0;
  }
}

string cpu_kernel_executor_t::as_str(einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  using std::holds_alternative;
  using std::get;
  if(holds_alternative<batch_matmul_t>(kernel)) {
    return "batch_matmul";
  } else if(holds_alternative<contraction_t>(kernel)) {
    return "contraction";
  } else if(holds_alternative<unary_straight_ew_t>(kernel)) {
    return "straight_uew";
  } else if(holds_alternative<binary_straight_ew_t>(kernel)) {
    return "straight_bew";
  } else if(holds_alternative<ternary_straight_ew_t>(kernel)) {
    return "straight_tew";
  } else if(holds_alternative<binary_212_ew_t>(kernel)) {
    return "b212";
  } else if(holds_alternative<ternary_2112_ew_t>(kernel)) {
    return "t2112";
  } else if(holds_alternative<tensor_permute_t>(kernel)) {
    return "permute";
  } else if(holds_alternative<reduction_ab_a_t>(kernel)) {
    return "red_ab_a";
  } else if(holds_alternative<sum_then_scale_ab_a_t>(kernel)) {
    return "sum_scl_ab_a";
  } else if(holds_alternative<broadcast_b_ab_t>(kernel)) {
    return "bro_b_ab";
  } else if(holds_alternative<broadcast_a_ab_t>(kernel)) {
    return "bro_a_ab";
  } else if(holds_alternative<kernel_t>(kernel)) {
    return "misc_kernel";
  } else {
    throw std::runtime_error("workspace size: kernel unaccounted for");
  }
}

vector<int> cpu_kernel_executor_t::donatables(einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  vector<int> maybe;
  maybe.reserve(2);

  if(std::holds_alternative<unary_straight_ew_t>(kernel)) {
    dtype_t inn = e.inn_dtype(0);
    maybe.push_back(0);
  } else if(std::holds_alternative<binary_straight_ew_t>(kernel)) {
    maybe.push_back(0);
    maybe.push_back(1);
  } else if(std::holds_alternative<ternary_straight_ew_t>(kernel)) {
    maybe.push_back(0);
    maybe.push_back(1);
    maybe.push_back(2);
  } else if(std::holds_alternative<binary_212_ew_t>(kernel)) {
    bool const& swapargs = std::get<binary_212_ew_t>(kernel).swapargs;
    if(swapargs) {
      maybe.push_back(1);
    } else {
      maybe.push_back(0);
    }
  } else if(std::holds_alternative<ternary_2112_ew_t>(kernel)) {
    maybe.push_back(0);
  }

  dtype_t out_dtype = e.out_dtype();

  // Things will end badly if the input dtype
  // does not match the output dtype. For instance, if
  // e = "ijk->ijk" and converts from f32 to f64, then
  // donating the first input will only have half
  // as much memory as required.

  vector<int> ret;
  ret.reserve(maybe.size());
  for(int const& inn: maybe) {
    if(e.inn_dtype(inn) == out_dtype) {
      ret.push_back(inn);
    }
  }

  return ret;
}

void cpu_kernel_executor_t::operator()(
  touch_t const& touch,
  void* out,
  void const* inn) const
{
  //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("touch");
  execute_touch(touch.simplify(), out, inn);
}

void cpu_kernel_executor_t::operator()(
  einsummable_t const& e,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:total");
  auto const& info = get_built_kernel_info(e);
  call(info, out, inns, maybe_workspace);
}

void cpu_kernel_executor_t::operator()(
  einsummable_t const& e,
  void* out,
  vector<void const*> inns,
  optional<buffer_t> maybe_workspace) const
{
  if(maybe_workspace) {
    buffer_t& b = maybe_workspace.value();
    return this->operator()(e, out, inns, tuple<void*, uint64_t>{ b->raw(), b->size });
  } else {
    return this->operator()(e, out, inns, optional<tuple<void*, uint64_t>>());
  }
}

void cpu_kernel_executor_t::call(
  cpu_kernel_executor_t::kernel_info_t const& kernel,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace)
{
  using std::holds_alternative;
  using std::get;

  auto assert_num_inputs = [&inns](int n) {
    if(inns.size() != n) {
      throw std::runtime_error("kernel manager: incorrect number of input tensors");
    }
  };

  if(holds_alternative<batch_matmul_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:batch_matmul");
    assert_num_inputs(2);
    auto const& b = get<batch_matmul_t>(kernel);
    batch_matrix_multiply(
      b.dtype,
      b.nb,
      b.info.batched_out, b.info.batched_lhs, b.info.batched_rhs,
      b.ni, b.nj, b.nk,
      b.info.trans_lhs, b.info.trans_rhs,
      out, inns[0], inns[1]);
  } else if(holds_alternative<contraction_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:contraction");
    assert_num_inputs(2);
    auto const& c = get<contraction_t>(kernel);
    if(c.workspace_size == 0) {
      c(nullptr, out, inns[0], inns[1]);
    } else if(!maybe_workspace) {
      throw std::runtime_error("workspace required; none given");
    } else {
      auto [workspace, wsz] = maybe_workspace.value();
      if(wsz < c.workspace_size) {
        throw std::runtime_error("provided workspace is too small");
      }
      c(workspace, out, inns[0], inns[1]);
    }
  } else if(holds_alternative<unary_straight_ew_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:uew");
    assert_num_inputs(1);
    auto const& [n,data,f] = get<unary_straight_ew_t>(kernel);
    f(data.data(), n, out, inns[0]);
  } else if(holds_alternative<binary_straight_ew_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:bew");
    assert_num_inputs(2);
    auto const& [n,data,f] = get<binary_straight_ew_t>(kernel);
    f(data.data(), n, out, inns[0], inns[1]);
  } else if(holds_alternative<ternary_straight_ew_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:tew");
    assert_num_inputs(3);
    auto const& [n,data,f] = get<ternary_straight_ew_t>(kernel);
    f(data.data(), n, out, inns[0], inns[1], inns[2]);
  } else if(holds_alternative<binary_212_ew_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:b212");
    assert_num_inputs(2);
    auto const& [na,nb,data,f,swapargs] = get<binary_212_ew_t>(kernel);
    if(swapargs) {
      f(data.data(), na, nb, out, inns[1], inns[0]);
    } else {
      f(data.data(), na, nb, out, inns[0], inns[1]);
    }
  } else if(holds_alternative<ternary_2112_ew_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:b212");
    assert_num_inputs(3);
    auto const& [na,nb,data,f] = get<ternary_2112_ew_t>(kernel);
    f(data.data(), na, nb, out, inns[0], inns[1], inns[2]);
  } else if(holds_alternative<tensor_permute_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:tensor_permute");
    assert_num_inputs(1);
    auto const& [dtype, inn_shape, out_perm] = get<tensor_permute_t>(kernel);
    permute_kernel(dtype, 1024, inn_shape, out_perm, out, inns[0]);
  } else if(holds_alternative<reduction_ab_a_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:red_ab_a");
    assert_num_inputs(1);
    auto const& [na,nb,f] = get<reduction_ab_a_t>(kernel);
    f(na,nb,out,inns[0]);
  } else if(holds_alternative<sum_then_scale_ab_a_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:red_ab_a");
    assert_num_inputs(1);
    auto const& [na,nb,val] = get<sum_then_scale_ab_a_t>(kernel);
    sum_then_scale_ab_a_kernel(val,na,nb,out,inns[0]);
  } else if(holds_alternative<broadcast_b_ab_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:bro_b_ab");
    assert_num_inputs(1);
    auto const& [nelem_a,sz_b] = get<broadcast_b_ab_t>(kernel);
    broadcast_b_ab_kernel(nelem_a, sz_b, out, inns[0]);
  } else if(holds_alternative<broadcast_a_ab_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:bro_a_ab");
    assert_num_inputs(1);
    auto const& [dtype,nelem_a,nelem_b] = get<broadcast_a_ab_t>(kernel);
    broadcast_a_ab_kernel(dtype, nelem_a, nelem_b, out, inns[0]);
  } else if(holds_alternative<kernel_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:misc");
    auto const& f = get<kernel_t>(kernel);
    f(out, inns);
  } else {
    throw std::runtime_error("workspace size: kernel unaccounted for");
  }
}

cpu_kernel_executor_t::kernel_info_t const&
cpu_kernel_executor_t::get_built_kernel_info(einsummable_t const& e) const
{
  auto iter = kernels.find(e.merge_adjacent_dims());
  if(iter == kernels.end()) {
    throw std::runtime_error("get_built_kernel_info: this einsummable has not been built");
  }
  return iter->second;
}

optional<cpu_kernel_executor_t::batch_matmul_t>
cpu_kernel_executor_t::make_batch_matmul(einsummable_t const& e)
{
  if(!e.is_contraction()) {
    return std::nullopt;
  }
  auto iter = binfos.find(e.str());
  if(iter == binfos.end()) {
    return std::nullopt;
  }
  auto const& info = iter->second;

  auto inn_shapes = e.inn_shapes();
  auto const& lhs = inn_shapes[0];
  auto const& rhs = inn_shapes[1];
  auto out = e.out_shape();

  uint64_t nb = 0;
  uint64_t ni = 0;
  uint64_t nj = 0;
  uint64_t nk = 0;

  if(info.batched_lhs) {
    nb = lhs[0];
    if(info.trans_lhs) {
      // bji
      nj = lhs[1];
      ni = lhs[2];
    } else {
      // bij
      ni = lhs[1];
      nj = lhs[2];
    }
  } else {
    if(info.trans_lhs) {
      // ji
      nj = lhs[0];
      ni = lhs[1];
    } else {
      // ij
      ni = lhs[0];
      nj = lhs[1];
    }
  }

  if(info.batched_rhs) {
    nb = rhs[0];
    if(info.trans_rhs) {
      // bkj
      nk = rhs[1];
      nj = rhs[2];
    } else {
      // bjk
      nj = rhs[1];
      nk = rhs[2];
    }
  } else {
    nb = 1;
    if(info.trans_rhs) {
      // kj
      nk = rhs[0];
      nj = rhs[1];
    } else {
      // jk
      nj = rhs[0];
      nk = rhs[1];
    }
  }

  if(nb == 0 || ni == 0 || nj == 0 || nk == 0) {
    throw std::runtime_error("all sizes should be set");
  }

  return batch_matmul_t {
    .dtype = e.out_dtype(),
    .info = info,
    .nb = nb,
    .ni = ni,
    .nj = nj,
    .nk = nk
  };
}

std::function<void(void*, vector<void const*>)>
build_einsummable(einsummable_t const& e)
{
  cpu_kernel_executor_t ks;
  auto maybe = ks.build(e);
  if(!maybe) {
    throw std::runtime_error("could not build the kernel");
  }
  if(maybe.value() != 0) {
    throw std::runtime_error("build_einsummable: this kernel requires a workspace");
  }
  auto const& meta_info = ks.get_built_kernel_info(e);
  return [meta_info](void* out, vector<void const*> inns) {
    cpu_kernel_executor_t::call(meta_info, out, inns);
  };
}

template <typename T>
inline T _pow(T const& v, double const& power) {
  return std::pow(v, power);
}

template <>
inline float16_t _pow(float16_t const& v, double const& power) {
  return half_float::pow(v, float16_t(power));
}

template <typename T>
inline T _exp(T const& v) {
  return std::exp(v);
}

template <>
inline float16_t _exp(float16_t const& v) {
  return half_float::exp(v);
}

template <typename T>
inline T _log(T const& v) {
  return std::log(v);
}

template <>
inline float16_t _log(float16_t const& v) {
  return half_float::log(v);
}

template <typename T>
inline T _conj(T const& v) {
  return std::conj(v);
}

template <typename T>
inline T _real(std::complex<T> const& v) {
  return std::real(v);
}

template <typename T>
inline T _imag(std::complex<T> const& v) {
  return std::imag(v);
}

template <typename T>
inline std::complex<T> _complex(T const& x, T const& y) {
  return std::complex<T>(x,y);
}

#define _unary_ew_loop(name, TO, T, op) \
  void name( \
    uint8_t const* d, \
    uint64_t n, \
    void* _out, \
    void const* _x0) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T const* x0 = reinterpret_cast<T const*>(_x0); \
    for(uint64_t i = 0; i != n; ++i) { \
      out[i] = op; \
    } \
  }

// a,a->a
// ab,a->ab
// ab,b->ab
#define _binary_ew_loop(name1, name2, name3, TO, T0, T1, op) \
  void name1( \
    uint8_t const* d, \
    uint64_t n, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T0 const* x0 = reinterpret_cast<T0 const*>(_x0); \
    T1 const* x1 = reinterpret_cast<T1 const*>(_x1); \
    for(uint64_t _i = 0; _i != n; ++_i) { \
      uint64_t const& iO = _i; \
      uint64_t const& i0 = _i; \
      uint64_t const& i1 = _i; \
      out[iO] = op; \
    } \
  } \
  void name2( \
    uint8_t const* d, \
    uint64_t n1, \
    uint64_t n2, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T0 const* x0 = reinterpret_cast<T0 const*>(_x0); \
    T1 const* x1 = reinterpret_cast<T1 const*>(_x1); \
    for(uint64_t _i = 0; _i != n1; ++_i) { \
    for(uint64_t _j = 0; _j != n2; ++_j) { \
      uint64_t        iO = _i*n2 + _j; \
      uint64_t const& i0 = iO; \
      uint64_t const& i1 = _i; \
      out[iO] = op; \
    }} \
  } \
  void name3( \
    uint8_t const* d, \
    uint64_t n1, \
    uint64_t n2, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T0 const* x0 = reinterpret_cast<T0 const*>(_x0); \
    T1 const* x1 = reinterpret_cast<T1 const*>(_x1); \
    for(uint64_t _i = 0; _i != n1; ++_i) { \
    for(uint64_t _j = 0; _j != n2; ++_j) { \
      uint64_t        iO = _i*n2 + _j; \
      uint64_t const& i0 = iO; \
      uint64_t const& i1 = _j; \
      out[iO] = op; \
    }} \
  }

// a,a,a->a
// ab,a,a->ab
#define _ternary_ew_loop(name1, name2, TO, T0, T1, T2, op) \
  void name1( \
    uint8_t const* d, \
    uint64_t n, \
    void* _out, \
    void const* _x0, \
    void const* _x1, \
    void const* _x2) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T0 const* x0 = reinterpret_cast<T0 const*>(_x0); \
    T1 const* x1 = reinterpret_cast<T1 const*>(_x1); \
    T2 const* x2 = reinterpret_cast<T2 const*>(_x2); \
    for(uint64_t i = 0; i != n; ++i) { \
      uint64_t const& iO = i; \
      uint64_t const& i0 = i; \
      uint64_t const& i1 = i; \
      uint64_t const& i2 = i; \
      out[iO] = op; \
    } \
  } \
  void name2( \
    uint8_t const* d, \
    uint64_t n1, \
    uint64_t n2, \
    void* _out, \
    void const* _x0, \
    void const* _x1, \
    void const* _x2) \
  { \
    TO* out     = reinterpret_cast<TO*>(_out); \
    T0 const* x0 = reinterpret_cast<T0 const*>(_x0); \
    T1 const* x1 = reinterpret_cast<T1 const*>(_x1); \
    T2 const* x2 = reinterpret_cast<T2 const*>(_x2); \
    for(uint64_t _i = 0; _i != n1; ++_i) { \
    for(uint64_t _j = 0; _j != n2; ++_j) { \
      uint64_t        iO = _i*n2 + _j; \
      uint64_t const& i0 = iO; \
      uint64_t const& i1 = _i; \
      uint64_t const& i2 = _i; \
      out[iO] = op; \
    }} \
  }

_unary_ew_loop(u0,float,float,((*((float*)(d+0)))>=x0[i]?(*((float*)(d+4))):x0[i]))
_unary_ew_loop(u1,float16_t,float16_t,((*((float16_t*)(d+0)))>=x0[i]?(*((float16_t*)(d+2))):x0[i]))
_unary_ew_loop(u2,double,double,((*((double*)(d+0)))>=x0[i]?(*((double*)(d+8))):x0[i]))
_unary_ew_loop(u3,float16_t,float16_t,(x0[i]*_pow(((*((float16_t*)(d+0)))+_exp(((*((float16_t*)(d+2)))*x0[i]))),(*((double*)(d+4))))))
_unary_ew_loop(u4,float,float,_exp(x0[i]))
_unary_ew_loop(u5,float,float,_pow(x0[i],(*((double*)(d+0)))))
_unary_ew_loop(u6,float,float,((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i])))
_unary_ew_loop(u7,float,float,_pow(x0[i],(*((double*)(d+0)))))
_unary_ew_loop(u8,float16_t,float,float16_t(x0[i]))
_unary_ew_loop(u9,float16_t,float16_t,((*((float16_t*)(d+0)))*x0[i]))
_unary_ew_loop(u10,float,float16_t,float(x0[i]))
_unary_ew_loop(u12,double,double,(x0[i]*_pow(((*((double*)(d+0)))+_exp(((*((double*)(d+8)))*x0[i]))),(*((double*)(d+16))))))
_unary_ew_loop(u13,double,double,((*((double*)(d+0)))+x0[i]))
_unary_ew_loop(u14,double,double,((*((double*)(d+0)))*x0[i]))
_unary_ew_loop(u15,float,float,(x0[i]*_pow(((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x0[i]))),(*((double*)(d+8))))))
_unary_ew_loop(u16,float,float,((*((float*)(d+0)))+x0[i]))
_unary_ew_loop(u17,float,float,((*((float*)(d+0)))*x0[i]))
_unary_ew_loop(u18,float,double,float(x0[i]))
_unary_ew_loop(u19,double,float,double(x0[i]))
_unary_ew_loop(u20,float16_t,double,float16_t(x0[i]))
_unary_ew_loop(u21,double,float16_t,double(x0[i]))
_unary_ew_loop(u22,double,double,_pow(x0[i],(*((double*)(d+0)))))
_unary_ew_loop(u23,float,float,((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i])))
_unary_ew_loop(u24,double,double,((*((double*)(d+0)))+((*((double*)(d+8)))*x0[i])))
_unary_ew_loop(u25,double,double,_exp(x0[i]))
_unary_ew_loop(u26,float,float,((*((float*)(d+0)))+_pow(x0[i],(*((double*)(d+4))))))
_unary_ew_loop(u27,float,float,_pow(_log(x0[i]),(*((double*)(d+0)))))
_unary_ew_loop(u28,float,float,_log(x0[i]))

_binary_ew_loop(b0,c0,d0,float,float,float,_pow((x0[i0]+((*((float*)(d+0)))*x1[i1])),(*((double*)(d+4)))))
_binary_ew_loop(b1,c1,d1,float,float,float,((*((float*)(d+0)))*(x0[i0]+((*((float*)(d+4)))*x1[i1]))))
_binary_ew_loop(b2,c2,d2,float,float,float,(x0[i0]*x1[i1]))
_binary_ew_loop(b3,c3,d3,float,float,float,(((*((float*)(d+0)))>=x0[i0]?(*((float*)(d+4))):(*((float*)(d+8))))*x1[i1]))
_binary_ew_loop(b4,c4,d4,float,float,float,(x0[i0]+((*((float*)(d+0)))*((*((float*)(d+4)))*x1[i1]))))
_binary_ew_loop(b5,c5,d5,float16_t,float16_t,float16_t,_pow((x0[i0]+((*((float16_t*)(d+0)))*x1[i1])),(*((double*)(d+2)))))
_binary_ew_loop(b6,c6,d6,float16_t,float16_t,float16_t,((*((float16_t*)(d+0)))*(x0[i0]+((*((float16_t*)(d+2)))*x1[i1]))))
_binary_ew_loop(b7,c7,d7,float16_t,float16_t,float16_t,(x0[i0]*x1[i1]))
_binary_ew_loop(b8,c8,d8,float16_t,float16_t,float16_t,(((*((float16_t*)(d+0)))>=x0[i0]?(*((float16_t*)(d+2))):(*((float16_t*)(d+4))))*x1[i1]))
_binary_ew_loop(b9,c9,d9,float16_t,float16_t,float16_t,(x0[i0]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i1]))))
_binary_ew_loop(b10,c10,d10,float16_t,float16_t,float16_t,(x0[i0]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i1]))))
_binary_ew_loop(b11,c11,d11,double,double,double,_pow((x0[i0]+((*((double*)(d+0)))*x1[i1])),(*((double*)(d+8)))))
_binary_ew_loop(b12,c12,d12,double,double,double,((*((double*)(d+0)))*(x0[i0]+((*((double*)(d+8)))*x1[i1]))))
_binary_ew_loop(b13,c13,d13,double,double,double,(x0[i0]*x1[i1]))
_binary_ew_loop(b14,c14,d14,double,double,double,(((*((double*)(d+0)))>=x0[i0]?(*((double*)(d+8))):(*((double*)(d+16))))*x1[i1]))
_binary_ew_loop(b15,c15,d15,double,double,double,(x0[i0]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i1]))))
_binary_ew_loop(b16,c16,d16,double,double,double,(x0[i0]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i1]))))
_binary_ew_loop(b17,c17,d17,float,float,float,(x0[i0]*_pow(x1[i1],(*((double*)(d+0))))))
_binary_ew_loop(b18,c18,d18,float16_t,float16_t,float16_t,(x0[i0]+x1[i1]))
_binary_ew_loop(b19,c19,d19,float,float,float,(x0[i0]+x1[i1]))
_binary_ew_loop(b20,c20,d20,float16_t,float16_t,float16_t,(x0[i0]<x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b21,c21,d21,float16_t,float16_t,float16_t,(x0[i0]>x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b22,c22,d22,float,float,float,(x0[i0]<x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b23,c23,d23,float,float,float,(x0[i0]>x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b24,c24,d24,double,double,double,(x0[i0]+x1[i1]))
_binary_ew_loop(b25,c25,d25,double,double,double,(x0[i0]<x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b26,c26,d26,double,double,double,(x0[i0]>x1[i1]?x0[i0]:x1[i1]))
_binary_ew_loop(b27,c27,d27,float16_t,float16_t,float16_t,(x0[i0]+((*((float16_t*)(d+0)))*x1[i1])))
_binary_ew_loop(b28,c28,d28,float,float,float,(x0[i0]+((*((float*)(d+0)))*x1[i1])))
_binary_ew_loop(b29,c29,d29,double,double,double,(x0[i0]+((*((double*)(d+0)))*x1[i1])))
_binary_ew_loop(b30,c30,d30,double,double,double,(x0[i0]*_pow(x1[i1],(*((double*)(d+0))))))
_binary_ew_loop(b31,c31,d31,float,float,float,(x0[i0]*((*((float*)(d+0)))>=x1[i1]?(*((float*)(d+4))):(*((float*)(d+8))))))
_binary_ew_loop(b32,c32,d32,double,double,double,(x0[i0]*((*((double*)(d+0)))>=x1[i1]?(*((double*)(d+4))):(*((double*)(d+8))))))
_binary_ew_loop(b33,c33,d33,float,float,float,((*((float*)(d+0)))*((*((float*)(d+4)))*(x0[i0]+((*((float*)(d+8)))*x1[i1])))))
_binary_ew_loop(b34,c34,d34,double,double,double,((*((double*)(d+0)))*((*((double*)(d+4)))*(x0[i0]+((*((double*)(d+8)))*x1[i1])))))
_binary_ew_loop(b35,c35,d35,float,float,float,(((*((float*)(d+0)))*x0[i0])+((*((float*)(d+4)))*x1[i1])))
_binary_ew_loop(b36,c36,d36,float,float,float,(((*((float*)(d+0)))*x0[i0])+((*((float*)(d+4)))*_pow(x1[i1],(*((double*)(d+8)))))))
_binary_ew_loop(b37,c37,d37,float,float,float,(x0[i0]==x1[i1]?(*((float*)(d+0))):(*((float*)(d+4)))))
_binary_ew_loop(b38,c38,d38,float,float,float,(x0[i0]*_exp(x1[i1])))
_binary_ew_loop(b39,c39,d39,float,float,float,(x0[i0]*((*((float*)(d+0)))*_pow(x1[i1],(*((double*)(d+4)))))))
_binary_ew_loop(b40,c40,d40,float,float,float,(x0[i0]*((*((float*)(d+0)))*x1[i1])))
_binary_ew_loop(b41,c41,d41,float,float,float,(x0[i0]*(_pow(((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x1[i1]))),(*((double*)(d+8))))+(x1[i1]*((*((float*)(d+16)))*(_pow(((*((float*)(d+20)))+_exp(((*((float*)(d+24)))*x1[i1]))),(*((double*)(d+28))))*((*((float*)(d+36)))*_exp(((*((float*)(d+40)))*x1[i1])))))))))
_binary_ew_loop(b42,c42,d42,float,float,float,(x0[i0]*((*((float*)(d+0)))*_pow(x1[i1],(*((double*)(d+4)))))))
_binary_ew_loop(b43,c43,d43,float,float,float,(x0[i0]*((*((float*)(d+0)))*x1[i1])))

_ternary_ew_loop(tstraight_0,t2112_0,float,float,float,float,(x0[i0]*(x1[i1]*((*((float*)(d+0)))*_pow(x2[i2],(*((double*)(d+4))))))))
_ternary_ew_loop(tstraight_1,t2112_1,float,float,float,float,((x0[i0]*_pow(x1[i1],(*((double*)(d+0)))))*x2[i2]))

optional<
  tuple<vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, void*, void const*)> >
lookup_unary_straight_ew_kernel(scalarop_t op)
{
  using kernel_t = void(*)(uint8_t const*, uint64_t, void*, void const*);

  // TODO: this shouldn't have to happen as op should always be simplified
  //       to a unique value. For some reason
  //       a kernel wasn't normalized in the same way as the key
  //       requires...
  op = op.simplify();

  auto [op_str, bytes] = op.to_cpp_bytes();
  string key = op.type_signature() + "|" + op_str;

  static map<string, kernel_t> kernels = {
    { "f32->f32|((*((float*)(d+0)))>=x0[i]?(*((float*)(d+4))):x0[i])", u0 },
    { "f16->f16|((*((float16_t*)(d+0)))>=x0[i]?(*((float16_t*)(d+2))):x0[i])", u1 },
    { "f64->f64|((*((double*)(d+0)))>=x0[i]?(*((double*)(d+8))):x0[i])", u2 },
    { "f16->f16|(x0[i]*_pow(((*((float16_t*)(d+0)))+_exp(((*((float16_t*)(d+2)))*x0[i]))),(*((double*)(d+4)))))", u3 },
    { "f32->f32|_exp(x0[i])", u4 },
    { "f32->f32|_pow(x0[i],(*((double*)(d+0))))", u5 },
    { "f32->f32|((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i]))", u6 },
    { "f32->f32|_pow(x0[i],(*((double*)(d+0))))", u7 },
    { "f32->f16|float16_t(x0[i])", u8 },
    { "f16->f16|((*((float16_t*)(d+0)))*x0[i])", u9 },
    { "f16->f32|float(x0[i])", u10 },
    { "f64->f64|(x0[i]*_pow(((*((double*)(d+0)))+_exp(((*((double*)(d+8)))*x0[i]))),(*((double*)(d+16)))))", u12 },
    { "f64->f64|((*((double*)(d+0)))+x0[i])", u13 },
    { "f64->f64|((*((double*)(d+0)))*x0[i])", u14 },
    { "f32->f32|(x0[i]*_pow(((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x0[i]))),(*((double*)(d+8)))))", u15 },
    { "f32->f32|((*((float*)(d+0)))+x0[i])", u16 },
    { "f32->f32|((*((float*)(d+0)))*x0[i])", u17 },
    { "f64->f32|float(x0[i])", u18 },
    { "f32->f64|double(x0[i])", u19 },
    { "f64->f16|float16_t(x0[i])", u20 },
    { "f16->f64|double(x0[i])", u21 },
    { "f64->f64|_pow(x0[i],(*((double*)(d+0))))", u22 },
    { "f32->f32|((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i]))", u23 },
    { "f64->f64|((*((double*)(d+0)))+((*((double*)(d+8)))*x0[i]))", u24 },
    { "f64->f64|_exp(x0[i])", u25 },
    { "f32->f32|((*((float*)(d+0)))+_pow(x0[i],(*((double*)(d+4)))))", u26 },
    { "f32->f32|_pow(_log(x0[i]),(*((double*)(d+0))))", u27 },
    { "f32->f32|_log(x0[i])", u28 }
  };

  auto iter = kernels.find(key);
  if(iter == kernels.end()) {
    return std::nullopt;
  }
  using tt = tuple<vector<uint8_t>, kernel_t>;
  return tt{bytes, iter->second};
}

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, void*, void const*, void const*)> >
lookup_binary_straight_ew_kernel(
  scalarop_t op)
{
  // TODO: this shouldn't have to happen as op should always be simplified
  //       to a unique value. For some reason
  //       a kernel wasn't normalized in the same way as the key
  //       requires...
  op = op.simplify();

  auto [op_str, bytes] = op.to_cpp_bytes();
  string key = op.type_signature() + "|" + op_str;

  using kernel_t =
    void(*)(uint8_t const*, uint64_t, void*, void const*, void const*);

  static map<string, kernel_t> kernels = {
    { "f32,f32->f32|_pow((x0[i]+((*((float*)(d+0)))*x1[i])),(*((double*)(d+4))))", b0 },
    { "f32,f32->f32|((*((float*)(d+0)))*(x0[i]+((*((float*)(d+4)))*x1[i])))", b1 },
    { "f32,f32->f32|(x0[i]*x1[i])", b2 },
    { "f32,f32->f32|(((*((float*)(d+0)))>=x0[i]?(*((float*)(d+4))):(*((float*)(d+8))))*x1[i])", b3 },
    { "f32,f32->f32|(x0[i]+((*((float*)(d+0)))*((*((float*)(d+4)))*x1[i])))", b4 },
    { "f16,f16->f16|_pow((x0[i]+((*((float16_t*)(d+0)))*x1[i])),(*((double*)(d+2))))", b5 },
    { "f16,f16->f16|((*((float16_t*)(d+0)))*(x0[i]+((*((float16_t*)(d+2)))*x1[i])))", b6 },
    { "f16,f16->f16|(x0[i]*x1[i])", b7 },
    { "f16,f16->f16|(((*((float16_t*)(d+0)))>=x0[i]?(*((float16_t*)(d+2))):(*((float16_t*)(d+4))))*x1[i])", b8 },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i])))", b9 },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i])))", b10 },
    { "f64,f64->f64|_pow((x0[i]+((*((double*)(d+0)))*x1[i])),(*((double*)(d+8))))", b11 },
    { "f64,f64->f64|((*((double*)(d+0)))*(x0[i]+((*((double*)(d+8)))*x1[i])))", b12 },
    { "f64,f64->f64|(x0[i]*x1[i])", b13 },
    { "f64,f64->f64|(((*((double*)(d+0)))>=x0[i]?(*((double*)(d+8))):(*((double*)(d+16))))*x1[i])", b14 },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i])))", b15 },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i])))", b16 },
    { "f32,f32->f32|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", b17 },
    { "f16,f16->f16|(x0[i]+x1[i])", b18 },
    { "f32,f32->f32|(x0[i]+x1[i])", b19 },
    { "f16,f16->f16|(x0[i]<x1[i]?x0[i]:x1[i])", b20 },
    { "f16,f16->f16|(x0[i]>x1[i]?x0[i]:x1[i])", b21 },
    { "f32,f32->f32|(x0[i]<x1[i]?x0[i]:x1[i])", b22 },
    { "f32,f32->f32|(x0[i]>x1[i]?x0[i]:x1[i])", b23 },
    { "f64,f64->f64|(x0[i]+x1[i])", b24 },
    { "f64,f64->f64|(x0[i]<x1[i]?x0[i]:x1[i])", b25 },
    { "f64,f64->f64|(x0[i]>x1[i]?x0[i]:x1[i])", b26 },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*x1[i]))", b27 },
    { "f32,f32->f32|(x0[i]+((*((float*)(d+0)))*x1[i]))", b28 },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*x1[i]))", b29 },
    { "f64,f64->f64|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", b30 },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))>=x1[i]?(*((float*)(d+4))):(*((float*)(d+8)))))", b31 },
    { "f64,f64->f64|(x0[i]*((*((double*)(d+0)))>=x1[i]?(*((double*)(d+4))):(*((double*)(d+8)))))", b32 },
    { "f32,f32->f32|((*((float*)(d+0)))*((*((float*)(d+4)))*(x0[i]+((*((float*)(d+8)))*x1[i]))))", b33 },
    { "f32,f32->f32|((*((double*)(d+0)))*((*((double*)(d+4)))*(x0[i]+((*((double*)(d+8)))*x1[i]))))", b34 },
    { "f32,f32->f32|(((*((float*)(d+0)))*x0[i])+((*((float*)(d+4)))*x1[i]))", b35 },
    { "f32,f32->f32|(((*((float*)(d+0)))*x0[i])+((*((float*)(d+4)))*_pow(x1[i],(*((double*)(d+8))))))", b36 },
    { "f32,f32->f32|(x0[i]==x1[i]?(*((float*)(d+0))):(*((float*)(d+4))))", b37 },
    { "f32,f32->f32|(x0[i]*_exp(x1[i]))", b38 },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*_pow(x1[i],(*((double*)(d+4))))))", b39 },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*x1[i]))", b40 },
    { "f32,f32->f32|(x0[i]*(_pow(((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x1[i]))),(*((double*)(d+8))))+(x1[i]*((*((float*)(d+16)))*(_pow(((*((float*)(d+20)))+_exp(((*((float*)(d+24)))*x1[i]))),(*((double*)(d+28))))*((*((float*)(d+36)))*_exp(((*((float*)(d+40)))*x1[i]))))))))", b41 },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*_pow(x1[i],(*((double*)(d+4))))))", b42 },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*x1[i]))", b43 }
  };

  auto iter = kernels.find(key);
  if(iter == kernels.end()) {
    return std::nullopt;
  }
  using tt = tuple<vector<uint8_t>, kernel_t>;
  return optional<tt>(tt{bytes, iter->second});
}

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, void*, void const*, void const*, void const*)> >
lookup_ternary_straight_ew_kernel(
  scalarop_t op)
{
  // TODO: this shouldn't have to happen as op should always be simplified
  //       to a unique value. For some reason
  //       a kernel wasn't normalized in the same way as the key
  //       requires...
  op = op.simplify();

  auto [op_str, bytes] = op.to_cpp_bytes();
  string key = op.type_signature() + "|" + op_str;

  using kernel_t =
    void(*)(uint8_t const*, uint64_t, void*, void const*, void const*, void const*);

  static map<string, kernel_t> kernels = {
    { "f32,f32,f32->f32|(x0[i]*(x1[i]*((*((float*)(d+0)))*_pow(x2[i],(*((double*)(d+4)))))))", tstraight_0 },
    { "f32,f32,f32->f32|((x0[i]*_pow(x1[i],(*((double*)(d+0)))))*x2[i])", tstraight_1 }
  };

  auto iter = kernels.find(key);
  if(iter == kernels.end()) {
    return std::nullopt;
  }
  using tt = tuple<vector<uint8_t>, kernel_t>;
  return optional<tt>(tt{bytes, iter->second});
}

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*)> >
lookup_binary_212_ew_kernel(
  scalarop_t op,
  bool is_ab_a)
{
  // TODO: this shouldn't have to happen as op should always be simplified
  //       to a unique value. For some reason
  //       a kernel wasn't normalized in the same way as the key
  //       requires...
  op = op.simplify();

  auto [op_str, bytes] = op.to_cpp_bytes();
  string key = op.type_signature() + "|" + op_str;

  using kernel_t =
    void(*)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*);

  static map< string, tuple<kernel_t, kernel_t> > kernels = {
    { "f32,f32->f32|_pow((x0[i]+((*((float*)(d+0)))*x1[i])),(*((double*)(d+4))))", { c0, d0} },
    { "f32,f32->f32|((*((float*)(d+0)))*(x0[i]+((*((float*)(d+4)))*x1[i])))", { c1, d1} },
    { "f32,f32->f32|(x0[i]*x1[i])", { c2, d2} },
    { "f32,f32->f32|(((*((float*)(d+0)))>=x0[i]?(*((float*)(d+4))):(*((float*)(d+8))))*x1[i])", { c3, d3} },
    { "f32,f32->f32|(x0[i]+((*((float*)(d+0)))*((*((float*)(d+4)))*x1[i])))", { c4, d4} },
    { "f16,f16->f16|_pow((x0[i]+((*((float16_t*)(d+0)))*x1[i])),(*((double*)(d+2))))", { c5, d5} },
    { "f16,f16->f16|((*((float16_t*)(d+0)))*(x0[i]+((*((float16_t*)(d+2)))*x1[i])))", { c6, d6} },
    { "f16,f16->f16|(x0[i]*x1[i])", { c7, d7} },
    { "f16,f16->f16|(((*((float16_t*)(d+0)))>=x0[i]?(*((float16_t*)(d+2))):(*((float16_t*)(d+4))))*x1[i])", { c8, d8} },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i])))", { c9, d9} },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*((*((float16_t*)(d+2)))*x1[i])))", { c10, d10} },
    { "f64,f64->f64|_pow((x0[i]+((*((double*)(d+0)))*x1[i])),(*((double*)(d+8))))", { c11, d11} },
    { "f64,f64->f64|((*((double*)(d+0)))*(x0[i]+((*((double*)(d+8)))*x1[i])))", { c12, d12} },
    { "f64,f64->f64|(x0[i]*x1[i])", { c13, d13} },
    { "f64,f64->f64|(((*((double*)(d+0)))>=x0[i]?(*((double*)(d+8))):(*((double*)(d+16))))*x1[i])", { c14, d14} },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i])))", { c15, d15} },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*((*((double*)(d+8)))*x1[i])))", { c16, d16} },
    { "f32,f32->f32|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", { c17, d17} },
    { "f16,f16->f16|(x0[i]+x1[i])", { c18, d18} },
    { "f32,f32->f32|(x0[i]+x1[i])", { c19, d19} },
    { "f16,f16->f16|(x0[i]<x1[i]?x0[i]:x1[i])", { c20, d20} },
    { "f16,f16->f16|(x0[i]>x1[i]?x0[i]:x1[i])", { c21, d21} },
    { "f32,f32->f32|(x0[i]<x1[i]?x0[i]:x1[i])", { c22, d22} },
    { "f32,f32->f32|(x0[i]>x1[i]?x0[i]:x1[i])", { c23, d23} },
    { "f64,f64->f64|(x0[i]+x1[i])", { c24, d24} },
    { "f64,f64->f64|(x0[i]<x1[i]?x0[i]:x1[i])", { c25, d25} },
    { "f64,f64->f64|(x0[i]>x1[i]?x0[i]:x1[i])", { c26, d26} },
    { "f16,f16->f16|(x0[i]+((*((float16_t*)(d+0)))*x1[i]))", { c27, d27} },
    { "f32,f32->f32|(x0[i]+((*((float*)(d+0)))*x1[i]))", { c28, d28} },
    { "f64,f64->f64|(x0[i]+((*((double*)(d+0)))*x1[i]))", { c29, d29} },
    { "f64,f64->f64|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", { c30, d30} },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))>=x1[i]?(*((float*)(d+4))):(*((float*)(d+8)))))", {c31, d31} },
    { "f64,f64->f64|(x0[i]*((*((double*)(d+0)))>=x1[i]?(*((double*)(d+4))):(*((double*)(d+8)))))", {c32, d32} },
    { "f32,f32->f32|((*((float*)(d+0)))*((*((float*)(d+4)))*(x0[i]+((*((float*)(d+8)))*x1[i]))))", {c33, d33} },
    { "f32,f32->f32|((*((double*)(d+0)))*((*((double*)(d+4)))*(x0[i]+((*((double*)(d+8)))*x1[i]))))", {c34, d34} },
    { "f32,f32->f32|(((*((float*)(d+0)))*x0[i])+((*((float*)(d+4)))*x1[i]))", {c35,d35} },
    { "f32,f32->f32|(((*((float*)(d+0)))*x0[i])+((*((float*)(d+4)))*_pow(x1[i],(*((double*)(d+8))))))", {c36,d36} },
    { "f32,f32->f32|(x0[i]==x1[i]?(*((float*)(d+0))):(*((float*)(d+4))))", {c37,d37} },
    { "f32,f32->f32|(x0[i]*_exp(x1[i]))", {c38,d38} },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*_pow(x1[i],(*((double*)(d+4))))))", {c39,d39} },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*x1[i]))", {c40,d40} },
    { "f32,f32->f32|(x0[i]*(_pow(((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x1[i]))),(*((double*)(d+8))))+(x1[i]*((*((float*)(d+16)))*(_pow(((*((float*)(d+20)))+_exp(((*((float*)(d+24)))*x1[i]))),(*((double*)(d+28))))*((*((float*)(d+36)))*_exp(((*((float*)(d+40)))*x1[i]))))))))", {c41,d41} },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*_pow(x1[i],(*((double*)(d+4))))))", {c42,d42} },
    { "f32,f32->f32|(x0[i]*((*((float*)(d+0)))*x1[i]))", {c43,d43} }
  };

  auto iter = kernels.find(key);
  if(iter == kernels.end()) {
    return std::nullopt;
  }
  using tt = tuple<vector<uint8_t>, kernel_t>;
  if(is_ab_a) {
    return optional<tt>(tt{bytes, std::get<0>(iter->second)});
  } else {
    return optional<tt>(tt{bytes, std::get<1>(iter->second)});
  }
}

optional<tuple<
  vector<uint8_t>,
  void(*)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*, void const*)> >
lookup_ternary_2112_ew_kernel(
  scalarop_t op)
{
  // TODO: this shouldn't have to happen as op should always be simplified
  //       to a unique value. For some reason
  //       a kernel wasn't normalized in the same way as the key
  //       requires...
  op = op.simplify();

  auto [op_str, bytes] = op.to_cpp_bytes();
  string key = op.type_signature() + "|" + op_str;

  using kernel_t =
    void(*)(uint8_t const*, uint64_t, uint64_t, void*, void const*, void const*, void const*);

  static map< string, kernel_t > kernels = {
    { "f32,f32,f32->f32|(x0[i]*(x1[i]*((*((float*)(d+0)))*_pow(x2[i],(*((double*)(d+4)))))))", t2112_0 },
    { "f32,f32,f32->f32|((x0[i]*_pow(x1[i],(*((double*)(d+0)))))*x2[i])", t2112_1 }
  };

  auto iter = kernels.find(key);
  if(iter == kernels.end()) {
    return std::nullopt;
  }
  using tt = tuple<vector<uint8_t>, kernel_t>;
  return optional<tt>(tt{bytes, iter->second});
}

#define _real_reduction_ab_a(name, op) \
  template<typename T> \
  void name(uint64_t n1, uint64_t n2, T* out, T const* inn) { \
    double total; \
    for(uint64_t i = 0; i != n1; ++i) { \
      total = inn[i*n2]; \
      for(uint64_t j = 1; j != n2; ++j) { \
        uint64_t ij = i*n2 + j; \
        total op ; \
      } \
      out[i] = total; \
    } \
  }
#define _reduction_ab_a(name, op) \
   template<typename T> \
   void name(uint64_t n1, uint64_t n2, T* out, T const* inn) { \
     for(uint64_t i = 0; i != n1; ++i) { \
       out[i] = inn[i*n2]; \
       for(uint64_t j = 1; j != n2; ++j) { \
         uint64_t ij = i*n2 + j; \
         out[i] op ; \
       } \
     } \
   }
_real_reduction_ab_a(real_reduction_ab_a_add, += double(inn[ij])                );
_real_reduction_ab_a(real_reduction_ab_a_mul, *= double(inn[ij])                );

_reduction_ab_a(reduction_ab_a_add, += inn[ij]                  );
_reduction_ab_a(reduction_ab_a_mul, *= inn[ij]                  );
_reduction_ab_a(reduction_ab_a_min, =  std::min(out[i], inn[ij]));
_reduction_ab_a(reduction_ab_a_max, =  std::max(out[i], inn[ij]));

// TODO: better precision algorithm for reductions would be preferred

#define _real_reduction_lambda(castable,T) \
  [](uint64_t na, uint64_t nb, void* out, void const* inn) { \
    real_reduction_ab_a_##castable(na, nb, (T*)out, (T const*)inn); \
  };
#define _reduction_lambda(castable,T) \
  [](uint64_t na, uint64_t nb, void* out, void const* inn) { \
    reduction_ab_a_##castable(na, nb, (T*)out, (T const*)inn); \
  };

std::function<void(uint64_t, uint64_t, void*, void const*)>
build_ab_a_reduction_kernel(dtype_t dtype, castable_t castable) {
  if(dtype == dtype_t::f16) {
    if(castable == castable_t::add) {
      return _real_reduction_lambda(add, float16_t);
      //return _reduction_lambda(add, float16_t);
    } else if(castable == castable_t::mul) {
      return _real_reduction_lambda(mul, float16_t);
      //return _reduction_lambda(mul, float16_t);
    } else if(castable == castable_t::min) {
      return _reduction_lambda(min, float16_t);
    } else if(castable == castable_t::max) {
      return _reduction_lambda(max, float16_t);
    }
  } else if(dtype == dtype_t::f32) {
    if(castable == castable_t::add) {
      return _real_reduction_lambda(add, float);
      //return _reduction_lambda(add, float);
    } else if(castable == castable_t::mul) {
      return _real_reduction_lambda(mul, float);
      //return _reduction_lambda(mul, float);
    } else if(castable == castable_t::min) {
      return _reduction_lambda(min, float);
    } else if(castable == castable_t::max) {
      return _reduction_lambda(max, float);
    }
  } else if(dtype == dtype_t::f64) {
    if(castable == castable_t::add) {
      return _reduction_lambda(add, double);
    } else if(castable == castable_t::mul) {
      return _reduction_lambda(mul, double);
    } else if(castable == castable_t::min) {
      return _reduction_lambda(min, double);
    } else if(castable == castable_t::max) {
      return _reduction_lambda(max, double);
    }
  } else if(dtype == dtype_t::c64) {
    if(castable == castable_t::add) {
      return _reduction_lambda(add, std::complex<float>);
    } else if(castable == castable_t::mul) {
      return _reduction_lambda(mul, std::complex<float>);
    }
  }
  throw std::runtime_error("could not build ab_a reduction kernel");
}

// trans lhs   trans rhs
// F           F          ij,jk->ik
// T           F          ji,jk->ik
// F           T          ji,jk->ik
// T           T          ji,kj->ik
void matrix_multiply_update(
  dtype_t const& dtype,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs,
  bool is_zero_else_one)
{
  if(dtype == dtype_t::f16) {
    using f16_t = MKL_F16;
    // Use the half library (float16_t) to set one and zero,
    // then convert it to whatever type mkl sets MKL_F16 to.
    float16_t one_(1.0);
    float16_t zero_(0.0);
    f16_t& one  = reinterpret_cast<f16_t&>(one_);
    f16_t& zero = reinterpret_cast<f16_t&>(zero_);

    cblas_hgemm(
      CblasRowMajor,
      trans_lhs ? CblasTrans : CblasNoTrans,
      trans_rhs ? CblasTrans : CblasNoTrans,
      ni,nk,nj,
      one,
      (f16_t const*)lhs,
      trans_lhs ? ni : nj,
      (f16_t const*)rhs,
      trans_rhs ? nj : nk,
      is_zero_else_one ? zero : one,
      (f16_t*)out,
      nk);
  } else if(dtype == dtype_t::f32) {
    cblas_sgemm(
      CblasRowMajor,
      trans_lhs ? CblasTrans : CblasNoTrans,
      trans_rhs ? CblasTrans : CblasNoTrans,
      ni,nk,nj,
      1.0f,
      (float const*)lhs,
      trans_lhs ? ni : nj,
      (float const*)rhs,
      trans_rhs ? nj : nk,
      is_zero_else_one ? 0.0f : 1.0f,
      (float*)out,
      nk);
  } else if(dtype == dtype_t::f64) {
    cblas_dgemm(
      CblasRowMajor,
      trans_lhs ? CblasTrans : CblasNoTrans,
      trans_rhs ? CblasTrans : CblasNoTrans,
      ni,nk,nj,
      1.0,
      (double const*)lhs,
      trans_lhs ? ni : nj,
      (double const*)rhs,
      trans_rhs ? nj : nk,
      is_zero_else_one ? 0.0 : 1.0,
      (double*)out,
      nk);
  } else if(dtype == dtype_t::c64) {
    std::complex<float> one(1.0, 0.0);
    std::complex<float> zero(0.0, 0.0);
    cblas_cgemm(
      CblasRowMajor,
      trans_lhs ? CblasTrans : CblasNoTrans,
      trans_rhs ? CblasTrans : CblasNoTrans,
      ni,nk,nj,
      (void*)&one,
      lhs,
      trans_lhs ? ni : nj,
      rhs,
      trans_rhs ? nj : nk,
      is_zero_else_one ? (void*)&zero : (void*)&one,
      out,
      nk);
  } else {
    throw std::runtime_error("matmul type missing");
  }
}

void matrix_multiply(
  dtype_t const& dtype,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs)
{
  matrix_multiply_update(dtype, ni,nj,nk, trans_lhs,trans_rhs, out,lhs,rhs, true);
}

// b<ij> , b<jk> -> b<ik>
//
// This kernel includes things like
//   bij,jk->ik
//   ji,bjk->bik
//   ij,jk->bik
//   bij,bjk->ik
// by just looping over the batched dimension
void batch_matrix_multiply(
  dtype_t const& dtype,
  uint64_t const& nb,
  bool const& batched_out,
  bool const& batched_lhs,
  bool const& batched_rhs,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  void* _out,
  void const* _lhs,
  void const* _rhs)
{
  if(nb == 1) {
    matrix_multiply(dtype, ni,nj,nk,trans_lhs, trans_rhs, _out, _lhs, _rhs);
    return;
  }

  uint8_t      * out = (uint8_t      *)_out;
  uint8_t const* lhs = (uint8_t const*)_lhs;
  uint8_t const* rhs = (uint8_t const*)_rhs;

  uint64_t offset_lhs = batched_lhs ? dtype_size(dtype)*ni*nj : 0 ;
  uint64_t offset_rhs = batched_rhs ? dtype_size(dtype)*nj*nk : 0 ;
  if(batched_out) {
    uint64_t offset_out = dtype_size(dtype)*ni*nk;
    for(int b = 0; b != nb; ++b) {
      matrix_multiply(
        dtype, ni, nj, nk, trans_lhs, trans_rhs,
        (void*)out, (void const*)lhs, (void const*)rhs);
      lhs += offset_lhs;
      rhs += offset_rhs;
      out += offset_out;
    }
  } else {
    matrix_multiply_update(
      dtype, ni, nj, nk, trans_lhs, trans_rhs,
      (void*)out, (void const*)lhs, (void const*)rhs,
      true);
    lhs += offset_lhs;
    rhs += offset_rhs;
    for(int b = 1; b != nb; ++b) {
      matrix_multiply_update(
        dtype, ni, nj, nk, trans_lhs, trans_rhs,
        (void*)out, (void const*)lhs, (void const*)rhs,
        false);
      lhs += offset_lhs;
      rhs += offset_rhs;
    }
  }
}

void c64_mul_abcd_bd_to_abcd(
  uint64_t na,
  uint64_t nb,
  uint64_t nc,
  uint64_t nd,
  std::complex<float>* out,
  std::complex<float> const* lhs,
  std::complex<float> const* rhs)
{
  for(uint64_t a = 0; a != na; ++a) {
  for(uint64_t b = 0; b != nb; ++b) {
  for(uint64_t c = 0; c != nc; ++c) {
  for(uint64_t d = 0; d != nd; ++d) {
    uint64_t ol = a*nb*nc*nd + b*nc*nd + c*nd + d;
    uint64_t rr = b*nd + d;
    out[ol] = lhs[ol] * rhs[rr];
  }}}}
}
void c64_mulconj_abcd_bd_to_abcd(
  uint64_t na,
  uint64_t nb,
  uint64_t nc,
  uint64_t nd,
  std::complex<float>* out,
  std::complex<float> const* lhs,
  std::complex<float> const* rhs)
{
  for(uint64_t a = 0; a != na; ++a) {
  for(uint64_t b = 0; b != nb; ++b) {
  for(uint64_t c = 0; c != nc; ++c) {
  for(uint64_t d = 0; d != nd; ++d) {
    uint64_t ol = a*nb*nc*nd + b*nc*nd + c*nd + d;
    uint64_t rr = b*nd + d;
    out[ol] = lhs[ol] * _conj(rhs[rr]);
  }}}}
}

void permute_kernel(
  dtype_t dtype,
  uint64_t permute_block_size,
  vector<uint64_t> const& inn_shape,
  vector<int> const& out_perm,
  void* out,
  void const* inn)
{
  permute_t permute(permute_block_size);

  if(dtype == dtype_t::f16) {
    using T = float16_t;
    permute(inn_shape, out_perm, (T*)out, (T const*)inn);
  } else if(dtype == dtype_t::f32) {
    using T = float;
    permute(inn_shape, out_perm, (T*)out, (T const*)inn);
  } else if(dtype == dtype_t::f64) {
    using T = double;
    permute(inn_shape, out_perm, (T*)out, (T const*)inn);
  } else if(dtype == dtype_t::c64) {
    using T = std::complex<float>;
    permute(inn_shape, out_perm, (T*)out, (T const*)inn);
  } else {
    throw std::runtime_error("permute kernel missing dtype");
  }
}

// b->ab
void broadcast_b_ab_kernel(
  uint64_t nelem_a,
  uint64_t sz_b,
  void* _out,
  void const* inn)
{
  uint8_t* out = reinterpret_cast<uint8_t*>(_out);
  for(uint64_t i = 0; i != nelem_a; ++i) {
    std::memcpy(
      reinterpret_cast<void*>(out), inn, sz_b);
    out += sz_b;
  }
}

// a->ab
template <typename T>
void _broadcast_a_ab_kernel(
  uint64_t nelem_a,
  uint64_t nelem_b,
  T* out,
  T const* inn)
{
  for(uint64_t a = 0; a != nelem_a; ++a) {
    T const& val = inn[a];
    for(uint64_t b = 0; b != nelem_b; ++b) {
      *out++ = val;
    }
  }
}

void broadcast_a_ab_kernel(
  dtype_t dtype,
  uint64_t nelem_a,
  uint64_t nelem_b,
  void* out,
  void const* inn)
{
  if(dtype == dtype_t::f16) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint16_t*>(out),
      reinterpret_cast<uint16_t const*>(inn));
  } else if(dtype == dtype_t::f32) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint32_t*>(out),
      reinterpret_cast<uint32_t const*>(inn));
  } else if(dtype == dtype_t::f64 || dtype == dtype_t::c64) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint64_t*>(out),
      reinterpret_cast<uint64_t const*>(inn));
  } else {
    throw std::runtime_error("broadcast_a_ab_kernel: missing dtype implementation");
  }
}

template <typename T>
void _sum_then_scale_ab_a_kernel(
  T const& val,
  uint64_t n1,
  uint64_t n2,
  T* out,
  T const* inn)
{
  double total;
  for(uint64_t i = 0; i != n1; ++i) {
    double total = inn[i*n2];
    for(uint64_t j = 1; j != n2; ++j) {
      uint64_t ij = i*n2 + j;
      total += double(inn[ij]);
    }
    out[i] = T(double(val)*total);
  }
}

void sum_then_scale_ab_a_kernel(
  scalar_t value,
  uint64_t nelem_a,
  uint64_t nelem_b,
  void* out,
  void const* inn)
{
  if(value.dtype == dtype_t::f16) {
    _sum_then_scale_ab_a_kernel(
      value.f16(),
      nelem_a, nelem_b,
      reinterpret_cast<float16_t*>(out),
      reinterpret_cast<float16_t const*>(inn));
  } else if(value.dtype == dtype_t::f32) {
    _sum_then_scale_ab_a_kernel(
      value.f32(),
      nelem_a, nelem_b,
      reinterpret_cast<float*>(out),
      reinterpret_cast<float const*>(inn));
  } else if(value.dtype == dtype_t::f64) {
    _sum_then_scale_ab_a_kernel(
      value.f64(),
      nelem_a, nelem_b,
      reinterpret_cast<double*>(out),
      reinterpret_cast<double const*>(inn));
  } else {
    throw std::runtime_error("sum_then_scale: missing dtype case");
  }
}

void custom01_float_ab_ab_a_to_a(
  uint64_t na,
  uint64_t nb,
  float* out,
  float const* x0,
  float const* x1,
  float const* x2)
{
  double total;
  double v2;
  for(uint64_t a = 0; a != na; ++a) {
    v2 = -1.0 * _pow(double(x2[a]), -2.0);
    total = 0.0;
    float const* xx0 = x0 + a*nb;
    float const* xx1 = x1 + a*nb;
    for(uint64_t b = 0; b != nb; ++b) {
      total += xx0[b]*(xx1[b]*v2);
    }
    out[a] = float(total);
  }
}

template <typename T>
void _constant_fill(uint64_t nelem, T* out, T const& val)
{
  std::fill(out, out + nelem, val);
}

void constant_fill(uint64_t nelem, void* out, scalar_t value)
{
  if(value.dtype == dtype_t::f16) {
    _constant_fill(
      nelem,
      reinterpret_cast<uint16_t*>(out),
      *reinterpret_cast<uint16_t const*>(value.data));
  } else if(value.dtype == dtype_t::f32) {
    _constant_fill(
      nelem,
      reinterpret_cast<uint32_t*>(out),
      *reinterpret_cast<uint32_t const*>(value.data));
  } else if(value.dtype == dtype_t::f64 || value.dtype == dtype_t::c64){
    _constant_fill(
      nelem,
      reinterpret_cast<uint64_t*>(out),
      *reinterpret_cast<uint64_t const*>(value.data));
  } else {
    throw std::runtime_error("missing dtype impl: constant_fill");
  }
}

