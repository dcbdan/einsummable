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

static
void reduce_exp_sub(
  uint64_t na, uint64_t nb,
  void* out_, void const* lhs_, void const* rhs_);

static
void reduce_exp_subscale(
  uint64_t na, uint64_t nb, float const& alpha,
  void* out_, void const* lhs_, void const* rhs_);

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

    scalarop_t join = einsummable.join;
    if(swap_args) {
      // for castables, this won't do anything, but for subtraction and div,
      // this is required!
      join.remap_inputs({ {0, 1}, {1, 0} });
    }
    auto maybe = lookup_binary_212_ew_kernel(join, is_ab_a);

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

  if(estr == "ab,a->a") {
    dtype_t dtype = dtype_t::f32;
    scalarop_t sub_then_exp = scalarop_t::combine(
      scalarop_t::make_exp(dtype),
      { scalarop_t::make_sub(dtype) });
    if(einsummable.join == sub_then_exp && 
       einsummable.castable.value() == castable_t::add)
    {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      kernels.insert({einsummable, 
        [na,nb](void* out, vector<void const*> inns) {
          reduce_exp_sub(na,nb,out,inns[0],inns[1]);
        }
      });
      return 0; 
    } 

   string match_key = "f32,f32->f32|_exp(((*((float*)(d+0)))*(x0[i0]-x1[i1])))";
   auto const& join = einsummable.join;
   auto [_cpp, bytes] = join.to_cpp_bytes();
   string key = join.type_signature() + "|" + _cpp;
   if(match_key == key &&
      einsummable.castable.value() == castable_t::add)
    {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      kernels.insert({einsummable, 
        [na,nb,bytes](void* out, vector<void const*> inns) {
          float const& alpha = *reinterpret_cast<float const*>(bytes.data());
          reduce_exp_subscale(na,nb,alpha,out,inns[0],inns[1]);
        }
      });
      return 0; 
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
    dtype_t dtype = einsummable.out_dtype();
    scalar_t scale;
    if(einsummable.join.is_identity()) {
      scale = scalar_t::one(dtype);
    } else {
      auto maybe = einsummable.join.get_scale_from_scale();
      if(maybe) {
        scale = maybe.value();
      } else {
        return std::nullopt;
      }
    }

    kernels.insert({einsummable,
      broadcast_b_ab_t {
        .scalar = scale,
        .nelem_a = einsummable.join_shape[0],
        .nelem_b = einsummable.join_shape[1],
      }
    });
    return 0;
  }

  if(estr == "a->ab") {
    dtype_t dtype = einsummable.out_dtype();
    scalar_t scale;
    if(einsummable.join.is_identity()) {
      scale = scalar_t::one(dtype);
    } else {
      auto maybe = einsummable.join.get_scale_from_scale();
      if(maybe) {
        scale = maybe.value();
      } else {
        return std::nullopt;
      }
    }

    kernels.insert({einsummable,
      broadcast_a_ab_t {
        .scalar = scale,
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
      0, b.nb,
      b.info.batched_out, b.info.batched_lhs, b.info.batched_rhs,
      b.ni, 0, b.ni,        b.nj, b.nk,
      b.info.trans_lhs, b.info.trans_rhs,
      out, inns[0], inns[1],
      false);
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
    auto const& [scale,nelem_a,nelem_b] = get<broadcast_b_ab_t>(kernel);
    broadcast_b_ab_kernel(scale, nelem_a, nelem_b, out, inns[0]);
  } else if(holds_alternative<broadcast_a_ab_t>(kernel)) {
    //auto gremlin = cpu_kernel_timetracker.make_totals_gremlin("es:bro_a_ab");
    assert_num_inputs(1);
    auto const& [scale,nelem_a,nelem_b] = get<broadcast_a_ab_t>(kernel);
    broadcast_a_ab_kernel(scale, nelem_a, nelem_b, out, inns[0]);
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
  auto meta_info = ks.get_built_kernel_info(e);
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
inline T _square(T const& v) {
  return v*v;
}

template <typename T>
inline T _sqrt(T const& v) {
  return std::sqrt(v);
}

template <>
inline float16_t _sqrt(float16_t const& v) {
  return half_float::sqrt(v);
}

template <typename T>
inline T _invsqrt(T const& v) {
	return 1 / _sqrt(v); 
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
    for(uint64_t i0 = 0; i0 != n; ++i0) { \
      out[i0] = op; \
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
#define _binary_ew_loop_mkl(name1, name2, name3, T, vecop) \
  void name1( \
    uint8_t const* d, \
    uint64_t n, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    T* out     = reinterpret_cast<T*>(_out); \
    T const* x0 = reinterpret_cast<T const*>(_x0); \
    T const* x1 = reinterpret_cast<T const*>(_x1); \
    vecop(n, x0, 1, x1, 1, out, 1); \
  } \
  void name2( \
    uint8_t const* d, \
    uint64_t n1, \
    uint64_t n2, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    T* out     = reinterpret_cast<T*>(_out); \
    T const* x0 = reinterpret_cast<T const*>(_x0); \
    T const* x1 = reinterpret_cast<T const*>(_x1); \
    for(uint64_t _i = 0; _i != n1; ++_i) { \
      vecop(n2, \
        x0 + n2*_i,  1,  \
        x1 + _i,     0,  \
        out + n2*_i, 1); \
    } \
  } \
  void name3( \
    uint8_t const* d, \
    uint64_t n1, \
    uint64_t n2, \
    void* _out, \
    void const* _x0, \
    void const* _x1) \
  { \
    T* out     = reinterpret_cast<T*>(_out); \
    T const* x0 = reinterpret_cast<T const*>(_x0); \
    T const* x1 = reinterpret_cast<T const*>(_x1); \
    for(uint64_t _i = 0; _i != n1; ++_i) { \
      vecop(n2, \
        x0 + n2*_i,  1,  \
        x1,          1,  \
        out + n2*_i, 1); \
    } \
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

//_unary_ew_loop(u0,float,float,((*((float*)(d+0)))*x0[i0]))
inline void u0( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0)
{
  // fill the output memory with zero
  std::memset(_out, 0, sizeof(float)*n);

  // out = alpha*x0 + out
  float const& alpha = *((float*)d);
  cblas_saxpy(n, alpha, 
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float      *>(_out), 1);
}

_unary_ew_loop(u1,float,float,(x0[i0]/((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x0[i0])))))

//_unary_ew_loop(u2,float,float,((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i0])))
inline void u2( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0) 
{
  // compute alpha + beta*x
  float const& alpha = *((float*)(d+0));
  float const& beta  = *((float*)(d+4));

  float* out = reinterpret_cast<float*>(_out);
  std::fill(out, out + n, alpha);

  float const* inn = reinterpret_cast<float const*>(_x0);
  cblas_saxpy(n, beta, inn, 1, out, 1);
}

//_unary_ew_loop(u3,float,float,_exp(x0[i0]))
inline void u3( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0) 
{
  vsExpI(n, 
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float      *>(_out), 1);
}

//_unary_ew_loop(u4,float,float,x0[i0])
inline void u4( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0) 
{
  cblas_scopy(n, 
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float      *>(_out), 1);
}

//_unary_ew_loop(u5,float,float,_invsqrt(x0[i0]))
inline void u5( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0) 
{
  vsInvSqrtI(n, 
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float      *>(_out), 1);
}

//_unary_ew_loop(u6,float,float,_square(x0[i0]))
inline void u6( 
  uint8_t const* d, 
  uint64_t n, 
  void* _out, 
  void const* _x0) 
{
  vsMulI(n, 
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float const*>(_x0),  1,
    reinterpret_cast<float      *>(_out), 1);
}

_binary_ew_loop(b0,c0,d0,std::complex<float>,std::complex<float>,std::complex<float>,(x0[i0]*x1[i1]))

//_binary_ew_loop(b1,c1,d1,float,float,float,(x0[i0]*x1[i1]))
_binary_ew_loop_mkl(b1,c1,d1,float,vsMulI);

//_binary_ew_loop(b2,c2,d2,float,float,float,(x0[i0]/x1[i1]))
_binary_ew_loop_mkl(b2,c2,d2,float,vsDivI);

//_binary_ew_loop(b3,c3,d3,float,float,float,(x0[i0]-x1[i1]))
_binary_ew_loop_mkl(b3,c3,d3,float,vsSubI);

//_binary_ew_loop(b4,c4,d4,float,float,float,(x0[i0]+x1[i1]))
_binary_ew_loop_mkl(b4,c4,d4,float,vsAddI);

_binary_ew_loop(b5,c5,d5,float,float,float,_exp((x0[i0]-x1[i1])));

_ternary_ew_loop(t0,u0,float,float,float,float,(_exp((x0[i0]-x1[i1]))/x2[i2]));
_ternary_ew_loop(t1,u1,float,float,float,float,(_exp(((*((float*)(d+0)))*(x0[i0]-x1[i1])))/x2[i2]));

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
    { "f32->f32|((*((float*)(d+0)))*x0[i0])", u0 },
    { "f32->f32|(x0[i0]/((*((float*)(d+0)))+_exp(((*((float*)(d+4)))*x0[i0]))))", u1 },
    { "f32->f32|((*((float*)(d+0)))+((*((float*)(d+4)))*x0[i0]))", u2 },
    { "f32->f32|_exp(x0[i0])", u3 },
    { "f32->f32|x0[i0]", u4 },
    { "f32->f32|_invsqrt(x0[i0])", u5 },
    { "f32->f32|_square(x0[i0])", u6 }
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
    { "c64,c64->c64|(x0[i0]*x1[i1])", b0 },
    { "f32,f32->f32|(x0[i0]*x1[i1])", b1 },
    { "f32,f32->f32|(x0[i0]/x1[i1])", b2 },
    { "f32,f32->f32|(x0[i0]-x1[i1])", b3 },
    { "f32,f32->f32|(x0[i0]+x1[i1])", b4 },
    { "f32,f32->f32|_exp((x0[i0]-x1[i1]))", b5 },
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
    { "f32,f32,f32->f32|(_exp((x0[i0]-x1[i1]))/x2[i2])", t0 },
    { "f32,f32,f32->f32|(_exp(((*((float*)(d+0)))*(x0[i0]-x1[i1])))/x2[i2])", t1 }
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
    { "c64,c64->c64|(x0[i0]*x1[i1])", { c0, d0} },
    { "f32,f32->f32|(x0[i0]*x1[i1])", { c1, d1} },
    { "f32,f32->f32|(x0[i0]/x1[i1])", { c2, d2} },
    { "f32,f32->f32|(x0[i0]-x1[i1])", { c3, d3} },
    { "f32,f32->f32|(x0[i0]+x1[i1])", { c4, d4} },
    { "f32,f32->f32|_exp((x0[i0]-x1[i1]))", { c5, d5 } }
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
    { "f32,f32,f32->f32|(_exp((x0[i0]-x1[i1]))/x2[i2])", u0 },
    { "f32,f32,f32->f32|(_exp(((*((float*)(d+0)))*(x0[i0]-x1[i1])))/x2[i2])", u1 }
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
  dtype_t dtype,
  uint64_t ni, uint64_t offset_i, uint64_t size_i,
  uint64_t nj,
  uint64_t nk,
  bool trans_lhs,
  bool trans_rhs,
  void* out_,
  void const* lhs_,
  void const* rhs_,
  bool update)
{
  uint8_t      * out = reinterpret_cast<uint8_t      *>(out_);
  uint8_t const* lhs = reinterpret_cast<uint8_t const*>(lhs_);
  uint8_t const* rhs = reinterpret_cast<uint8_t const*>(rhs_);

  out += dtype_size(dtype)*(offset_i*nk);
  lhs += dtype_size(dtype)*( trans_lhs ? ( offset_i ) : ( offset_i*nj ) );

  // trans_lhs ? ji : ij
  // trans_rhs ? kj : jk

  auto tl = trans_lhs ? CblasTrans : CblasNoTrans;
  auto tr = trans_rhs ? CblasTrans : CblasNoTrans;

  auto sl = trans_lhs ? ni : nj;
  auto sr = trans_rhs ? nj : nk;

  if(dtype == dtype_t::f16) {
    using f16_t = MKL_F16;
    // Use the half library (float16_t) to set one and zero,
    // then convert it to whatever type mkl sets MKL_F16 to.
    float16_t one_(1.0);
    float16_t zero_(0.0);
    f16_t& one  = reinterpret_cast<f16_t&>(one_);
    f16_t& zero = reinterpret_cast<f16_t&>(zero_);

    cblas_hgemm(
      CblasRowMajor,tl,tr,
      size_i,nk,nj,
      one,
      reinterpret_cast<f16_t const*>(lhs),sl,
      reinterpret_cast<f16_t const*>(rhs),sr,
      update ? one : zero,
      reinterpret_cast<f16_t*>(out),
      nk);
  } else if(dtype == dtype_t::f32) {
    cblas_sgemm(
      CblasRowMajor,tl,tr,
      size_i,nk,nj,
      1.0f,
      reinterpret_cast<float const*>(lhs),sl,
      reinterpret_cast<float const*>(rhs),sr,
      update ? 1.0f : 0.0f,
      reinterpret_cast<float*>(out),
      nk);
  } else if(dtype == dtype_t::f64) {
    cblas_dgemm(
      CblasRowMajor,tl,tr,
      size_i,nk,nj,
      1.0,
      reinterpret_cast<double const*>(lhs),sl,
      reinterpret_cast<double const*>(rhs),sr,
      update ? 1.0 : 0.0,
      reinterpret_cast<double*>(out),
      nk);
  } else if(dtype == dtype_t::c64) {
    std::complex<float> one(1.0, 0.0);
    std::complex<float> zero(0.0, 0.0);
    cblas_cgemm(
      CblasRowMajor,tl,tr,
      size_i,nk,nj,
      (void*)&one,
      lhs,sl,
      rhs,sr,
      update ? (void*)&one : (void*)&zero,
      out,
      nk);
  } else {
    throw std::runtime_error("matmul type missing");
  }
}

void matrix_multiply(
  dtype_t dtype,
  uint64_t ni, uint64_t offset_i, uint64_t size_i,
  uint64_t nj,
  uint64_t nk,
  bool trans_lhs,
  bool trans_rhs,
  void* out,
  void const* lhs,
  void const* rhs)
{
  matrix_multiply_update(dtype,
    ni,offset_i,size_i,nj,nk,
    trans_lhs,trans_rhs,
    out,lhs,rhs,
    false);
}

// b<ij> , b<jk> -> b<ik>
void batch_matrix_multiply(
  dtype_t dtype,
  uint64_t offset_b, uint64_t size_b,
  bool batched_out,
  bool batched_lhs,
  bool batched_rhs,
  uint64_t ni, uint64_t offset_i, uint64_t size_i,
  uint64_t nj,
  uint64_t nk,
  bool trans_lhs,
  bool trans_rhs,
  void* _out,
  void const* _lhs,
  void const* _rhs,
  bool update)
{
  uint8_t      * out = (uint8_t      *)_out;
  uint8_t const* lhs = (uint8_t const*)_lhs;
  uint8_t const* rhs = (uint8_t const*)_rhs;

  uint64_t offset_lhs = batched_lhs ? dtype_size(dtype)*ni*nj : 0 ;
  uint64_t offset_rhs = batched_rhs ? dtype_size(dtype)*nj*nk : 0 ;

  lhs += offset_b * offset_lhs;
  rhs += offset_b * offset_rhs;

  if(batched_out) {
    uint64_t offset_out = dtype_size(dtype)*ni*nk;
    out += offset_b * offset_out;
    for(int b = 0; b != size_b; ++b) {
      matrix_multiply_update(
        dtype, ni, offset_i, size_i, nj, nk, trans_lhs, trans_rhs,
        reinterpret_cast<void*>(out),
        reinterpret_cast<void const*>(lhs),
        reinterpret_cast<void const*>(rhs),
        update);
      lhs += offset_lhs;
      rhs += offset_rhs;
      out += offset_out;
    }
  } else {
    matrix_multiply_update(
      dtype, ni, offset_i, size_i, nj, nk, trans_lhs, trans_rhs,
      reinterpret_cast<void*>(out),
      reinterpret_cast<void const*>(lhs),
      reinterpret_cast<void const*>(rhs),
      update);
    lhs += offset_lhs;
    rhs += offset_rhs;
    for(int b = 1; b != size_b; ++b) {
      matrix_multiply_update(
        dtype, ni, offset_i, size_i, nj, nk, trans_lhs, trans_rhs,
        reinterpret_cast<void*>(out),
        reinterpret_cast<void const*>(lhs),
        reinterpret_cast<void const*>(rhs),
        true);
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

template <typename T>
void _scale_kernel(
  T const& v,
  uint64_t n,
  void* _out)
{
  T* out = reinterpret_cast<T*>(_out);
  for(uint64_t i = 0; i != n; ++i) {
    out[i] *= v;
  }
}

void scale_kernel(
  scalar_t const& v,
  uint64_t nelem,
  void* out)
{
  if(v.dtype == dtype_t::f16) {
    if(v.f16() == float16_t(1.0)) {
      return;
    }
    _scale_kernel(v.f16(), nelem, out);
  } else if(v.dtype == dtype_t::f32) {
    if(v.f32() == float(1.0)) {
      return;
    }
    _scale_kernel(v.f32(), nelem, out);
  } else if(v.dtype == dtype_t::f64) {
    if(v.f64() == double(1.0)) {
      return;
    }
    _scale_kernel(v.f64(), nelem, out);
  } else if(v.dtype == dtype_t::c64) {
    if(v.c64() == std::complex<float>(1.0, 0.0)) {
      return;
    }
    _scale_kernel(v.c64(), nelem, out);
  } else {
    throw std::runtime_error("missing dtype: scale_kernel");
  }
}

// b->ab
void broadcast_b_ab_kernel(
  scalar_t const& scale,
  uint64_t nelem_a,
  uint64_t nelem_b,
  void* out,
  void const* inn)
{
  {
    uint8_t* _out = reinterpret_cast<uint8_t*>(out);
    uint64_t sz_b = dtype_size(scale.dtype) * nelem_b;
    for(uint64_t i = 0; i != nelem_a; ++i) {
      std::memcpy(
        reinterpret_cast<void*>(_out), inn, sz_b);
      _out += sz_b;
    }
  }

  scale_kernel(scale, nelem_a*nelem_b, out);
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
  scalar_t const& scalar,
  uint64_t nelem_a,
  uint64_t nelem_b,
  void* out,
  void const* inn)
{
  if(scalar.dtype == dtype_t::f16) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint16_t*>(out),
      reinterpret_cast<uint16_t const*>(inn));
  } else if(scalar.dtype == dtype_t::f32) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint32_t*>(out),
      reinterpret_cast<uint32_t const*>(inn));
  } else if(scalar.dtype == dtype_t::f64 || scalar.dtype == dtype_t::c64) {
    _broadcast_a_ab_kernel(
      nelem_a, nelem_b,
      reinterpret_cast<uint64_t*>(out),
      reinterpret_cast<uint64_t const*>(inn));
  } else {
    throw std::runtime_error("broadcast_a_ab_kernel: missing dtype implementation");
  }

  scale_kernel(scalar, nelem_a*nelem_b, out);
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

void constant_fill(scalar_t value, uint64_t nelem, void* out)
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

template <typename T>
void _lowertri_fill(
  T one, T zero,
  int64_t nrow, int64_t ncol, int64_t start,
  T* out)
{
  if(start >= nrow) {
    std::fill(out, out + nrow*ncol, zero);
    return;
  }

  if(start <= 1 - ncol) {
    std::fill(out, out + nrow*ncol, one);
    return;
  }

  int64_t _start = std::max(int64_t(0), start);
  int64_t _stop = std::min(nrow, ncol + start - 1);

  for(int64_t i = 0; i != _start; ++i) {
    T* p = out + i*ncol;
    std::fill(p, p + ncol, zero);
  }

  for(int64_t i = _start; i != _stop; ++i) {
    int64_t nc = i-start+1;
    T* p = out + i*ncol;
    std::fill(p,      p + nc,   one);
    std::fill(p + nc, p + ncol, zero);
  }
  for(int64_t i = _stop; i != nrow; ++i) {
    T* p = out + i*ncol;
    std::fill(p, p + ncol, one);
  }
}

void lowertri_fill(
  scalar_t one, scalar_t zero,
  int64_t nrow, int64_t ncol, int64_t start,
  void* out)
{
  if(one.dtype == dtype_t::f16) {
    _lowertri_fill(
      one.f16(),
      zero.f16(),
      nrow, ncol, start,
      reinterpret_cast<float16_t*>(out));
  } else if(one.dtype == dtype_t::f32) {
    _lowertri_fill(
      one.f32(),
      zero.f32(),
      nrow, ncol, start,
      reinterpret_cast<float*>(out));
  } else if(one.dtype == dtype_t::f64) {
    _lowertri_fill(
      one.f64(),
      zero.f64(),
      nrow, ncol, start,
      reinterpret_cast<double*>(out));
  } else if(one.dtype == dtype_t::c64) {
    _lowertri_fill(
      one.c64(),
      zero.c64(),
      nrow, ncol, start,
      reinterpret_cast<std::complex<float>*>(out));
  } else {
    throw std::runtime_error("lowertri_fill: missing dtype");
  }
}

void initialize_fill(fill_t const& fill, void* out)
{
  if(fill.is_constant()) {
    auto const& c = fill.get_constant();
    constant_fill(c.value, product(c.shape), out);
  } else if(fill.is_lowertri()) {
    auto const& l = fill.get_lowertri();
    lowertri_fill(l.lower, l.upper, int64_t(l.nrow), int64_t(l.ncol), l.start, out);
  } else {
    throw std::runtime_error("initialize_fill: should not reach");
  }
}

void reduce_exp_sub(
  uint64_t na, uint64_t nb,
  void* out_, void const* lhs_, void const* rhs_)
{
  float*       out = reinterpret_cast<float      *>(out_);
  float const* lhs = reinterpret_cast<float const*>(lhs_);
  float const* rhs = reinterpret_cast<float const*>(rhs_);

  // exceute "ab,a->a" with castable = add, join = exp(lhs-rhs)
  for(uint64_t i = 0; i != na; ++i) {
    float const* l = lhs + (i*nb);
    float const& r = rhs[i];
    out[i] = _exp(l[0]-r);
    for(uint64_t j = 1; j != nb; ++j) {
      out[i] += _exp(l[j]-r);
    }
  }
}

static
void reduce_exp_subscale(
  uint64_t na, uint64_t nb, float const& alpha,
  void* out_, void const* lhs_, void const* rhs_)
{
  float*       out = reinterpret_cast<float      *>(out_);
  float const* lhs = reinterpret_cast<float const*>(lhs_);
  float const* rhs = reinterpret_cast<float const*>(rhs_);

  // exceute "ab,a->a" with castable = add, join = exp(alpha*(lhs-rhs))
  for(uint64_t i = 0; i != na; ++i) {
    float const* l = lhs + (i*nb);
    float const& r = rhs[i];
    out[i] = _exp(alpha*(l[0]-r));
    for(uint64_t j = 1; j != nb; ++j) {
      out[i] += _exp(alpha*(l[j]-r));
    }
  }
}


