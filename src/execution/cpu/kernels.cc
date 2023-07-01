#include "kernels.h"

#include <mkl_cblas.h>
#include <mkl.h>

#include "permute.h"

kernel_manager_t::kernel_manager_t()
{
  auto fix = einsummable_t::normalize_str;

  binfos = {
   { fix("ij,jk->ik"), { false,false, false,false,false } },
   { fix("ij,kj->ik"), { false, true, false,false,false } },
   { fix("ji,jk->ik"), {  true,false, false,false,false } },
   { fix("ji,kj->ik"), {  true,false, false,false,false } },

   { fix("bij,jk->ik"), { false,false, true,false,false } },
   { fix("bij,kj->ik"), { false, true, true,false,false } },
   { fix("bji,jk->ik"), {  true,false, true,false,false } },
   { fix("bji,kj->ik"), {  true,false, true,false,false } },

   { fix("ij,bjk->ik"), { false,false, false,true,false } },
   { fix("ij,bkj->ik"), { false, true, false,true,false } },
   { fix("ji,bjk->ik"), {  true,false, false,true,false } },
   { fix("ji,bkj->ik"), {  true,false, false,true,false } },

   { fix("bij,bjk->ik"), { false,false, true,true,false } },
   { fix("bij,bkj->ik"), { false, true, true,true,false } },
   { fix("bji,bjk->ik"), {  true,false, true,true,false } },
   { fix("bji,bkj->ik"), {  true,false, true,true,false } },

   { fix("bij,jk->bik"), { false,false, true,false,true } },
   { fix("bij,kj->bik"), { false, true, true,false,true } },
   { fix("bji,jk->bik"), {  true,false, true,false,true } },
   { fix("bji,kj->bik"), {  true,false, true,false,true } },

   { fix("ij,bjk->bik"), { false,false, false,true,true } },
   { fix("ij,bkj->bik"), { false, true, false,true,true } },
   { fix("ji,bjk->bik"), {  true,false, false,true,true } },
   { fix("ji,bkj->bik"), {  true,false, false,true,true } },

   { fix("bij,bjk->bik"), { false,false, true,true,true } },
   { fix("bij,bkj->bik"), { false, true, true,true,true } },
   { fix("bji,bjk->bik"), {  true,false, true,true,true } },
   { fix("bji,bkj->bik"), {  true,false, true,true,true } }
  };
}

optional<uint64_t> kernel_manager_t::build(einsummable_t const& e_)
{
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
    } else {
      return std::nullopt;
    }
  }

  auto estr = einsummable.str();

  if(estr == "ab,a->ab" || estr == "ab,b->ab") {
    bool is_ab_a = estr == "ab,a->ab";
    auto maybe = lookup_binary_212_ew_kernel(einsummable.join, is_ab_a);
    if(maybe) {
      auto const& [data,f] = maybe.value();
      kernels.insert({einsummable,
        binary_212_ew_t {
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

  if(estr == "abcd,bd->abcd")
  {
    if(einsummable.join.is_mul() && dtype_t::c64 == einsummable.out_dtype())
    {
      uint64_t na = einsummable.join_shape[0];
      uint64_t nb = einsummable.join_shape[1];
      uint64_t nc = einsummable.join_shape[2];
      uint64_t nd = einsummable.join_shape[3];
      kernel_t kernel = [na,nb,nc,nd](void* out, vector<void const*> inns) {
        using T = std::complex<float>;
        return c64_mul_abcd_bd_to_abcd(
          na,nb,nc,nd,
          (T*)out, (T const*)inns[0], (T const*)inns[1]);
      };
      kernels.insert({einsummable, kernel});
      return 0;
    } else {
      return std::nullopt;
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
uint64_t kernel_manager_t::workspace_size(einsummable_t const& e) const
{
  auto const& kernel = get_built_kernel_info(e);

  if(std::holds_alternative<contraction_t>(kernel)) {
    return std::get<contraction_t>(kernel).workspace_size;
  } else {
    return 0;
  }
}

vector<int> kernel_manager_t::donatables(einsummable_t const& e) const
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
  } else if(std::holds_alternative<binary_212_ew_t>(kernel)) {
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

#ifdef KERNEL_TIMING

#include <memory>
#include <mutex>
#include <atomic>

struct total_timer_t {
  total_timer_t(std::atomic_uint64_t& t):
    total(t), start(clock_now())
  {}

  ~total_timer_t() {
    auto end = clock_now();
    using namespace std::chrono;
    total += duration_cast<microseconds>(end - start).count();
  }

  std::atomic_uint64_t& total;
  timestamp_t const start;
};

struct timer_totaler_t {
  total_timer_t operator()(string key) {
    std::unique_lock lk(m);

    auto iter = totals.find(key);
    if(iter == totals.end()) {
      auto [_i, _] = totals.insert({key, std::make_unique<std::atomic_uint64_t>(0)});
      iter = _i;
    }
    return total_timer_t(*(iter->second));
  }

  void reset() {
    std::unique_lock lk(m);
    for(auto const& [k,v_ptr]: totals) {
      *v_ptr = 0;
    }
  }

  std::mutex m;
  map<string, std::unique_ptr<std::atomic_uint64_t>> totals;
};

timer_totaler_t kernel_timer_totaler;

map<string, double> kernel_manager_t::get_times() {
  map<string, double> ret;
  for(auto const& [key, v_ptr]: kernel_timer_totaler.totals) {
    using namespace std::chrono;
    double time = (double) (v_ptr->load())
                / (double) duration_cast<microseconds>(1s).count();
    ret.insert({key, time});
  }
  return ret;
}

void kernel_manager_t::reset_times() {
  kernel_timer_totaler.reset();
}

#define setup_gremlin(key) total_timer_t gremlin = kernel_timer_totaler(key);
#else
#define setup_gremlin(key)
#endif

void kernel_manager_t::operator()(
  touch_t const& touch,
  void* out,
  void const* inn) const
{
  // TODO: there is no reason to wrap the touch kernel in a lambda;
  //       create touch_kernel
  auto f = build_touch(touch);
  setup_gremlin("touch");
  f(out, inn);
}

void kernel_manager_t::operator()(
  einsummable_t const& e,
  void* out,
  vector<void const*> inns,
  optional<tuple<void*, uint64_t>> maybe_workspace) const
{
  auto const& info = get_built_kernel_info(e);
  call(info, out, inns, maybe_workspace);
}

void kernel_manager_t::call(
  kernel_manager_t::kernel_info_t const& kernel,
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
    assert_num_inputs(2);
    auto const& b = get<batch_matmul_t>(kernel);
    setup_gremlin("bmm");
    batch_matrix_multiply(
      b.dtype,
      b.nb,
      b.info.batched_out, b.info.batched_lhs, b.info.batched_rhs,
      b.ni, b.nj, b.nk,
      b.info.trans_lhs, b.info.trans_rhs,
      out, inns[0], inns[1]);
  } else if(holds_alternative<contraction_t>(kernel)) {
    assert_num_inputs(2);
    auto const& c = get<contraction_t>(kernel);
    setup_gremlin("con");
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
    assert_num_inputs(1);
    auto const& [n,data,f] = get<unary_straight_ew_t>(kernel);
    setup_gremlin("uew");
    f(data.data(), n, out, inns[0]);
  } else if(holds_alternative<binary_straight_ew_t>(kernel)) {
    assert_num_inputs(2);
    auto const& [n,data,f] = get<binary_straight_ew_t>(kernel);
    setup_gremlin("bew");
    f(data.data(), n, out, inns[0], inns[1]);
  } else if(holds_alternative<binary_212_ew_t>(kernel)) {
    assert_num_inputs(2);
    auto const& [na,nb,data,f] = get<binary_212_ew_t>(kernel);
    setup_gremlin("b212");
    f(data.data(), na, nb, out, inns[0], inns[1]);
  } else if(holds_alternative<tensor_permute_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [dtype, inn_shape, out_perm] = get<tensor_permute_t>(kernel);
    setup_gremlin("perm");
    permute_kernel(dtype, 1024, inn_shape, out_perm, out, inns[0]);
  } else if(holds_alternative<reduction_ab_a_t>(kernel)) {
    assert_num_inputs(1);
    auto const& [na,nb,f] = get<reduction_ab_a_t>(kernel);
    setup_gremlin("red");
    f(na,nb,out,inns[0]);
  } else if(holds_alternative<kernel_t>(kernel)) {
    auto const& f = get<kernel_t>(kernel);
    setup_gremlin("misc");
    f(out, inns);
  } else {
    throw std::runtime_error("workspace size: kernel unaccounted for");
  }
}

kernel_manager_t::kernel_info_t const&
kernel_manager_t::get_built_kernel_info(einsummable_t const& e) const
{
  auto iter = kernels.find(e.merge_adjacent_dims());
  if(iter == kernels.end()) {
    throw std::runtime_error("get_built_kernel_info: this einsummable has not been built");
  }
  return iter->second;
}

optional<kernel_manager_t::batch_matmul_t>
kernel_manager_t::make_batch_matmul(einsummable_t const& e)
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
  kernel_manager_t ks;
  auto maybe = ks.build(e);
  if(!maybe) {
    throw std::runtime_error("could not build the kernel");
  }
  if(maybe.value() != 0) {
    throw std::runtime_error("build_einsummable: this kernel requires a workspace");
  }
  auto const& meta_info = ks.get_built_kernel_info(e);
  return [meta_info](void* out, vector<void const*> inns) {
    kernel_manager_t::call(meta_info, out, inns);
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
    for(uint64_t i = 0; i != n; ++i) { \
      uint64_t const& i0 = i; \
      uint64_t const& i1 = i; \
      out[i] = op; \
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
    for(uint64_t i = 0; i != n1; ++i) { \
    for(uint64_t j = 0; j != n2; ++j) { \
      uint64_t i0 = i*n2 + j; \
      uint64_t const& i1 = i; \
      out[i0] = op; \
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
    for(uint64_t i = 0; i != n1; ++i) { \
    for(uint64_t j = 0; j != n2; ++j) { \
      uint64_t i0 = i*n2 + j; \
      uint64_t const& i1 = j; \
      out[i0] = op; \
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
    { "f64->f64|_exp(x0[i])", u25 }
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
    { "f64,f64->f64|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", b30 }
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
    { "f64,f64->f64|(x0[i]*_pow(x1[i],(*((double*)(d+0)))))", { c30, d30} }
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

#define _touch1(name, op) \
  template <typename T> \
  void name(touchdim_t const& t0, T* out, T const* inn) { \
    out += t0.offset_out; \
    inn += t0.offset_inn; \
    for(uint64_t i = 0; i != t0.size; ++i) { \
      op; \
    } \
  }

#define _touch2(name, op) \
  template <typename T> \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    T* out, \
    T const* inn) \
  { \
    out += t0.offset_out*t1.d_out + t1.offset_out; \
    inn += t0.offset_inn*t1.d_inn + t1.offset_inn; \
    for(uint64_t i0 = 0; i0 != t0.size; ++i0) { \
      for(uint64_t i = 0; i != t1.size; ++i) { \
        op; \
      } \
      out += t1.d_out; \
      inn += t1.d_inn; \
    } \
  }

#define _touch3(name, op) \
  template <typename T> \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    T* out, \
    T const* inn) \
  { \
    out += t0.offset_out*t1.d_out*t2.d_out + t1.offset_out*t2.d_out + t2.offset_out; \
    inn += t0.offset_inn*t1.d_inn*t2.d_inn + t1.offset_inn*t2.d_inn + t2.offset_inn; \
    for(uint64_t i0 = 0; i0 != t0.size; ++i0) { \
      for(uint64_t i1 = 0; i1 != t1.size; ++i1) { \
        for(uint64_t i  = 0; i  != t2.size; ++i ) { \
          op; \
        } \
        out += t2.d_out; \
        inn += t2.d_inn; \
      } \
      out += t2.d_out * (t1.d_out - t1.size); \
      inn += t2.d_inn * (t1.d_inn - t1.size); \
    } \
  }

#define _touch4(name, op) \
  template <typename T> \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    touchdim_t const& t3, \
    T* out, \
    T const* inn) \
  { \
    out += t0.offset_out*t1.d_out*t2.d_out*t3.d_out + \
           t1.offset_out*t2.d_out*t3.d_out + \
           t2.offset_out*t3.d_out + \
           t3.offset_out; \
    inn += t0.offset_inn*t1.d_inn*t2.d_inn*t3.d_inn + \
           t1.offset_inn*t2.d_inn*t3.d_inn + \
           t2.offset_inn*t3.d_inn + \
           t3.offset_inn; \
    for(uint64_t i0 = 0; i0 != t0.size; ++i0) { \
      for(uint64_t i1 = 0; i1 != t1.size; ++i1) { \
        for(uint64_t i2 = 0; i2 != t2.size; ++i2) { \
          for(uint64_t i  = 0; i  != t3.size; ++i ) { \
            op; \
          } \
          out += t3.d_out; \
          inn += t3.d_inn; \
        } \
        out += t3.d_out * (t2.d_out - t2.size); \
        inn += t3.d_inn * (t2.d_inn - t2.size); \
      } \
      out += t3.d_out * t2.d_out * (t1.d_out - t1.size); \
      inn += t3.d_inn * t2.d_inn * (t1.d_inn - t1.size); \
    } \
  }

_touch1(touch1_none, out[i] =  inn[i]                  );
_touch1(touch1_add,  out[i] += inn[i]                  );
_touch1(touch1_mul,  out[i] *= inn[i]                  );
_touch1(touch1_min,  out[i] =  std::min(out[i], inn[i]));
_touch1(touch1_max,  out[i] =  std::max(out[i], inn[i]));

_touch2(touch2_none, out[i] =  inn[i]                  );
_touch2(touch2_add,  out[i] += inn[i]                  );
_touch2(touch2_mul,  out[i] *= inn[i]                  );
_touch2(touch2_min,  out[i] =  std::min(out[i], inn[i]));
_touch2(touch2_max,  out[i] =  std::max(out[i], inn[i]));

_touch3(touch3_none, out[i] =  inn[i]                  );
_touch3(touch3_add,  out[i] += inn[i]                  );
_touch3(touch3_mul,  out[i] *= inn[i]                  );
_touch3(touch3_min,  out[i] =  std::min(out[i], inn[i]));
_touch3(touch3_max,  out[i] =  std::max(out[i], inn[i]));

_touch4(touch4_none, out[i] =  inn[i]                  );
_touch4(touch4_add,  out[i] += inn[i]                  );
_touch4(touch4_mul,  out[i] *= inn[i]                  );
_touch4(touch4_min,  out[i] =  std::min(out[i], inn[i]));
_touch4(touch4_max,  out[i] =  std::max(out[i], inn[i]));

#define _touch_lambda_f_1(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype == dtype_t::f16) { \
      using T = float16_t; \
      name(ts[0], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f32) { \
      using T = float; \
      name(ts[0], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f64) { \
      using T = double; \
      name(ts[0], (T*)out, (T const*)inn); \
    } else { \
      throw std::runtime_error("shoud not reach: touch lambda"); \
    } \
  }
#define _touch_lambda_f_2(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype == dtype_t::f16) { \
      using T = float16_t; \
      name(ts[0], ts[1], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f32) { \
      using T = float; \
      name(ts[0], ts[1], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f64) { \
      using T = double; \
      name(ts[0], ts[1], (T*)out, (T const*)inn); \
    } else { \
      throw std::runtime_error("shoud not reach: touch lambda"); \
    } \
  }
#define _touch_lambda_f_3(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype == dtype_t::f16) { \
      using T = float16_t; \
      name(ts[0], ts[1], ts[2], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f32) { \
      using T = float; \
      name(ts[0], ts[1], ts[2], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f64) { \
      using T = double; \
      name(ts[0], ts[1], ts[2], (T*)out, (T const*)inn); \
    } else { \
      throw std::runtime_error("shoud not reach: touch lambda"); \
    } \
  }
#define _touch_lambda_f_4(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype == dtype_t::f16) { \
      using T = float16_t; \
      name(ts[0], ts[1], ts[2], ts[3], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f32) { \
      using T = float; \
      name(ts[0], ts[1], ts[2], ts[3], (T*)out, (T const*)inn); \
    } else if(dtype == dtype_t::f64) { \
      using T = double; \
      name(ts[0], ts[1], ts[2], ts[3], (T*)out, (T const*)inn); \
    } else { \
      throw std::runtime_error("shoud not reach: touch lambda"); \
    } \
  }

#define _touch_lambda_c_1(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype != dtype_t::c64) { throw std::runtime_error("wrong dtype: touch"); } \
    using T = std::complex<float>; \
    name(ts[0], (T*)out, (T const*)inn); \
  }
#define _touch_lambda_c_2(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype != dtype_t::c64) { throw std::runtime_error("wrong dtype: touch"); } \
    using T = std::complex<float>; \
    name(ts[0], ts[1], (T*)out, (T const*)inn); \
  }
#define _touch_lambda_c_3(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype != dtype_t::c64) { throw std::runtime_error("wrong dtype: touch"); } \
    using T = std::complex<float>; \
    name(ts[0], ts[1], ts[2], (T*)out, (T const*)inn); \
  }
#define _touch_lambda_c_4(name) \
  [dtype,ts](void* out, void const* inn) { \
    if(dtype != dtype_t::c64) { throw std::runtime_error("wrong dtype: touch"); } \
    using T = std::complex<float>; \
    name(ts[0], ts[1], ts[2], ts[3], (T*)out, (T const*)inn); \
  }

// This guy is wrapped in a lambda so
// the it creates a single output which
// can then be returned.
// So do
//   kernel_t my_function() {
//     return _touch_dispatch(1)
//    }
// as opposed to the non-wrapped version
// which would have
//  kernel_t my_function() {
//     _touch_dispatch(1)
//  }
// which mysteriously looks like it
// doesn't return anything.
#define _touch_dispatch_f(i) \
  [&]() -> std::function<void(void*, void const*)> { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_f_##i ( touch##i##_add); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_f_##i ( touch##i##_mul); \
      } else if(c == castable_t::min) { \
        return _touch_lambda_f_##i ( touch##i##_min); \
      } else if(c == castable_t::max) { \
        return  _touch_lambda_f_##i ( touch##i##_max); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_f_##i ( touch##i##_none); \
    } \
  }()
// For the complex case, don't include the min or the max
// cases
#define _touch_dispatch_c(i) \
  [&]() -> std::function<void(void*, void const*)> { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_c_##i ( touch##i##_add); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_c_##i ( touch##i##_mul); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_c_##i ( touch##i##_none); \
    } \
  }()

touch_kernel_t
build_touch(touch_t const& touch_)
{
  touch_t touch = touch_.simplify();

  auto const& dtype = touch.dtype;
  auto const& ts = touch.selection;
  if(ts.size() == 1) {
    return dtype == dtype_t::c64 ?
      _touch_dispatch_c(1) :
      _touch_dispatch_f(1) ;
  }

  if(ts.size() == 2) {
    return dtype == dtype_t::c64 ?
      _touch_dispatch_c(2) :
      _touch_dispatch_f(2) ;
  }

  if(ts.size() == 3) {
    return dtype == dtype_t::c64 ?
      _touch_dispatch_c(3) :
      _touch_dispatch_f(3) ;
  }

  if(ts.size() == 4) {
    return dtype == dtype_t::c64 ?
      _touch_dispatch_c(4) :
      _touch_dispatch_f(4) ;
  }

  throw std::runtime_error("touch kernel not implemented");
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


