#include "kernels.h"

#include <mkl_cblas.h>
#include <mkl.h>

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
#define _binary_ew_loop(name, TO, T0, T1, op) \
  void name( \
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
      out[i] = op; \
    } \
  }

std::function<void(uint8_t const*, uint64_t, void*, void const*)>
get_unary_kernel(string const& op_str)
{
  using kernel_t = std::function<
    void(uint8_t const*, uint64_t, void*, void const*)>;

  static map<string, kernel_t> kernels = {
  };

  auto iter = kernels.find(op_str);
  if(iter == kernels.end()) {
    throw std::runtime_error("kernel undefined for " + op_str);
  }
  return iter->second;
}

std::function<void(uint8_t const*, uint64_t, void*, void const*, void const*)>
get_binary_kernel(string const& op_str)
{
  using kernel_t = std::function<
    void(uint8_t const*, uint64_t, void*, void const*, void const*)>;

  static map<string, kernel_t> kernels = {
  };

  auto iter = kernels.find(op_str);
  if(iter == kernels.end()) {
    throw std::runtime_error("kernel undefined for " + op_str);
  }
  return iter->second;
}

vector<tuple<uint64_t, uint64_t>>
_zip_parts(
  vector<uint64_t> const& parts)
{
  vector<tuple<uint64_t, uint64_t>> ret;
  ret.reserve(parts.size());
  uint64_t offset = 0;
  for(auto const& p: parts) {
    ret.emplace_back(offset, offset + p);
    offset += p;
  }
  return ret;
}

kernel_t
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto [op_str, bytes] = op.to_cpp_bytes();
  auto f = get_unary_kernel(op_str);

  if(num_threads == 0 || n < num_threads) {
    return [f,n,bytes](void* out, vector<void const*> inns) {
      return f(bytes.data(), n, out, inns[0]);
    };
  }

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f,n,ranges,bytes](void* out, vector<void const*> inns) {
    vector<std::thread> ts;
    void const* inn = inns[0];
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f,
        bytes.data(),
        upper-lower,
        (void*)((char*)out + lower),
        (void*)((char*)inn + lower));
    }
    for(auto& t: ts) {
      t.join();
    }
  };
}

kernel_t
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto [op_str, bytes] = op.to_cpp_bytes();
  auto f = get_binary_kernel(op_str);

  if(num_threads == 0 || n < num_threads) {
    return [f,n,bytes](void* out, vector<void const*> inns) {
      return f(bytes.data(), n, out, inns[0], inns[1]);
    };
  }

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f,n,ranges,bytes](void* out, vector<void const*> inns) {
    vector<std::thread> ts;
    void const* lhs = inns[0];
    void const* rhs = inns[1];
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f,
        bytes.data(),
        upper-lower,
        (void*)((char*)out + lower),
        (void*)((char*)lhs + lower),
        (void*)((char*)rhs + lower));
    }
    for(auto& t: ts) {
      t.join();
    }
  };
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
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs,
  float const& beta)
{
  cblas_sgemm(
    CblasRowMajor,
    trans_lhs ? CblasTrans : CblasNoTrans,
    trans_rhs ? CblasTrans : CblasNoTrans,
    ni,nk,nj,
    1.0f,
    lhs,
    trans_lhs ? ni : nj,
    rhs,
    trans_rhs ? nj : nk,
    beta,
    out,
    nk);
}

void matrix_multiply(
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs)
{
  matrix_multiply_update(ni,nj,nk, trans_lhs,trans_rhs, out,lhs,rhs, 0.0);
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
  uint64_t const& nb,
  bool const& batched_out,
  bool const& batched_lhs,
  bool const& batched_rhs,
  uint64_t const& ni,
  uint64_t const& nj,
  uint64_t const& nk,
  bool const& trans_lhs,
  bool const& trans_rhs,
  float* out,
  float const* lhs,
  float const* rhs)
{
  if(nb == 1) {
    matrix_multiply(ni,nj,nk,trans_lhs, trans_rhs, out, lhs, rhs);
  }

  uint64_t offset_lhs = batched_lhs ? ni*nj : 0 ;
  uint64_t offset_rhs = batched_rhs ? nj*nk : 0 ;
  if(batched_out) {
    uint64_t offset_out = ni*nk;
    for(int b = 0; b != nb; ++b) {
      matrix_multiply(ni, nj, nk, trans_lhs, trans_rhs, out, lhs, rhs);
      lhs += offset_lhs;
      rhs += offset_rhs;
      out += offset_out;
    }
  } else {
    matrix_multiply_update(ni, nj, nk, trans_lhs, trans_rhs, out, lhs, rhs, 0.0);
    lhs += offset_lhs;
    rhs += offset_rhs;
    for(int b = 1; b != nb; ++b) {
      matrix_multiply_update(ni, nj, nk, trans_lhs, trans_rhs, out, lhs, rhs, 1.0);
      lhs += offset_lhs;
      rhs += offset_rhs;
    }
  }
}

// Note: This test isn't perfect. It won't determine that
//   (ijkl,jkmn->ijmn) is actually a matrix multiply.
//   (TODO, perhaps as an einsummable.simplify method...)
// Note: This test also won't determine that
//   (jk,ij->ik) is actually a matrix multiply with the inputs flipped.
//   (TODO...)
optional<kernel_t>
_make_matrix_multiply(
  einsummable_t const& einsummable)
{
  if(einsummable.join_shape.size() != 3 ||
     einsummable.out_rank          != 2)
  {
    return std::nullopt;;
  }

  int i = 0;
  int j = 2;
  int k = 1;

  // 02,21->01  is the no transpose variant.
  uint64_t const& ni = einsummable.join_shape[i];
  uint64_t const& nj = einsummable.join_shape[j];
  uint64_t const& nk = einsummable.join_shape[k];

  bool trans_lhs, trans_rhs;

  {
    auto const& idxs_lhs = einsummable.inns[0];
    if(idxs_lhs == vector<int>{i,j}) {
      trans_lhs = false;
    } else if(idxs_lhs == vector<int>({j,i})) {
      trans_lhs = true;
    } else {
      return std::nullopt;
    }
  }

  {
    auto const& idxs_rhs = einsummable.inns[1];
    if(idxs_rhs == vector<int>{j,k}) {
      trans_rhs = false;
    } else if(idxs_rhs == vector<int>({k,j})) {
      trans_rhs = true;
    } else {
      return std::nullopt;
    }
  }

  return [ni,nj,nk,trans_lhs,trans_rhs]
    (void* out, vector<void const*> inns)
  {
  //  return matrix_multiply(
  //    ni,nj,nk,
  //    trans_lhs,trans_rhs,
  //    out, inns[0], inns[1]);
  };
}

// TODO: see _make_matrix_multiply todo.
//       Also, this function is just too ugly
optional<kernel_t>
_make_batch_matrix_multiply(
  einsummable_t const& e)
{
  if(e.join_shape.size() != 4 || e.inns.size() != 2) {
    return std::nullopt;
  }

  bool batched_lhs, batched_rhs, batched_out;

  {
    int rank_lhs = e.inns[0].size();
    if(rank_lhs == 2) {
      batched_lhs = false;
    } else if(rank_lhs == 3) {
      batched_lhs = true;
    } else {
      return std::nullopt;
    }
  }

  {
    int rank_rhs = e.inns[1].size();
    if(rank_rhs == 2) {
      batched_rhs = false;
    } else if(rank_rhs == 3) {
      batched_rhs = true;
    } else {
      return std::nullopt;
    }
  }

  {
    if(e.out_rank == 2) {
      batched_out = false;
    } else if(e.out_rank == 3) {
      batched_out = true;
    } else {
      return std::nullopt;
    }
  }

  int b,i,j,k;

  if(batched_out) {
    batched_out = true;
    // (maybe b) ij, (maybe b) jk, -> bik
    //  0        13   0        32     012

    b = 0;
    i = 1;
    j = 3;
    k = 2;

  } else {
    // e.out_rank == 2
    // (maybe b) ij, (maybe b) jk, -> ik
    //  3        02   3        21     01
    //  _OR_
    //  2        03   2        31     01
    // TODO: unless einsummable is modified to not
    //       allow the indeterminism
    if(batched_lhs) {
      b = e.inns[0][0];
    } else if(batched_rhs) {
      b = e.inns[1][0];
    } else {
      // this is a matrix multiply, not a batched matrix multiply
      return std::nullopt;
    }

    if(b == 2 || b == 3) {
      j = b == 3 ? 2 : 3;
    } else {
      return std::nullopt;
    }

    i = 0;
    k = 1;
  }

  uint64_t const& nb = e.join_shape[b];
  uint64_t const& ni = e.join_shape[i];
  uint64_t const& nj = e.join_shape[j];
  uint64_t const& nk = e.join_shape[k];

  bool trans_lhs, trans_rhs;

  {
    auto const& idxs_lhs = e.inns[0];
    if(batched_lhs) {
      if(idxs_lhs == vector<int>{b,i,j}) {
        trans_lhs == false;
      } else if(idxs_lhs == vector<int>{b,j,i}) {
        trans_lhs == true;
      } else {
        return std::nullopt;
      }
    } else {
      if(idxs_lhs == vector<int>{i,j}) {
        trans_lhs == false;
      } else if(idxs_lhs == vector<int>{j,i}) {
        trans_lhs == true;
      } else {
        return std::nullopt;
      }
    }
  }

  {
    auto const& idxs_rhs = e.inns[1];
    if(batched_rhs) {
      if(idxs_rhs == vector<int>{b,j,k}) {
        trans_rhs == false;
      } else if(idxs_rhs == vector<int>{b,k,j}) {
        trans_rhs == true;
      } else {
        return std::nullopt;
      }
    } else {
      if(idxs_rhs == vector<int>{j,k}) {
        trans_rhs == false;
      } else if(idxs_rhs == vector<int>{k,j}) {
        trans_rhs == true;
      } else {
        return std::nullopt;
      }
    }
  }

  return [nb,batched_out,batched_lhs,batched_rhs,ni,nj,nk,trans_lhs,trans_rhs]
    (void* out, vector<void const*> inns)
  {
  //  return batch_matrix_multiply(
  //    nb,batched_out,batched_lhs,batched_rhs,
  //    ni,nj,nk,
  //    trans_lhs,trans_rhs,
  //    out, inns[0], inns[1]);
  };
}

kernel_t
build_einsummable(
  int num_threads,
  einsummable_t const& einsummable_)
{
  einsummable_t einsummable = einsummable_.merge_adjacent_dims();

  if(einsummable.is_straight_elementwise()) {
    int n = einsummable.inns.size();
    if(n == 1) {
      return build_unary_elementwise_kernel(
        num_threads,
        product(einsummable.join_shape),
        einsummable.join);
    } else if(n == 2) {
      return build_binary_elementwise_kernel(
        num_threads,
        product(einsummable.join_shape),
        einsummable.join);
    } else {
      throw std::runtime_error(
              "straight elementwise kernel with " + std::to_string(n) +
              " inputs not supported!");
    }
  }

  auto matmul = _make_matrix_multiply(einsummable);
  if(matmul) {
    return matmul.value();
  }

  auto batch_matmul = _make_batch_matrix_multiply(einsummable);
  if(batch_matmul) {
    return batch_matmul.value();
  }

  throw std::runtime_error("could not acquire kernel for " + write_with_ss(einsummable));
}
