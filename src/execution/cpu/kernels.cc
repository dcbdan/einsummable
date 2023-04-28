#include "kernels.h"

#define _unary_ew_loop(name, op) \
  void name( \
    uint64_t n, \
    float* out, \
    float const* x0) \
  { \
    for(uint64_t i = 0; i != n; ++i) { \
      out[i] = op; \
    } \
  }
#define _binary_ew_loop(name, op) \
  void name( \
    uint64_t n, \
    float* out, \
    float const* x0, \
    float const* x1) \
  { \
    for(uint64_t i = 0; i != n; ++i) { \
      out[i] = op; \
    } \
  }

// *[ite_<[hole@0,constant{0},constant{0},constant{1}],hole@1]
_binary_ew_loop(b0, ((x0[i]<0?0:1)*x1[i]))

// +[hole@0,*[*[hole@1,constant{0.3}],constant{-1}]]
_binary_ew_loop(b1, (x0[i]+((x1[i]*0.3)*-1)))

// +[hole@0,*[hole@1,constant{-1}]]
_binary_ew_loop(b2, (x0[i]+(x1[i]*-1)));

// +[hole@0,hole@1]
_binary_ew_loop(b3, (x0[i]+x1[i]));

// +[hole@0,*[*[hole@1,constant{0.1}],constant{-1}]]
_binary_ew_loop(b4, (x0[i]+((x1[i]*0.1)*-1)));

// ite_<[hole@0,constant{0},constant{0},hole@0]
_unary_ew_loop(u0, (x0[i]<0?0:x0[i]));

std::function<void(uint64_t, float*, float const*)> get_unary_kernel(scalarop_t const& op)
{
  using kernel_t = std::function<void(uint64_t, float*, float const*)>;

  static map<string, kernel_t> kernels {
    {
      "ite_<[hole@0,constant{0},constant{0},hole@0]",
      u0
    }
  };


  auto op_str = write_with_ss(op);
  auto iter = kernels.find(op_str);
  if(iter == kernels.end()) {
    throw std::runtime_error("kernel undefined for " + op_str);
  }
  return iter->second;
}

std::function<void(uint64_t, float*, float const*, float const*)>
get_binary_kernel(scalarop_t const& op)
{
  using kernel_t = std::function<void(uint64_t, float*, float const*, float const*)>;

  static map<string, kernel_t> kernels {
    {
      "*[ite_<[hole@0,constant{0},constant{0},constant{1}],hole@1]",
      b0
    },
    {
      "+[hole@0,*[*[hole@1,constant{0.3}],constant{-1}]]",
      b1
    },
    {
      "+[hole@0,*[hole@1,constant{-1}]]",
      b2
    },
    {
      "+[hole@0,hole@1]",
      b3
    },
    {
      "+[hole@0,*[*[hole@1,constant{0.1}],constant{-1}]]",
      b4
    }
  };

  auto op_str = write_with_ss(op);
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

std::function<void(float*,vector<float const*>)>
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto f = get_unary_kernel(op);

  if(num_threads == 0) {
    return [f,n](float* out, vector<float const*> inns) {
      return f(n, out, inns[0]);
    };
  }

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f,n,ranges](float* out, vector<float const*> inns) {
    vector<std::thread> ts;
    float const* inn = inns[0];
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f,
        upper-lower,
        out + lower,
        inn + lower);
    }
    for(auto& t: ts) {
      t.join();
    }
  };
}

std::function<void(float*,vector<float const*>)>
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto f = get_binary_kernel(op);

  if(num_threads == 0) {
    return [f,n](float* out, vector<float const*> inns) {
      return f(n, out, inns[0], inns[1]);
    };
  }

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f,n,ranges](float* out, vector<float const*> inns) {
    vector<std::thread> ts;
    float const* lhs = inns[0];
    float const* rhs = inns[1];
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f,
        upper-lower,
        out + lower,
        lhs + lower,
        rhs + lower);
    }
    for(auto& t: ts) {
      t.join();
    }
  };
}

#define _touch1(name, op) \
  void name(touchdim_t const& t0, float* out, float const* inn) { \
    out += t0.offset_out; \
    inn += t0.offset_inn; \
    for(uint64_t i = 0; i != t0.size; ++i) { \
      op; \
    } \
  }

#define _touch2(name, op) \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    float* out, \
    float const* inn) \
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
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    float* out, \
    float const* inn) \
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
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    touchdim_t const& t3, \
    float* out, \
    float const* inn) \
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

#define _touch_lambda_1(name) \
  [ts](float* out, float const* inn) { \
    name(ts[0], out, inn); \
  }
#define _touch_lambda_2(name) \
  [ts](float* out, float const* inn) { \
    name(ts[0], ts[1], out, inn); \
  }
#define _touch_lambda_3(name) \
  [ts](float* out, float const* inn) { \
    name(ts[0], ts[1], ts[2], out, inn); \
  }
#define _touch_lambda_4(name) \
  [ts](float* out, float const* inn) { \
    name(ts[0], ts[1], ts[2], ts[3], out, inn); \
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
#define _touch_dispatch(i) \
  [&]() -> std::function<void(float*, float const*)> { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_##i ( touch##i##_add); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_##i ( touch##i##_mul); \
      } else if(c == castable_t::min) { \
        return _touch_lambda_##i ( touch##i##_min); \
      } else if(c == castable_t::max) { \
        return  _touch_lambda_##i ( touch##i##_max); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_##i ( touch##i##_none); \
    } \
  }()

std::function<void(float*, float const*)>
build_touch(touch_t const& touch)
{
  auto const& ts = touch.selection;
  if(ts.size() == 1) {
    return _touch_dispatch(1);
  }

  if(ts.size() == 2) {
    return _touch_dispatch(2);
  }

  if(ts.size() == 3) {
    return _touch_dispatch(3);
  }

  if(ts.size() == 4) {
    return _touch_dispatch(4);
  }

  throw std::runtime_error("touch kernel not implemented");
}
