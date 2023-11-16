#include "touch.h"

#define _touch1_frame(name, type_t, inner_ops) \
  void name(touchdim_t const& t0, type_t* out, type_t const* inn) { \
    out += t0.offset_out; \
    inn += t0.offset_inn; \
    inner_ops \
  }

#define _touch2_frame(name, type_t, inner_ops) \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    type_t* out, \
    type_t const* inn) \
  { \
    out += t0.offset_out*t1.d_out + t1.offset_out; \
    inn += t0.offset_inn*t1.d_inn + t1.offset_inn; \
    for(uint64_t i0 = 0; i0 != t0.size; ++i0) { \
      inner_ops \
      out += t1.d_out; \
      inn += t1.d_inn; \
    } \
  }

#define _touch3_frame(name, type_t, inner_ops) \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    type_t* out, \
    type_t const* inn) \
  { \
    out += t0.offset_out*t1.d_out*t2.d_out + t1.offset_out*t2.d_out + t2.offset_out; \
    inn += t0.offset_inn*t1.d_inn*t2.d_inn + t1.offset_inn*t2.d_inn + t2.offset_inn; \
    for(uint64_t i0 = 0; i0 != t0.size; ++i0) { \
      for(uint64_t i1 = 0; i1 != t1.size; ++i1) { \
        inner_ops \
        out += t2.d_out; \
        inn += t2.d_inn; \
      } \
      out += t2.d_out * (t1.d_out - t1.size); \
      inn += t2.d_inn * (t1.d_inn - t1.size); \
    } \
  }

#define _touch4_frame(name, type_t, inner_ops) \
  void name(\
    touchdim_t const& t0, \
    touchdim_t const& t1, \
    touchdim_t const& t2, \
    touchdim_t const& t3, \
    type_t* out, \
    type_t const* inn) \
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
          inner_ops \
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

#define _touch1_line(name, op) \
  template <typename T> \
  _touch1_frame(name, T, \
    for(uint64_t i = 0; i != t0.size; ++i) { \
      op; \
    })
#define _touch2_line(name, op) \
  template <typename T> \
  _touch2_frame(name, T, \
    for(uint64_t i = 0; i != t1.size; ++i) { \
      op; \
    })
#define _touch3_line(name, op) \
  template <typename T> \
  _touch3_frame(name, T, \
    for(uint64_t i = 0; i != t2.size; ++i) { \
      op; \
    })
#define _touch4_line(name, op) \
  template <typename T> \
  _touch4_frame(name, T, \
    for(uint64_t i = 0; i != t3.size; ++i) { \
      op; \
    })

template <typename T>
_touch1_frame(touch1_none, T,
  std::memcpy(
    reinterpret_cast<void*>(out),
    reinterpret_cast<void const*>(inn),
    sizeof(T)*t0.size);
)

template <typename T>
_touch2_frame(touch2_none, T,
  std::memcpy(
    reinterpret_cast<void*>(out),
    reinterpret_cast<void const*>(inn),
    sizeof(T)*t1.size);
)

template <typename T>
_touch3_frame(touch3_none, T,
  std::memcpy(
    reinterpret_cast<void*>(out),
    reinterpret_cast<void const*>(inn),
    sizeof(T)*t2.size);
)

template <typename T>
_touch4_frame(touch4_none, T,
  std::memcpy(
    reinterpret_cast<void*>(out),
    reinterpret_cast<void const*>(inn),
    sizeof(T)*t3.size);
)

_touch1_line(touch1_add,  out[i] += inn[i]                  );
_touch1_line(touch1_mul,  out[i] *= inn[i]                  );
_touch1_line(touch1_min,  out[i] =  std::min(out[i], inn[i]));
_touch1_line(touch1_max,  out[i] =  std::max(out[i], inn[i]));

_touch2_line(touch2_add,  out[i] += inn[i]                  );
_touch2_line(touch2_mul,  out[i] *= inn[i]                  );
_touch2_line(touch2_min,  out[i] =  std::min(out[i], inn[i]));
_touch2_line(touch2_max,  out[i] =  std::max(out[i], inn[i]));

_touch3_line(touch3_add,  out[i] += inn[i]                  );
_touch3_line(touch3_mul,  out[i] *= inn[i]                  );
_touch3_line(touch3_min,  out[i] =  std::min(out[i], inn[i]));
_touch3_line(touch3_max,  out[i] =  std::max(out[i], inn[i]));

_touch4_line(touch4_add,  out[i] += inn[i]                  );
_touch4_line(touch4_mul,  out[i] *= inn[i]                  );
_touch4_line(touch4_min,  out[i] =  std::min(out[i], inn[i]));
_touch4_line(touch4_max,  out[i] =  std::max(out[i], inn[i]));

#define _call_touch1(name, T) \
  name(ts[0], \
    reinterpret_cast<T*>(out), \
    reinterpret_cast<T const*>(inn))
#define _call_touch2(name, T) \
  name(ts[0], ts[1], \
    reinterpret_cast<T*>(out), \
    reinterpret_cast<T const*>(inn))
#define _call_touch3(name, T) \
  name(ts[0], ts[1], ts[2], \
    reinterpret_cast<T*>(out), \
    reinterpret_cast<T const*>(inn))
#define _call_touch4(name, T) \
  name(ts[0], ts[1], ts[2], ts[3], \
    reinterpret_cast<T*>(out), \
    reinterpret_cast<T const*>(inn))

#define _call_touch_s(T, f1, f2, f3, f4) \
  if(ts.size() == 1) { \
    _call_touch1(f1, T); \
  } else if(ts.size() == 2) { \
    _call_touch2(f2, T); \
  } else if(ts.size() == 3) { \
    _call_touch3(f3, T); \
  } else if(ts.size() == 4) { \
    _call_touch4(f4, T); \
  } else { \
    throw std::runtime_error("_call_touch_s: not enough dims"); \
  } \
  return;

void execute_touch(
  touch_t const& touch,
  void* out,
  void const* inn)
{
  auto const [ts, maybe_castable, dtype] = touch;

  if(!maybe_castable) {
    if(ts.size() == 1) {
      if(dtype == dtype_t::f16) {
        return _call_touch1(touch1_none, uint16_t);
      } else if(dtype == dtype_t::f32) {
        return _call_touch1(touch1_none, uint32_t);
      } else if(dtype == dtype_t::f64 || dtype == dtype_t::c64) {
        return _call_touch1(touch1_none, uint64_t);
      }
    } else if(ts.size() == 2) {
      if(dtype == dtype_t::f16) {
        return _call_touch2(touch2_none, uint16_t);
      } else if(dtype == dtype_t::f32) {
        return _call_touch2(touch2_none, uint32_t);
      } else if(dtype == dtype_t::f64 || dtype == dtype_t::c64) {
        return _call_touch2(touch2_none, uint64_t);
      }
    } else if(ts.size() == 3) {
      if(dtype == dtype_t::f16) {
        return _call_touch3(touch3_none, uint16_t);
      } else if(dtype == dtype_t::f32) {
        return _call_touch3(touch3_none, uint32_t);
      } else if(dtype == dtype_t::f64 || dtype == dtype_t::c64) {
        return _call_touch3(touch3_none, uint64_t);
      }
    } else if(ts.size() == 4) {
      if(dtype == dtype_t::f16) {
        return _call_touch4(touch4_none, uint16_t);
      } else if(dtype == dtype_t::f32) {
        return _call_touch4(touch4_none, uint32_t);
      } else if(dtype == dtype_t::f64 || dtype == dtype_t::c64) {
        return _call_touch4(touch4_none, uint64_t);
      }
    } else {
      throw std::runtime_error("too many selection dims; try calling simplify");
    }
    throw std::runtime_error("execute_touch: should not reach");
  }

  auto const& castable = maybe_castable.value();
  if(dtype == dtype_t::f16) {
    if(castable == castable_t::add) {
      _call_touch_s(float16_t, touch1_add, touch2_add, touch3_add, touch4_add);
    } else if(castable == castable_t::mul) {
      _call_touch_s(float16_t, touch1_mul, touch2_mul, touch3_mul, touch4_mul);
    } else if(castable == castable_t::min) {
      _call_touch_s(float16_t, touch1_min, touch2_min, touch3_min, touch4_min);
    } else if(castable == castable_t::max) {
      _call_touch_s(float16_t, touch1_max, touch2_max, touch3_max, touch4_max);
    }
  } else if(dtype == dtype_t::f32) {
    if(castable == castable_t::add) {
      _call_touch_s(float, touch1_add, touch2_add, touch3_add, touch4_add);
    } else if(castable == castable_t::mul) {
      _call_touch_s(float, touch1_mul, touch2_mul, touch3_mul, touch4_mul);
    } else if(castable == castable_t::min) {
      _call_touch_s(float, touch1_min, touch2_min, touch3_min, touch4_min);
    } else if(castable == castable_t::max) {
      _call_touch_s(float, touch1_max, touch2_max, touch3_max, touch4_max);
    }
  } else if(dtype == dtype_t::f64) {
    if(castable == castable_t::add) {
      _call_touch_s(double, touch1_add, touch2_add, touch3_add, touch4_add);
    } else if(castable == castable_t::mul) {
      _call_touch_s(double, touch1_mul, touch2_mul, touch3_mul, touch4_mul);
    } else if(castable == castable_t::min) {
      _call_touch_s(double, touch1_min, touch2_min, touch3_min, touch4_min);
    } else if(castable == castable_t::max) {
      _call_touch_s(double, touch1_max, touch2_max, touch3_max, touch4_max);
    }
  } else if(dtype == dtype_t::c64) {
    if(castable == castable_t::add) {
      _call_touch_s(std::complex<float>, touch1_add, touch2_add, touch3_add, touch4_add);
    } else if(castable == castable_t::mul) {
      _call_touch_s(std::complex<float>, touch1_mul, touch2_mul, touch3_mul, touch4_mul);
    }
  }
  throw std::runtime_error("execute_touch: should not reach ");
}
