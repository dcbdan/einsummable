#include "dbuffer.h"

dbuffer_t::dbuffer_t()
  : dtype(default_dtype()),
    data(nullptr)
{}

dbuffer_t::dbuffer_t(dtype_t d, buffer_t b)
  : dtype(d), data(b)
{
  if(data->size % dtype_size(dtype) != 0) {
    throw std::runtime_error("invalid dbuffer data size");
  }
}

void dbuffer_t::zeros() {
  fill(scalar_t::zero(dtype));
}

void dbuffer_t::ones() {
  scalar_t val;
  if(dtype == dtype_t::c64) {
    val = scalar_t(std::complex<float>(1.0, 1.0));
  } else {
    val = scalar_t::one(dtype);
  }

  fill(val);
}

void dbuffer_t::fill(scalar_t val) {
  if(dtype == dtype_t::f16) {
    auto ptr = f16();
    std::fill(ptr, ptr + nelem(), val.f16());
  } else if(dtype == dtype_t::f32) {
    auto ptr = f32();
    std::fill(ptr, ptr + nelem(), val.f32());
  } else if(dtype == dtype_t::f64) {
    auto ptr = f64();
    std::fill(ptr, ptr + nelem(), val.f64());
  } else if(dtype == dtype_t::c64) {
    auto ptr = c64();
    std::fill(ptr, ptr + nelem(), val.c64());
  } else {
    throw std::runtime_error("should not reach fill");
  }
}

void dbuffer_t::iota(int start) {
  if(dtype == dtype_t::f16) {
    auto ptr = f16();
    std::iota(ptr, ptr + nelem(), start*1.0);
  } else if(dtype == dtype_t::f32) {
    auto ptr = f32();
    std::iota(ptr, ptr + nelem(), start*1.0);
  } else if(dtype == dtype_t::f64) {
    auto ptr = f64();
    std::iota(ptr, ptr + nelem(), start*1.0);
  } else if(dtype == dtype_t::c64) {
    auto ptr = c64();
    std::iota(ptr, ptr + nelem(), start*1.0);
  } else {
    throw std::runtime_error("should not reach fill");
  }
}

void dbuffer_t::random(string b, string e) {
  if(dtype == dtype_t::c64) {
    view_c64_as_f32().random(b,e);
  } else {
    random(scalar_t(dtype, b), scalar_t(dtype, e));
  }
}

dbuffer_t dbuffer_t::copy() {
  auto ret = make_dbuffer(dtype, nelem());

  if(dtype == dtype_t::f16) {
    auto inn = f16();
    auto out = ret.f16();
    std::copy(inn, inn + nelem(), out);
  } else if(dtype == dtype_t::f32) {
    auto inn = f32();
    auto out = ret.f32();
    std::copy(inn, inn + nelem(), out);
  } else if(dtype == dtype_t::f64) {
    auto inn = f64();
    auto out = ret.f64();
    std::copy(inn, inn + nelem(), out);
  } else if(dtype == dtype_t::c64) {
    auto inn = c64();
    auto out = ret.c64();
    std::copy(inn, inn + nelem(), out);
  } else {
    throw std::runtime_error("should not reach fill");
  }

  return ret;
}

template <typename T>
void _uniform_random_fill(
  T lower,
  T upper,
  T* data,
  uint64_t size)
{
  std::uniform_real_distribution<T> dist(lower, upper);
  auto gen = [&dist]{ return dist(random_gen()); };
  std::generate(data, data + size, gen);
}

template <>
void _uniform_random_fill(
  float16_t lower,
  float16_t upper,
  float16_t* data,
  uint64_t size)
{
  std::uniform_real_distribution<float> dist(lower, upper);
  auto gen = [&dist]{ return float16_t(dist(random_gen())); };
  std::generate(data, data + size, gen);
}

void dbuffer_t::random(scalar_t lower, scalar_t upper) {
  if(lower.dtype != dtype || upper.dtype != dtype || dtype == dtype_t::c64) {
    throw std::runtime_error("msut be float dtype; scalars must be same");
  }

  if(dtype == dtype_t::f16) {
    _uniform_random_fill(lower.f16(), upper.f16(), f16(), nelem());
  } else if(dtype == dtype_t::f32) {
    _uniform_random_fill(lower.f32(), upper.f32(), f32(), nelem());
  } else if(dtype == dtype_t::f64) {
    _uniform_random_fill(lower.f64(), upper.f64(), f64(), nelem());
  } else {
    throw std::runtime_error("should not reach");
  }
}

dbuffer_t dbuffer_t::view_c64_as_f32() {
  if(dtype != dtype_t::c64) {
    throw std::runtime_error("expect c64");
  }
  return dbuffer_t(dtype_t::f32, data);
}

dbuffer_t dbuffer_t::view_f32_as_c64() {
  if(dtype != dtype_t::f32) {
    throw std::runtime_error("expect f32");
  }
  if(nelem() % 2 != 0) {
    throw std::runtime_error("must have even number of elems");
  }
  return dbuffer_t(dtype_t::c64, data);
}

scalar_t dbuffer_t::sum() const {
  if(dtype == dtype_t::f16) {
    return scalar_t(
      std::accumulate(f16(), f16() + nelem(), float16_t(0.0)));
  } else if(dtype == dtype_t::f32) {
    return scalar_t(
      std::accumulate(f32(), f32() + nelem(), float(0.0)));
  } else if(dtype == dtype_t::f64) {
    return scalar_t(
      std::accumulate(f64(), f64() + nelem(), double(0.0)));
  } else if(dtype == dtype_t::c64) {
    return scalar_t(
      std::accumulate(c64(), c64() + nelem(), std::complex<float>(0.0, 0.0)));
  } else {
    throw std::runtime_error("should not reach");
  }
}

uint64_t dbuffer_t::nelem() const {
  if(size() % dtype_size(dtype) != 0) {
    throw std::runtime_error("incorrect size for dtype");
  }
  return size() / dtype_size(dtype);
}
uint64_t const& dbuffer_t::size() const {
  return data->size;
}

void dbuffer_t::set(uint64_t which_elem, scalar_t const& val) {
  if(dtype != val.dtype) {
    throw std::runtime_error("invalid dtype");
  }
  if(dtype == dtype_t::f16) {
    f16()[which_elem] = val.f16();
  } else if(dtype == dtype_t::f32) {
    f32()[which_elem] = val.f32();
  } else if(dtype == dtype_t::f64) {
    f64()[which_elem] = val.f64();
  } else if(dtype == dtype_t::c64) {
    c64()[which_elem] = val.c64();
  } else {
    throw std::runtime_error("should not reach");
  }
}

void dbuffer_t::agg_into(uint64_t which_elem, castable_t castable, scalar_t const& val) {
  scalarop_t op = scalarop_t::make_from_castable(castable, dtype);
  set(
    which_elem,
    op.eval({get(which_elem), val}));
}

scalar_t dbuffer_t::get(uint64_t which_elem) const {
  if(dtype == dtype_t::f16) {
    return scalar_t(f16()[which_elem]);
  } else if(dtype == dtype_t::f32) {
    return scalar_t(f32()[which_elem]);
  } else if(dtype == dtype_t::f64) {
    return scalar_t(f64()[which_elem]);
  } else if(dtype == dtype_t::c64) {
    return scalar_t(c64()[which_elem]);
  } else {
    throw std::runtime_error("should not reach");
  }
}

void      * dbuffer_t::ptr()       { return data->data; }
void const* dbuffer_t::ptr() const { return data->data; }

float16_t* dbuffer_t::f16() {
  if(dtype != dtype_t::f16) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<float16_t*>(data->data);
}
float* dbuffer_t::f32() {
  if(dtype != dtype_t::f32) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<float*>(data->data);
}
double* dbuffer_t::f64() {
  if(dtype != dtype_t::f64) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<double*>(data->data);
}
std::complex<float>* dbuffer_t::c64() {
  if(dtype != dtype_t::c64) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<std::complex<float>*>(data->data);
}

float16_t const* dbuffer_t::f16() const {
  if(dtype != dtype_t::f16) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<float16_t*>(data->data);
}
float const* dbuffer_t::f32() const {
  if(dtype != dtype_t::f32) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<float*>(data->data);
}
double const* dbuffer_t::f64() const {
  if(dtype != dtype_t::f64) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<double*>(data->data);
}
std::complex<float> const* dbuffer_t::c64() const {
  if(dtype != dtype_t::c64) { throw std::runtime_error("incroect dtype"); }
  return reinterpret_cast<std::complex<float>*>(data->data);
}

dbuffer_t make_dbuffer(dtype_t dtype, uint64_t num_elems) {
  return dbuffer_t(dtype, make_buffer(dtype_size(dtype) * num_elems));
}

bool is_close(dbuffer_t const& ll, dbuffer_t const& rr, float eps) {
  if(ll.dtype != rr.dtype || ll.nelem() != rr.nelem()) {
    return false;
  }

  uint64_t n = ll.nelem();
  auto const& dtype = ll.dtype;
  if(dtype == dtype_t::f16) {
    auto lhs = ll.f16();
    auto rhs = rr.f16();
    for(int i = 0; i != n; ++i) {
      if(!is_close(lhs[i], rhs[i], eps)){ return false; }
    }
  } else if(dtype == dtype_t::f32) {
    auto lhs = ll.f32();
    auto rhs = rr.f32();
    for(int i = 0; i != n; ++i) {
      if(!is_close(lhs[i], rhs[i], eps)){ return false; }
    }
  } else if(dtype == dtype_t::f64) {
    auto lhs = ll.f64();
    auto rhs = rr.f64();
    for(int i = 0; i != n; ++i) {
      if(!is_close(lhs[i], rhs[i], eps)){ return false; }
    }
  } else if(dtype == dtype_t::c64) {
    auto lhs = ll.c64();
    auto rhs = rr.c64();
    for(int i = 0; i != n; ++i) {
      if(!is_close(lhs[i].real(), rhs[i].real(), eps)){ return false; }
      if(!is_close(lhs[i].imag(), rhs[i].imag(), eps)){ return false; }
    }
  } else {
    throw std::runtime_error("should not reach fill");
  }

  return true;
}

template <typename T>
void _print_elems(std::ostream& out, T const* v, int n) {
  if(n > 0) {
    out << v[0];
    for(int i = 1; i != n; ++i) {
      out << "," << v[i];
    }
  }
}

std::ostream& operator<<(std::ostream& out, dbuffer_t const& dbuffer)
{
  auto const& dtype = dbuffer.dtype;
  uint64_t n = dbuffer.nelem();
  out << "dbuffer[" << dtype << "|" << n << "]{";
  if(dtype == dtype_t::f16) {
    _print_elems(out, dbuffer.f16(), n);
  } else if(dtype == dtype_t::f32) {
    _print_elems(out, dbuffer.f32(), n);
  } else if(dtype == dtype_t::f64) {
    _print_elems(out, dbuffer.f64(), n);
  } else if(dtype == dtype_t::c64) {
    _print_elems(out, dbuffer.c64(), n);
  } else {
    throw std::runtime_error("should not reach");
  }
  out << "}";
  return out;
}

bool operator==(dbuffer_t const& lhs, dbuffer_t const& rhs) {
  return lhs.dtype == rhs.dtype && lhs.data == rhs.data;
}
bool operator!=(dbuffer_t const& lhs, dbuffer_t const& rhs) {
  return !(lhs == rhs);
}

