#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/reference.h"
#include "../src/einsummable/einsummable.h"
#include "../src/execution/cpu/kernels.h"

void main_mm(dtype_t dtype) {
  // ij,jk->ik
  uint64_t i = 5;
  uint64_t j = 6;
  uint64_t k = 7;

  einsummable_t matmul = einsummable_t::from_matmul(i,j,k, dtype);

  dbuffer_t lhs = make_dbuffer(dtype, i*j);
  dbuffer_t rhs = make_dbuffer(dtype, j*k);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  dbuffer_t out = make_dbuffer(dtype, i*k);
  auto f = build_einsummable(matmul);
  f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("MM ARE NOT CLOSE!");
  }
}

void main_bmm(dtype_t dtype) {
  // bij,jk->bik
  // 013 32  012
  uint64_t b = 3;
  uint64_t i = 5;
  uint64_t j = 6;
  uint64_t k = 7;

  einsummable_t matmul = einsummable_t(
    {b, i, k, j},
    { {0, 1, 3}, {3, 2} },
    3,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  dbuffer_t lhs = make_dbuffer(dtype, b*i*j);
  dbuffer_t rhs = make_dbuffer(dtype, j*k);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(matmul, {lhs, rhs});

  dbuffer_t out = make_dbuffer(dtype, b*i*k);
  auto f = build_einsummable(matmul);
  f(out.ptr(), {lhs.ptr(), rhs.ptr()});

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("BMM ARE NOT CLOSE!");
  }
}

int main() {
  main_mm(dtype_t::f16);
  main_mm(dtype_t::f32);
  main_mm(dtype_t::f64);
  main_mm(dtype_t::c64);

  main_bmm(dtype_t::f16);
  main_bmm(dtype_t::f32);
  main_bmm(dtype_t::f64);
  main_bmm(dtype_t::c64);
}
