#include "../src/einsummable/dbuffer.h"
#include "../src/einsummable/reference.h"
#include "../src/einsummable/einsummable.h"
#include "../src/execution/cpu/kernels.h"

#include <mkl_cblas.h>
#include <mkl.h>

void test_mm(dtype_t dtype, uint64_t i = 5, uint64_t j = 6, uint64_t k = 7) {
  // ij,jk->ik

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

void test_bmm(dtype_t dtype) {
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

void test_unary(scalarop_t scalarop) {
  //DOUT(scalarop);

  auto inn_dtype = scalarop.inn_dtype(0).value();
  auto out_dtype = scalarop.out_dtype();

  uint64_t a = 3;
  uint64_t b = 4;

  einsummable_t e = einsummable_t (
    {a, b},
    { { 0, 1 } },
    2,
    scalarop);

  dbuffer_t inn = make_dbuffer(inn_dtype, a*b);
  dbuffer_t out = make_dbuffer(out_dtype, a*b);

  inn.random();

  dbuffer_t out_ref = reference_einsummable(e, {inn});

  auto f = build_einsummable(e);
  f(out.raw(), { inn.raw() });

  if(!is_close(out_ref, out)) {
    DOUT(scalarop);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("test_unary fail");
  }
}

void test_straight_binary(scalarop_t scalarop) {
  auto lhs_dtype = scalarop.inn_dtype(0).value();
  auto rhs_dtype = scalarop.inn_dtype(1).value();
  auto out_dtype = scalarop.out_dtype();

  uint64_t a = 3;
  uint64_t b = 4;
  uint64_t c = 2;

  einsummable_t e = einsummable_t (
    {a, b, c},
    { { 0,1,2 }, { 0,1,2} },
    3,
    scalarop);

  try {
    auto f = build_einsummable(e);
  } catch(...) {
    DOUT(scalarop);
    return;
  }

  dbuffer_t lhs = make_dbuffer(lhs_dtype, a*b*c);
  dbuffer_t rhs = make_dbuffer(rhs_dtype, a*b*c);
  dbuffer_t out = make_dbuffer(out_dtype, a*b*c);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(e, {lhs,rhs});

  auto f = build_einsummable(e);
  f(out.raw(), { lhs.raw(), rhs.raw() });

  if(!is_close(out_ref, out)) {
    DOUT(scalarop);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("test binary straight fail");
  }
}

void test_ab_a_binary(scalarop_t scalarop) {
  auto lhs_dtype = scalarop.inn_dtype(0).value();
  auto rhs_dtype = scalarop.inn_dtype(1).value();
  auto out_dtype = scalarop.out_dtype();

  uint64_t a = 5;
  uint64_t b = 8;
  uint64_t c = 3;

  einsummable_t e = einsummable_t (
    {a, b, c},
    { { 0,1,2 }, { 0,1} },
    3,
    scalarop);

  try {
    auto f = build_einsummable(e);
  } catch(...) {
    DOUT(scalarop);
    return;
  }

  dbuffer_t lhs = make_dbuffer(lhs_dtype, a*b*c);
  dbuffer_t rhs = make_dbuffer(rhs_dtype, a*b  );
  dbuffer_t out = make_dbuffer(out_dtype, a*b*c);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(e, {lhs,rhs});

  auto f = build_einsummable(e);
  f(out.raw(), { lhs.raw(), rhs.raw() });

  if(!is_close(out_ref, out)) {
    DOUT(scalarop);
    DOUT(lhs);
    DOUT(rhs);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("test ab a binary fail");
  }
}

void test_ab_b_binary(scalarop_t scalarop) {
  auto lhs_dtype = scalarop.inn_dtype(0).value();
  auto rhs_dtype = scalarop.inn_dtype(1).value();
  auto out_dtype = scalarop.out_dtype();

  uint64_t a = 3;
  uint64_t b = 4;
  uint64_t c = 2;

  einsummable_t e = einsummable_t (
    {a, b, c},
    { { 0,1,2 }, { 2 } },
    3,
    scalarop);

  try {
    auto f = build_einsummable(e);
  } catch(...) {
    DOUT(scalarop);
    return;
  }

  dbuffer_t lhs = make_dbuffer(lhs_dtype, a*b*c);
  dbuffer_t rhs = make_dbuffer(rhs_dtype,     c);
  dbuffer_t out = make_dbuffer(out_dtype, a*b*c);

  lhs.random();
  rhs.random();

  dbuffer_t out_ref = reference_einsummable(e, {lhs,rhs});

  auto f = build_einsummable(e);
  f(out.raw(), { lhs.raw(), rhs.raw() });

  if(!is_close(out_ref, out)) {
    DOUT(scalarop);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("test ab b binary fail");
  }
}

void test_binary(scalarop_t scalarop) {
  test_straight_binary(scalarop);
  test_ab_a_binary(scalarop);
  test_ab_b_binary(scalarop);
}

void test_reduction_ab_a(dtype_t dtype, castable_t castable)
{
  uint64_t a = 10;
  uint64_t b = 4;

  einsummable_t e = einsummable_t (
    {a, b},
    { { 0,1 } },
    1,
    scalarop_t::make_identity(dtype),
    castable);

  dbuffer_t inn = make_dbuffer(dtype, a*b);
  dbuffer_t out = make_dbuffer(dtype, a  );

  inn.random();
  out.zeros();

  dbuffer_t out_ref = reference_einsummable(e, {inn});

  auto f = build_einsummable(e);
  f(out.raw(), { inn.raw() });

  if(!is_close(out_ref, out)) {
    DOUT(dtype);
    DOUT(castable);
    DOUT(out_ref);
    DOUT(out);
    throw std::runtime_error("test reduction ab a fail");
  }
}

void test_a_contraction() {
  // es[1,32,1,128,9]abce,aebd->abcd
  dtype_t dtype = dtype_t::f16;

  einsummable_t e(
    //{1, 32, 1, 128, 9},
    {1, 32, 1, 128, 9},
    { {0,1,2,4}, {0,4,1,3} },
    4,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  kernel_manager_t km;
  uint64_t workspace_size = km.build(e).value();

  auto inn_shapes = e.inn_shapes();
  dbuffer_t lhs = make_dbuffer(dtype, product(inn_shapes[0]));
  dbuffer_t rhs = make_dbuffer(dtype, product(inn_shapes[1]));

  lhs.random();
  rhs.random();

  buffer_t workspace_buffer = make_buffer(workspace_size);

  dbuffer_t out_ref = reference_einsummable(e, {lhs, rhs});

  dbuffer_t out = make_dbuffer(dtype, product(e.out_shape()));

  optional<tuple<void*, uint64_t>> workspace = tuple<void*, uint64_t>{
    workspace_buffer->raw(), workspace_size };

  km(e, out.raw(), { lhs.raw(), rhs.raw() }, workspace);

  DOUT(out_ref.sum_to_f64());
  DOUT(out.sum_to_f64());
}

void test_half_mm(uint64_t ni, uint64_t nj, uint64_t nk,
  bool trans_lhs = false, bool trans_rhs = false)
{
  using f16_t = MKL_F16;
  // Use the half library (float16_t) to set one and zero,
  // then convert it to whatever type mkl sets MKL_F16 to.
  float16_t one_(1.0);
  float16_t zero_(0.0);
  f16_t& one  = reinterpret_cast<f16_t&>(one_);
  f16_t& zero = reinterpret_cast<f16_t&>(zero_);

  dbuffer_t out = make_dbuffer(dtype_t::f16, ni*nk);

  dbuffer_t lhs = make_dbuffer(dtype_t::f16, ni*nj);
  dbuffer_t rhs = make_dbuffer(dtype_t::f16, nj*nk);

  lhs.random();
  rhs.random();

  cblas_hgemm(
    CblasRowMajor,
    trans_lhs ? CblasTrans : CblasNoTrans,
    trans_rhs ? CblasTrans : CblasNoTrans,
    ni,nk,nj,
    one,
    (f16_t const*)lhs.raw(),
    trans_lhs ? ni : nj,
    (f16_t const*)rhs.raw(),
    trans_rhs ? nj : nk,
    zero,
    (f16_t*)out.raw(),
    nk);

  einsummable_t e = einsummable_t::from_matmul(ni,nj,nk, dtype_t::f16);
  dbuffer_t out_ref = reference_einsummable(e, {lhs, rhs});

  DOUT(out_ref.sum_to_f64());
  DOUT(out.sum_to_f64());
}

int main() {
  //mkl_set_num_threads(1);
  //test_half_mm(1, 400, 33); // ijk: ij,jk->ik
  //test_half_mm(1, 400, 32); // ijk: ij,jk->ik

  //test_half_mm(33, 400, 1, false, true);
  //test_a_contraction();
  //DOUT("f16");
  //test_mm(dtype_t::f16, 1, 2, 128); // Fail
  //test_mm(dtype_t::f16, 1, 2, 100); // Fail
  //test_mm(dtype_t::f16, 1, 2, 64);  // Fail
  //test_mm(dtype_t::f16, 1, 2, 60);  // Fail
  //test_mm(dtype_t::f16, 1, 2, 33);
  //test_mm(dtype_t::f16, 1, 2, 32);
  //test_mm(dtype_t::f16, 1, 2, 24);
  //test_mm(dtype_t::f16, 1, 2, 18);
  //DOUT("f32");
  //test_mm(dtype_t::f32, 1, 2, 128);
  //test_mm(dtype_t::f32, 1, 2, 18);

  //test_bmm(dtype_t::f16);
  //return 0;

  //test_mm(dtype_t::f16);
  test_mm(dtype_t::f32);
  test_mm(dtype_t::f64);
  test_mm(dtype_t::c64);

  //test_bmm(dtype_t::f16);
  test_bmm(dtype_t::f32);
  test_bmm(dtype_t::f64);
  test_bmm(dtype_t::c64);

  test_unary(scalarop_t::make_relu(dtype_t::f64));
  test_unary(scalarop_t::make_silu(dtype_t::f64));
  test_unary(scalarop_t::make_increment(scalar_t(double(1.965))));
  test_unary(scalarop_t::make_scale(scalar_t(double(1.965))));

  test_unary(scalarop_t::make_relu(dtype_t::f32));
  test_unary(scalarop_t::make_silu(dtype_t::f32));
  test_unary(scalarop_t::make_increment(scalar_t(float(1.965))));
  test_unary(scalarop_t::make_scale(scalar_t(float(1.965))));

  test_binary(scalarop_t::make_add(dtype_t::f16));
  test_binary(scalarop_t::make_sub(dtype_t::f16));
  test_binary(scalarop_t::make_mul(dtype_t::f16));
  test_binary(scalarop_t::make_min(dtype_t::f16));
  test_binary(scalarop_t::make_max(dtype_t::f16));

  test_binary(scalarop_t::make_add(dtype_t::f32));
  test_binary(scalarop_t::make_sub(dtype_t::f32));
  test_binary(scalarop_t::make_mul(dtype_t::f32));
  test_binary(scalarop_t::make_min(dtype_t::f32));
  test_binary(scalarop_t::make_max(dtype_t::f32));

  test_binary(scalarop_t::make_add(dtype_t::f64));
  test_binary(scalarop_t::make_sub(dtype_t::f64));
  test_binary(scalarop_t::make_mul(dtype_t::f64));
  test_binary(scalarop_t::make_min(dtype_t::f64));
  test_binary(scalarop_t::make_max(dtype_t::f64));

  test_reduction_ab_a(dtype_t::f16, castable_t::add);
  test_reduction_ab_a(dtype_t::f16, castable_t::mul);
  test_reduction_ab_a(dtype_t::f16, castable_t::min);
  test_reduction_ab_a(dtype_t::f16, castable_t::max);

  test_reduction_ab_a(dtype_t::f32, castable_t::add);
  test_reduction_ab_a(dtype_t::f32, castable_t::mul);
  test_reduction_ab_a(dtype_t::f32, castable_t::min);
  test_reduction_ab_a(dtype_t::f32, castable_t::max);

  test_reduction_ab_a(dtype_t::f64, castable_t::add);
  test_reduction_ab_a(dtype_t::f64, castable_t::mul);
  test_reduction_ab_a(dtype_t::f64, castable_t::min);
  test_reduction_ab_a(dtype_t::f64, castable_t::max);

  test_reduction_ab_a(dtype_t::c64, castable_t::add);
  test_reduction_ab_a(dtype_t::c64, castable_t::mul);
}
