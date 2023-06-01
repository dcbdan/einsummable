#include "kernels.h"
#include "cuda_kernels.h"

void touch1(touchdim_t const& t0, float* out,
            float const* inn, cudaStream_t stream,
            int choice) {
  touch1_dispatch(out, inn, t0.offset_inn,
  t0.offset_out, t0.size, t0.d_inn, t0.d_out,
  stream, choice);
}

void touch2(touchdim_t const& t0, touchdim_t const& t1,
            float* out, float const* inn, cudaStream_t stream,
            int choice) {
  touch2_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t0.offset_out, t1.offset_out,
  t0.size, t1.size, t1.d_inn, t1.d_out,
  stream, choice);
}

void touch3(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, float* out, float const* inn,
            cudaStream_t stream, int choice) {
  touch3_dispatch(out, inn, t0.offset_inn,
  t1.offset_inn, t2.offset_inn, t0.offset_out,
  t1.offset_out, t2.offset_out,t0.size,
  t1.size, t2.size, t1.d_inn, t1.d_out,
  t2.d_inn, t2.d_out, stream, choice);
}

void touch4(touchdim_t const& t0, touchdim_t const& t1,
            touchdim_t const& t2, touchdim_t const& t3,
            float* out, float const* inn, cudaStream_t stream,
            int choice) {
  touch4_dispatch(out, inn, t0.offset_inn, t1.offset_inn,
  t2.offset_inn, t3.offset_inn,t0.offset_out, t1.offset_out,
  t2.offset_out, t3.offset_out,t0.size, t1.size, t2.size,
  t3.size, t1.d_inn, t1.d_out, t2.d_inn, t2.d_out, t3.d_inn,
  t3.d_out, stream, choice);
}

#define _touch_lambda_1(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch1(ts[0], out, inn, stream, choice); \
}

#define _touch_lambda_2(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch2(ts[0], ts[1], out, inn, stream, choice); \
}

#define _touch_lambda_3(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch3(ts[0], ts[1], ts[2], out, inn, stream, choice); \
}

#define _touch_lambda_4(choice) \
  [ts](cudaStream_t stream, float* out, const float* inn) -> void { \
    touch4(ts[0], ts[1], ts[2], ts[3],out, inn, stream, choice); \
}


#define _touch_dispatch(i) \
  [&]() -> touch_kernel_t { \
    if(touch.castable) { \
      castable_t const& c = touch.castable.value(); \
      if(c == castable_t::add) { \
        return _touch_lambda_##i(1); \
      } else if(c == castable_t::mul) { \
        return _touch_lambda_##i(2); \
      } else if(c == castable_t::min) { \
        return _touch_lambda_##i(3); \
      } else if(c == castable_t::max) { \
        return  _touch_lambda_##i(4); \
      } else { \
        throw std::runtime_error("castable should not reach"); \
      } \
    } else { \
      return _touch_lambda_##i(0); \
    } \
  }()

touch_kernel_t build_touch(touch_t const& touch)
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

cutensor_kernel_t
build_einsummable(einsummable_t const& e)
{
  if(is_contraction(e)) {
    throw std::runtime_error("build_einsummable must not be given a constraction");
  }

  string err_msg =
    "could not build a kernel for einsummable_t: " + write_with_ss(e);

  // is this something cutensor reduction can do?
  if(e.has_aggregation()) {
    if(e.inns.size() == 1 && e.castable.value() == castable_t::add) {
      vector<int> const& inn_modes = e.inns[0];

      auto inn_shape = e.inn_shapes()[0];

      vector<int> out_modes(e.out_rank);
      std::iota(out_modes.begin(), out_modes.end(), 0);

      auto out_shape = e.out_shape();

      return build_cutensor_reduction(
        inn_modes, inn_shape,
        out_modes, out_shape);
    }
    // if something has an aggregation but isn't either contraction
    // or a cutensor reduction, then there is no kernel for it
    throw std::runtime_error(err_msg);
  }

  // is this something that cutensor elementwise can do?
  auto maybe_cutensor_ew = make_cutensor_elementwise_op(e);
  if(maybe_cutensor_ew) {
    return build_cutensor_elementwise(maybe_cutensor_ew.value());
  }

  // is this straight elementwise?
  if(e.is_straight_elementwise()) {
    return build_straight_elementwise(e.join, product(e.join_shape));
  }

  throw std::runtime_error(err_msg);
}

void build_contraction(
  cutensorContractionDescriptor_t* desc,
  einsummable_t const& e)
{
  if(!is_contraction(e)) {
    throw std::runtime_error("build_contraction must be given a contraction");
  }
  // TODO
}

void execute_contraction(
  cudaStream_t,
  cutensorHandle_t const*,
  cutensorContractionDescriptor_t const*,
  float* out,
  float const* lhs,
  float const* rhs)
{
  // TODO
}

bool is_contraction(einsummable_t const& e)
{
  return e.inns.size() == 2               &&
    e.has_aggregation()                   &&
    e.castable.value() == castable_t::add &&
    e.join.is_mul();
}

cutensor_kernel_t
build_cutensor_reduction(
  vector<int> inn_modes, vector<uint64_t> inn_shape,
  vector<int> out_modes, vector<uint64_t> out_shape)
{
  // TODO
  return {};
}

cutensor_kernel_t
build_cutensor_elementwise(cutensor_elementwise_op_t op)
{
  // TODO
  return {};
}

optional<cutensor_elementwise_op_t>
make_cutensor_elementwise_op(
  einsummable_t const& e)
{
  // TODO
  return std::nullopt;
}

cutensor_kernel_t
build_straight_elementwise(
  scalarop_t op,
  uint64_t size)
{
  // TODO: dispatch to canned elementwise kernels here
  return {};
}
