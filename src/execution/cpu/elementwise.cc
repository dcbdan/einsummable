#include "elementwise.h"

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

std::function<void(float*,vector<float*>)>
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto f = op.build_unary();

  auto f_part = [f](float* out, float* inn, uint64_t n) {
    for(uint64_t i = 0; i != n; ++i) {
      out[i] = f(inn[i]);
    }
  };

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f_part, ranges](float* out, vector<float*> inns) {
    vector<std::thread> ts;
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f_part,
        out     + lower,
        inns[0] + lower,
        upper-lower);
    }
    for(auto& t: ts) {
      t.join();
    }
  };
}

std::function<void(float*,vector<float*>)>
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t op)
{
  if(n == 0) {
    throw std::runtime_error("elementwise: calling with zero");
  }

  auto f = op.build_binary();

  if(num_threads == 1) {
    return [f,n](float* out, vector<float*> inns) {
      auto const& lhs = inns[0];
      auto const& rhs = inns[1];
      for(int i = 0; i != n; ++i) {
        out[i] = f(lhs[i], rhs[i]);
      }
    };
  }

  auto f_part = [f](float* out, float* lhs, float* rhs, uint64_t n) {
    for(uint64_t i = 0; i != n; ++i) {
      out[i] = f(lhs[i], rhs[i]);
    }
  };

  auto ranges = _zip_parts(divide_evenly(num_threads, n));

  return [f_part, ranges](float* out, vector<float*> inns) {
    vector<std::thread> ts;
    for(auto const& [lower,upper]: ranges) {
      ts.emplace_back(
        f_part,
        out     + lower,
        inns[0] + lower,
        inns[1] + lower,
        upper-lower);
    }
    for(auto& t: ts) {
      t.join();
    }
  };
}
