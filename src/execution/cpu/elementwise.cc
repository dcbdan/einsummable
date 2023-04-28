#include "elementwise.h"

void print_elementwise_function(scalarop_t op) {
  auto write_hole = [](int i) {
    return "x" + write_with_ss(i) + "[i]";
  };

  using std::cout;
  using std::endl;

  cout << "// " << op << endl;
  cout << "void f(" << endl
       << "  uint64_t n," << endl;
  int n = op.num_inputs();
  if(n == 0) {
  cout << "  float* out" << endl;
  } else {
    cout << "  float* out, " << endl;
    int i;
    for(i = 0; i != n-1; ++i) {
      cout << "  float const* x" << i << "," << endl;
    }
    cout << "  float const* x" << i << endl;
  }
  cout << ") {" << endl;
  cout << "  for(uint64_t i = 0; i != n; ++i) {" << endl;
  cout << "    out[i] = " << op.to_cppstr(write_hole) << ";" << endl;
  cout << "  }" << endl;
  cout << "}" << endl;
}

// *[ite_<[hole@0,constant{0},constant{0},constant{1}],hole@1]
void b0(
  uint64_t n,
  float* out,
  float const* x0,
  float const* x1
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = ((x0[i]<0?0:1)*x1[i]);
  }
}
// +[hole@0,*[*[hole@1,constant{0.3}],constant{-1}]]
void b1(
  uint64_t n,
  float* out,
  float const* x0,
  float const* x1
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]+((x1[i]*0.3)*-1));
  }
}
// +[hole@0,*[hole@1,constant{-1}]]
void b2(
  uint64_t n,
  float* out,
  float const* x0,
  float const* x1
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]+(x1[i]*-1));
  }
}
// +[hole@0,hole@1]
void b3(
  uint64_t n,
  float* out,
  float const* x0,
  float const* x1
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]+x1[i]);
  }
}
// ite_<[hole@0,constant{0},constant{0},hole@0]
void u0(
  uint64_t n,
  float* out,
  float const* x0
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]<0?0:x0[i]);
  }
}
// +[hole@0,*[*[hole@1,constant{0.1}],constant{-1}]]
void b4(
  uint64_t n,
  float* out,
  float const* x0,
  float const* x1
) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]+((x1[i]*0.1)*-1));
  }
}

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
