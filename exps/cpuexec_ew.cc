#include "../src/execution/cpu/kernels.h"

#include "../src/einsummable/reference.h"

//void f(uint64_t n, float* out, float* lhs, float* rhs) {
//  for(uint64_t i = 0; i != n; ++i) {
//    out[i] = lhs[i] + rhs[i];
//  }
//}
void f(uint64_t n, float* out, vector<float*> const& inns) {
  auto const& lhs = inns[0];
  auto const& rhs = inns[1];
  for(uint64_t i = 0; i != n; ++i) {
    //out[i] = lhs[i] + rhs[i];
    out[i] = lhs[i] - 0.1 * rhs[i];
  }
}

void g(uint8_t* d, uint64_t n, float* out, vector<float*> const& inns) {
  auto const& x0 = inns[0];
  auto const& x1 = inns[1];
  for(uint64_t i = 0; i != n; ++i) {
    out[i] = (x0[i]+((x1[i]*(*((float*)(d+0))))*(*((float*)(d+4)))));
  }
}

void ff(uint64_t n, float* out, float* inn) {
  for(uint64_t i = 0; i != n; ++i) {
    out[i] += inn[i];
  }
}

void main01() {
  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole@0"),
      scalarop_t::make_scale(0.1)
    }
  );

  int num_threads = 1;
  uint64_t dn = 10000*10000;
  //uint64_t dn = 1000;
  auto f_built = build_binary_elementwise_kernel(
    num_threads,
    dn,
    scalarop_t::make_add());
    //gradupdate);
    //scalarop_t::make_sub());

  buffer_t lhs = std::make_shared<buffer_holder_t>(dn);
  lhs->ones();

  buffer_t rhs = std::make_shared<buffer_holder_t>(dn);
  rhs->random(-0.9, -0.1);

  buffer_t out = std::make_shared<buffer_holder_t>(dn);

  int nrep = 5;

  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("built");
    f_built(out->data, {lhs->data, rhs->data});
  }

  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("loop ");
    //f(dn, out->data, lhs->data, rhs->data);
    f(dn, out->data, {lhs->data, rhs->data});
  }
}

int main02() {
  //scalarop_t s = scalarop_t::from_string("+[hole@0,*[*[hole@1,constant{0.1}],constant{-1}]]");
  //print_elementwise_function(s);
  //main01();

  uint64_t dn = 10000*10000;

  buffer_t lhs = std::make_shared<buffer_holder_t>(dn);
  lhs->ones();

  buffer_t rhs = std::make_shared<buffer_holder_t>(dn);
  rhs->random(-0.9, -0.1);

  buffer_t out = std::make_shared<buffer_holder_t>(dn);

  int nrep = 5;

  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("A ");
    f(dn, out->data, {lhs->data, rhs->data});
  }
  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("B ");
    ff(dn, out->data, lhs->data);
  }
  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("C ");
    f(dn, out->data, {lhs->data,rhs->data});
  }

  return 0;
}




int main03() {
  scalarop_t gradupdate = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::from_string("hole@0"),
      scalarop_t::make_scale(0.1)
    }
  );

  auto [str, bytes] = gradupdate.to_cpp_bytes();

  uint64_t dn = 10000*10000;

  buffer_t lhs = std::make_shared<buffer_holder_t>(dn);
  lhs->ones();

  buffer_t rhs = std::make_shared<buffer_holder_t>(dn);
  rhs->random(-0.9, -0.1);

  buffer_t out = std::make_shared<buffer_holder_t>(dn);

  int nrep = 5;

  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("A ");
    f(dn, out->data, {lhs->data, rhs->data});
  }
  for(int i = 0; i != nrep; ++i) {
    raii_print_time_elapsed_t gremlin("B ");
    g(bytes.data(), dn, out->data, {lhs->data, rhs->data});
  }

  return 0;
}

int main() {
  // Print the for loops
  vector<scalarop_t> ops = {
    scalarop_t::from_string("*[ite_<[hole@0,constant{0},constant{0},constant{1}],hole@1]"),
    scalarop_t::from_string("+[hole@0,*[*[hole@1,constant{0.3}],constant{-1}]]"),
    scalarop_t::from_string("+[hole@0,*[hole@1,constant{-1}]]"),
    scalarop_t::from_string("+[hole@0,hole@1]"),
    scalarop_t::from_string("+[hole@0,*[*[hole@1,constant{0.1}],constant{-1}]]"),
    scalarop_t::from_string("power{2}[+[hole@0,*[hole@1,constant{-1}]]]"),
    scalarop_t::from_string("*[constant{2},+[hole@0,*[hole@1,constant{-1}]]]"),
    scalarop_t::from_string("*[hole@0,hole@1]"),
    scalarop_t::from_string("+[hole@0,*[*[hole@1,constant{0.001}],constant{-1}]]"),
    scalarop_t::from_string("ite_<[hole@0,constant{0},constant{0},hole@0]")
  };

  set<string> so_far;
  vector<string> us;
  vector<string> bs;
  int n_unary  = 0;
  int n_binary = 1;
  for(auto const& op: ops) {
    auto [opstr, _] = op.to_cpp_bytes();

    if(so_far.count(opstr) > 0) {
      continue;
    }
    so_far.insert(opstr);

    string name;
    if(op.is_unary()) {
      name = "u" + std::to_string(us.size());
      std::cout << "_unary_ew_loop(" + name + "," + opstr + ");" << std::endl;
      us.push_back("{\""+opstr+"\","+name+"}");
    } else if(op.is_binary()) {
      name = "b" + std::to_string(bs.size());
      std::cout << "_binary_ew_loop(" + name + "," + opstr + ");" << std::endl;
      bs.push_back("{\""+opstr+"\","+name+"}");
    }
  }

  std::cout << "kernels = {" << std::endl;
  std::cout << "  " << us[0];
  for(int i = 1; i != us.size(); ++i) {
    std::cout << ",\n  " << us[i];
  }
  std::cout << "\n}" << std::endl;

  std::cout << "kernels = {" << std::endl;
  std::cout << "  " << bs[0];
  for(int i = 1; i != bs.size(); ++i) {
    std::cout << ",\n  " << bs[i];
  }
  std::cout << "\n}" << std::endl;
}
