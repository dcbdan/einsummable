#include "../src/execution/cpu/elementwise.h"

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

int main() {
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
    gradupdate);
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
