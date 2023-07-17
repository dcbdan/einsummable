#include "../src/execution/cpu/kernels.h"
#include "../src/einsummable/dbuffer.h"

void test1() {
  kernel_manager_t kernel_manager;

  einsummable_t e = einsummable_t::from_matmul_tt(4000,4000,2000);
  kernel_manager.build(e);

  // ji,kj->ik
  // ca,bc->ab

  dbuffer_t lhs = make_dbuffer(default_dtype(), 4000*4000);
  dbuffer_t rhs = make_dbuffer(default_dtype(), 2000*4000);
  dbuffer_t out = make_dbuffer(default_dtype(), 4000*2000);

  DOUT(e.inn_shapes());
  DOUT(e.out_shape());

  vector<void const*> inns;
  inns.push_back(lhs.raw());
  inns.push_back(rhs.raw());
  kernel_manager(e, out.raw(), inns);
}

int main() {
  set_default_dtype(dtype_t::f16);

  kernel_manager_t kernel_manager;

  //vector<uint64_t> szs = {1,4,32,256,1000,2000,4000};
  //for(auto const& i: szs) {
  //for(auto const& j: szs) {
  //for(auto const& k: szs) {
  //  kernel_manager.build(einsummable_t::from_matmul_ss(i,j,k));
  //  kernel_manager.build(einsummable_t::from_matmul_st(i,j,k));
  //  kernel_manager.build(einsummable_t::from_matmul_ts(i,j,k));
  //  kernel_manager.build(einsummable_t::from_matmul_tt(i,j,k));
  //}}}

  vector<uint64_t> szs = {256,1000,2000,4000};
  for(auto const& i: szs) {
  for(auto const& j: szs) {
    kernel_manager.build(einsummable_t({i,j}, { {0,1}, {0,1} }, 2, scalarop_t::make_add()));
  }}

  kernel_manager.make_dataset(std::cout);
}
