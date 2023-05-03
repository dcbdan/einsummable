#include "../src/execution/cpu/kernels.h"

#include "../src/einsummable/reference.h"

int main() {
  uint64_t nb = 20;
  uint64_t ni = 10000;
  uint64_t nj = 10000;
  uint64_t nk = 10000;

  //buffer_t lhs = std::make_shared<buffer_holder_t>(nb*ni*nj);
  buffer_t lhs = std::make_shared<buffer_holder_t>(ni*nj);
  lhs->ones();

  buffer_t rhs = std::make_shared<buffer_holder_t>(nj*nk);
  rhs->ones();

  buffer_t out = std::make_shared<buffer_holder_t>(ni*nk);
  out->random();

  {
    raii_print_time_elapsed_t gremlin("mm");
    matrix_multiply(ni,nj,nk,false,false,out->data, lhs->data, rhs->data);
    //batch_matrix_multiply(
    //  nb,true,false,false,ni,nj,nk,false,false,
    //  out->data,lhs->data,rhs->data);
  }

  //DOUT(out);
}

