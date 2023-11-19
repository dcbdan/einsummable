#include "../src/base/args.h"
#include "../src/einsummable/dbuffer.h"
#include "../src/engine/cpu/kernel_executor.h"

void main_(int argc, char** argv) {
  args_t args(argc, argv);
  args.set_default("nb", uint64_t(2));
  args.set_default("ni", uint64_t(10));
  args.set_default("nj", uint64_t(10));
  args.set_default("nk", uint64_t(10));
  args.set_default("bout", true);
  args.set_default("blhs", true);
  args.set_default("brhs", true);

  uint64_t nb = args.get<uint64_t>("nb"); 
  uint64_t ni = args.get<uint64_t>("ni"); 
  uint64_t nj = args.get<uint64_t>("nj"); 
  uint64_t nk = args.get<uint64_t>("nk"); 

  bool bout = args.get<bool>("bout");
  bool blhs = args.get<bool>("blhs");
  bool brhs = args.get<bool>("brhs");

  dbuffer_t L  = make_dbuffer(dtype_t::f32, (blhs ? nb : 1)*ni*nj);
  dbuffer_t R  = make_dbuffer(dtype_t::f32, (brhs ? nb : 1)*nj*nk);
  dbuffer_t O  = make_dbuffer(dtype_t::f32, (bout ? nb : 1)*ni*nk);
  dbuffer_t O1 = make_dbuffer(dtype_t::f32, (bout ? nb : 1)*ni*nk);

  L.ones();
  R.ones();

  //for(int i = 0; i != 1; ++i) {
  //  bmm(nb, bout,blhs,brhs, ni,nj,nk, false, false, O1.raw(), L.raw(), R.raw()); 
  //  DLINEOUT("SUCCESS ON BMM");
  //}

  batch_matrix_multiply(
    dtype_t::f32, 
    nb, bout,blhs,brhs, ni,nj,nk, false, false, O.raw(), L.raw(), R.raw()); 

  DOUT(nb << " | " << ni << "," << nj << "," << nk);
  DOUT(O.min() << "          " << O.max());
}

int main(int argc, char** argv) {
  main_(argc, argv);
}
