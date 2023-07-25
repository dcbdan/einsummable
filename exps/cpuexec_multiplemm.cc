#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/executemg.h"
#include "../src/execution/cpu/mpi_class.h"

buffer_t make_random_buffer(uint64_t nelem) {
  buffer_t buffer = make_buffer(nelem * dtype_size(default_dtype()));
  dbuffer_t(default_dtype(), buffer).random("-0.0003", "0.003");
  return buffer;
}

struct state_t {
  state_t(int nt, uint64_t ni, uint64_t nj, uint64_t nk)
    : e(einsummable_t::from_matmul(ni,nj,nk))//, outs(nt)
  {
    kernel_manager.build(e);

    for(int i = 0; i != nt; ++i) {
      lhss.push_back(make_random_buffer(ni*nj));
      rhss.push_back(make_random_buffer(nj*nk));
      outs.push_back(make_random_buffer(ni*nk));
    }
  }

  void runner(int i) {
    //outs[i] = make_buffer(e.out_size());
    kernel_manager(e, outs[i]->raw(), { lhss[i]->raw(), rhss[i]->raw() });
  }

  void operator()() {
    vector<std::thread> runners;

    for(int i = 0; i != lhss.size(); ++i) {
      runners.emplace_back([i, this]{
        runner(i);
      });
    }

    for(auto& runner: runners) {
      runner.join();
    }
  }

  vector<buffer_t> lhss;
  vector<buffer_t> rhss;
  vector<buffer_t> outs;

  kernel_manager_t kernel_manager;
  einsummable_t e;
};

int main(int argc, char** argv) {
  if(argc != 5) {
    DOUT("usage: nt ni nj nk");
    return 1;
  }

  int nt;
  uint64_t ni, nj, nk;

  nt = parse_with_ss<int>(argv[1]);

  ni = parse_with_ss<uint64_t>(argv[2]);
  nj = parse_with_ss<uint64_t>(argv[3]);
  nk = parse_with_ss<uint64_t>(argv[4]);

  state_t state(nt, ni,nj,nk);

  gremlin_t gremlin(write_with_ss(nt) + "matmuls");
  state();
}
