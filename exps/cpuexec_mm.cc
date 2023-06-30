#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/execute.h"
#include "../src/execution/cpu/mpi_class.h"

#include <fstream>

#define ROUT(x) if(mpi.this_rank == 0) { std::cout << "(in cpu exec) " << x << std::endl; }

void usage() {
  std::cout << "Usage: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

int main(int argc, char** argv) {
  if(argc != 13) {
    usage();
    return 1;
  }

  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  try {
    ni = parse_with_ss<uint64_t>(argv[1]);
    nj = parse_with_ss<uint64_t>(argv[2]);
    nk = parse_with_ss<uint64_t>(argv[3]);

    li = parse_with_ss<int>(argv[4]);
    lj = parse_with_ss<int>(argv[5]);

    rj = parse_with_ss<int>(argv[6]);
    rk = parse_with_ss<int>(argv[7]);

    ji = parse_with_ss<int>(argv[8]);
    jj = parse_with_ss<int>(argv[9]);
    jk = parse_with_ss<int>(argv[10]);

    oi = parse_with_ss<int>(argv[11]);
    ok = parse_with_ss<int>(argv[12]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
  }
  DOUT("ni nj nk " << (vector<uint64_t>{ni,nj,nk}));
  DOUT("li lj    " << (vector< int    >{li,lj}   ));
  DOUT("rj rk    " << (vector< int    >{rj,rk}   ));
  DOUT("ji jj jk " << (vector< int    >{ji,jj,jk}));
  DOUT("oi ok    " << (vector< int    >{oi,ok}   ));

  mpi_t mpi(argc, argv);

  graph_constructor_t g;
  dtype_t dtype = default_dtype();

  int lhs = g.insert_input(partition_t({
    partdim_t::split(ni, li),
    partdim_t::split(nj, lj) }));
  int rhs = g.insert_input(partition_t({
    partdim_t::split(nj, rj),
    partdim_t::split(nk, rk) }));

  int join = g.insert_einsummable(
    partition_t({
      partdim_t::split(ni, ji),
      partdim_t::split(nk, jk),
      partdim_t::split(nj, jj)
    }),
    einsummable_t::from_matmul(ni, nj, nk),
    {lhs, rhs});

  int out = g.insert_formation(
    partition_t({
      partdim_t::split(ni, oi),
      partdim_t::split(nk, ok)
    }),
    join);

  auto pls = g.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, pls);

  if(mpi.this_rank == 0) {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  int num_threads_per_node = 8;

  settings_t execute_settings {
    .num_apply_runner = num_threads_per_node,
    .num_touch_runner = num_threads_per_node,
    .num_send_runner  = 1,
    .num_recv_runner  = 1,
    .num_apply_kernel_threads = 1
  };

  map<int, buffer_t> tensors;
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()) {
      auto const& [rank, size] = node.op.get_input();
      if(mpi.this_rank == rank) {
        buffer_t buffer = make_buffer(size);
        dbuffer_t(dtype, buffer).random("-0.0003", "0.003");
        tensors.insert({id, buffer});
      }
    }
  }

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  {
    mpi.barrier();
    raii_print_time_elapsed_t gremlin("cpuexec_mm time");
    execute(taskgraph, execute_settings, kernel_manager, &mpi, tensors);
    mpi.barrier();
  }
}
