#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/executetg.h"

#include <fstream>

void usage() {
  std::cout << "Usage: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

int main(int argc, char** argv) {
  // TODO: this should have an autoplacer setup...

  bool with_taskgraph = true;
  bool with_memgraph  = true;

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

  graph_constructor_t g;

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

  einsummable_t e({ni,nk}, { {0,1} }, 2, scalarop_t::make_relu());
  for(int i = 0; i != 20; ++i) {
    out = g.insert_einsummable(
      partition_t({
        partdim_t::split(ni, oi),
        partdim_t::split(nk, ok)
      }),
      e, { out });
  }

  taskgraph_t taskgraph;
  auto pls = g.get_placements();

  {
    auto [_0, _1, _taskgraph] = taskgraph_t::make(g.graph, pls);
    taskgraph = _taskgraph;
  }

  std::ofstream f("tg.gv");
  taskgraph.print_graphviz(f);
  DOUT("printed tg.gv");

  set_seed(0);

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  map<int, buffer_t> tensors;
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()) {
      auto const& [rank, size] = node.op.get_input();
      buffer_t buffer = make_buffer(size);
      dbuffer_t(default_dtype(), buffer).random("-0.0003", "0.003");
      tensors.insert({id, buffer});
    }
  }

  for(int i = 0; i != 10; ++i) {
    //{
    //  auto ops_in_order = random_taskgraph_order(taskgraph);

    //  // Note: this only works as expected if donation is not supported
    //  //       in execute_taskgraph_in_order
    //  map<int, buffer_t> tensors_shallow_copy = tensors;

    //  gremlin_t gremlin("mm single thread taskgraph execution (random)");
    //  execute_taskgraph_in_order(
    //    taskgraph,
    //    ops_in_order,
    //    kernel_manager,
    //    tensors_shallow_copy);
    //}
    {
      auto ops_in_order = temporal_taskgraph_order(taskgraph);

      // Note: this only works as expected if donation is not supported
      //       in execute_taskgraph_in_order
      map<int, buffer_t> tensors_shallow_copy = tensors;

      gremlin_t gremlin("mm single thread taskgraph execution (somewhat ordered)");
      execute_taskgraph_in_order(
        taskgraph,
        ops_in_order,
        kernel_manager,
        tensors_shallow_copy);
    }
  }
}
