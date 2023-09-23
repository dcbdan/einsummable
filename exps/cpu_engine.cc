#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"

#include "../src/einsummable/dbuffer.h"

#include <fstream>

void usage() {
  std::cout << "Usage: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

void execute_memgraph_cpu(memgraph_t const& memgraph, buffer_t buffer);

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

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  auto [inn_to_mem, out_to_mem, memgraph] = memgraph_t::make_without_evict(taskgraph);

  {
    std::ofstream f("mg.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }

  buffer_t buffer = make_buffer(memgraph.mem_sizes()[0]);

  for(auto const& [inn,mem]: inn_to_mem) {
    buffer_t tensor_ = make_buffer_reference(buffer->data + mem.offset, mem.size);
    dbuffer_t tensor(dtype, tensor_);
    tensor.random("1.0", "1.0");
    DOUT(tensor);
  }

  execute_memgraph_cpu(memgraph, buffer);

  for(auto const& [out,mem]: out_to_mem) {
    buffer_t tensor_ = make_buffer_reference(buffer->data + mem.offset, mem.size);
    dbuffer_t tensor(dtype, tensor_);
    DOUT(tensor);
  }
}

void execute_memgraph_cpu(memgraph_t const& memgraph, buffer_t buffer)
{
  cpu_kernel_executor_t executor;

  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(memgraph, 0, executor);

  cpu_workspace_manager_t cpu_workspace_manager;
  group_manager_t group_manager;
  global_buffer_t global_buffer(buffer->raw());

  resource_manager_t resource_manager {
    .cpu_workspace_manager = &cpu_workspace_manager,
    .group_manager = &group_manager,
    .global_buffer = &global_buffer
  };

  exec_state_t state(graph, resource_manager);

  state.event_loop();
}

