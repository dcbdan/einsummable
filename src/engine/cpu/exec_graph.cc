#include "../exec_graph.h"

exec_graph_t
exec_graph_t::make_cpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  cpu_kernel_executor_t& cpu_executor)
{
  return exec_graph_t {
    .cpu_executor = cpu_executor
  };
}

