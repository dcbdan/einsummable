#include "../llama/misc.h"
#include "../llama/modules.h"
#include "../llama/builder.h"
#include "../llama/reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

int main() {
  uint64_t ni = 100;
  uint64_t nj = 100;
  uint64_t nk = 100;

  graph_writer_t writer;

  using tensor_t = graph_writer_t::tensor_t;

  tensor_t lhs1 = writer.input({ni,nj});
  tensor_t rhs1 = writer.input({nj,nk});

  tensor_t lhs2 = writer.input({ni,nj});
  tensor_t rhs2 = writer.input({nj,nk});

  tensor_t out1 = writer.matmul(lhs1, rhs1);
  tensor_t out2 = writer.matmul(lhs2, rhs2);

  tensor_t out = writer.matmul(out1, out2).save();

  auto const& graph = writer.get_graph();

  vector<placement_t> pls;
  for(auto const& part: graph.make_singleton_partition()) {
    pls.emplace_back(part);
  }

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

  uint64_t GB = 1000000000;
  uint64_t nGB = 10;
  vector<uint64_t> mem_sizes(1, nGB*GB);

  auto [_2, _3, memgraph] = memgraph_t::make(taskgraph, {},  mem_sizes);

  std::ofstream f("mg.gv");
  memgraph.print_graphviz(f);
  DOUT("printed mg.gv");
}