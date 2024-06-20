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

  int num_gpus = 2;
  int num_computes_per_loc = 1;

  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);


  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);


  vector<uint64_t> mem_sizes(2, 100000);

  auto [_2, _3, memgraph] = memgraph_t::make(taskgraph, {},  mem_sizes);

  std::ofstream f("mg.gv");
  memgraph.print_graphviz(f);
  DOUT("printed mg.gv");
}