#include "../src/base/setup.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/memgraph.h"

#include <fstream>

void exp01() {
  graph_writer_t g;
  auto x = g.input({10,20,30});

  scalar_t factor(dtype_t::f32, "2.0");
  scalarop_t op = scalarop_t::make_scale(factor);

  einsummable_t e({10,20,30}, { {0,1,2} }, 3, op);
  auto y = g.ew(op, x);

  auto z = g.backprop(y, {x});

  string filename = "exp01.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

int main() {
  exp01();
}
