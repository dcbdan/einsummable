#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"

int main() {
  graph_writer_t g;
  auto x = g.input({10,10});
  auto y = g.input({10,10});
  auto z = g.matmul(x,y);
  auto a = g.add(y,z);
  a.save_inplace();

  apart01(g.get_graph(), 32);
}
