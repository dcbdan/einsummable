#include "../src/base/setup.h"
#include "../src/einsummable/graph.h"
#include "../src/matrixgraph/matrixgraph.h"

#include <fstream>

void exp01() {
  graph_writer_t g;
  auto x = g.input({10,20,30});

  scalar_t factor(dtype_t::f32, "9.9");
  scalarop_t op = scalarop_t::make_scale(factor);

  auto y = g.ew(op, x);

  DLINEOUT("x,y is " << x.get_id() << ", " << y.get_id());

  auto z = g.backprop(y, {x});

  string filename = "exp01.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp01_matrixgraph() {
  matrixgraph_t g;
  int x = g.insert_input(10,20);

  scalar_t factor(dtype_t::f32, "9.9");
  scalarop_t op = scalarop_t::make_scale(factor);
  int y = g.insert_ew(op, x);
  int z = g.backprop(y, {x})[0];

  auto [graph, _] = g.compile();

  string filename = "exp01_matrixgraph.gv";
  std::ofstream f(filename);
  graph.print_graphviz(f);
  DOUT("printed " << filename);

  // TODO: this is incorrect!!!!!
}
void exp02() {
  graph_writer_t g;
  auto x = g.input({10,20,30});

  scalarop_t op = scalarop_t::make_square();

  auto y = g.ew(op, x);

  auto z = g.backprop(y, {x});

  string filename = "exp02.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp03() {
  graph_writer_t g;
  auto x = g.input({10,20,30});

  scalarop_t op = scalarop_t::make_square();
  auto y = g.ew(op, x.transpose(0,2));
  auto z = g.backprop(y, {x})[0];

  z = z.save();

  string filename = "exp03.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp04() {
  graph_writer_t g;
  auto x = g.input({10,20});
  auto y = g.input({20,30});
  auto z = g.matmul(x,y);
  auto w = g.backprop(z, {x})[0];
  w.save_inplace();

  string filename = "exp04.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp05() {
  graph_writer_t g;
  auto x = g.input({10,20});
  auto y = g.input({20,30});
  auto z = g.matmul(x,y);
  auto ws = g.backprop(z, {x,y});
  ws[0].save_inplace();
  ws[1].save_inplace();

  string filename = "exp05.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp04_matrixgraph() {
  matrixgraph_t g;
  auto x = g.insert_input(10,20);
  auto y = g.insert_input(20,30);
  auto z = g.insert_matmul_ss(x,y);
  auto w = g.backprop(z, {x})[0];

  auto [graph, _] = g.compile();

  string filename = "exp04_matrixgraph.gv";
  std::ofstream f(filename);
  graph.print_graphviz(f);
  DOUT("printed " << filename);
}

// TODO: backprop_state_t::start needs to start off with a constant value..

int main() {
  //exp01(); // TODO: fails: constants not implemented

  exp02();

  exp03();

  exp04();

  exp05();
}
