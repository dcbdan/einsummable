#include "../src/base/setup.h"
#include "../src/einsummable/graph.h"

#include "../src/einsummable/taskgraph.h"

#include <fstream>

void exp01() {
  graph_writer_t g;
  auto x = g.input({10,20,30});

  scalar_t factor(dtype_t::f32, "9.9");
  scalarop_t op = scalarop_t::make_scale(factor);

  auto y = g.ew(op, x);

  auto z = g.backprop(y, {x});

  string filename = "exp01.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
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

void exp06(bool with_mm = true) {
  graph_writer_t g;
  auto x = g.input({10,20});
  auto y = g.input({20,20});
  auto z = g.input({30,20});
  auto w = g.concat(0, {x,y,z});
  if(with_mm) {
    auto a = g.input({20,10});
    auto b = g.matmul(w, a);
    auto grads = g.backprop(b, {x,y,z});
  } else {
    auto grads = g.backprop(w, {x,y,z});
  }

  /////
  string filename = "exp06.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp07() {
  graph_constructor_t g;
  int x = g.insert_input(
    partition_t( { partdim_t::from_sizes({11,9}) }),
    dtype_t::f32);
  int y = g.insert_to_complex(
    partition_t( { partdim_t::from_sizes({5,5}) }),
    x);
  g.graph.nodes[y].op.set_save(true);

  string filename = "exp07_graph.gv";
  std::ofstream f(filename);
  g.graph.print_graphviz(f);
  DOUT("printed " << filename);

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, g.get_placements());
  filename = "exp07.gv";
  f = std::ofstream(filename);
  taskgraph.print_graphviz(f);
  DOUT("printed " << filename);
}

void exp08() {
  select_t select = make_concat(0, default_dtype(),
    vector<vector<uint64_t>>{ {20,30}, {10,30}, {40,30} });
  {
    hrect_t out_hrect;
    out_hrect.emplace_back(0, 70);
    out_hrect.emplace_back(0, 30);
    for(auto const& [inn_hrect, which_inn]: select.collect(out_hrect)) {
      hrect_t out_portion = select.wrt_output_hrect(inn_hrect, which_inn);
      DOUT(inn_hrect << " from " << which_inn << " .. (write region " << out_portion << ")");
    }
  }
  DOUT("");
  DOUT("");
  {
    hrect_t out_hrect;
    out_hrect.emplace_back(20, 70);
    out_hrect.emplace_back(10, 30);
    for(auto const& [inn_hrect, which_inn]: select.collect(out_hrect)) {
      hrect_t out_portion = select.wrt_output_hrect(inn_hrect, which_inn);
      DOUT(inn_hrect << " from " << which_inn << " .. (write region " << out_portion << ")");
    }
  }
}

void exp09() {
  graph_writer_t g;
  auto x = g.input({100,400});
  auto y = g.input({200,400});
  auto x1 = g.subset({ {0,100}, {0,200} }, x);
  auto z = g.matmul(x1, y);
  auto grad_x = g.backprop(z, {x})[0];

  /////
  string filename = "exp09.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp10() {
  graph_writer_t g;
  auto x = g.input({100,200});
  auto y = g.subset({ {30,40}, {80,100} }, x);
  auto grads = g.backprop(y, {x});
  /////
  string filename = "exp10.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp11() {
  graph_writer_t g;
  auto x = g.input({100,200}, dtype_t::f32);

  auto y = g.input({100,100}, dtype_t::c64);
  auto z = g.to_real(y);
  auto w = g.add(x, z);
  auto grads = g.backprop(w, {x,y});
  /////
  string filename = "exp11.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp12() {
  graph_writer_t g;
  auto x = g.input({100,200}, dtype_t::f32);

  auto y = g.input({100,100}, dtype_t::c64);
  auto z = g.to_real(y);
  auto w = g.matmul(x, z.transpose(0,1));
  auto grads = g.backprop(w, {x,y});
  /////
  string filename = "exp12.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp13() {
  graph_writer_t g;
  auto x = g.input(vector<uint64_t>{100});
  auto y = g.broadcast(101, x);
  auto z = g.broadcast(102, y);
  auto a = g.input({35,36});
  a = a.transpose(0,1);
  a = a.scale("1.99");
  auto grads = g.backprop(z, {x});
  /////
  string filename = "exp13.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp14() {
  graph_writer_t g;
  auto x = g.input({20,30,40});
  auto a = g.reduction("ijk->ji", castable_t::add, x);
  auto b = g.reduction("ijk->k", castable_t::min, x);
  auto c = g.reduction("ijk->ik", castable_t::mul, x);
  auto d = g.reduction("ijk->j", castable_t::max, x);

  g.backprop(a, {x});
  g.backprop(b, {x});
  g.backprop(c, {x});
  g.backprop(d, {x});

  /////
  string filename = "exp14.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

void exp15() {
  graph_writer_t g;
  auto x = g.input({20,30,40});
  auto y = g.input({40,50});
  auto a = g.reduction("ijk->ik", castable_t::min, x);
  auto b = g.matmul(a,y).save();
  /////
  {
    string filename = "exp15_before.gv";
    std::ofstream f(filename);
    g.get_graph().print_graphviz(f);
    DOUT("printed " << filename);
  }

  g.backprop(b, {x});
  /////
  string filename = "exp15.gv";
  std::ofstream f(filename);
  g.get_graph().print_graphviz(f);
  DOUT("printed " << filename);
}

// TODO: check einsummable_t remove_broadcast
// TODO: fixed_grad_id in build grad term einsummable

int main() {
  //exp01();
  //exp02();
  //exp03();
  //exp04();
  //exp05();
  //exp06(false);
  //exp06(true);
  //exp07();
  //exp09();
  //exp10();
  //exp11();
  //exp12();
  //exp13();
  //exp14();
  exp15();
}
