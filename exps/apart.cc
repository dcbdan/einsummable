#include "../src/autoplace/apart.h"
#include <fstream>

#include "../src/matrixgraph/ff.h"

graph_t make_g01() {
  using id_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  id_t x = writer.input({1000, 2000});
  id_t y = writer.input({2000, 3000});
  id_t z = writer.matmul(x, y);

  z = z.save();

  return writer.get_graph();
}

graph_t make_g02() {
  using id_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  uint64_t bsz = 300;
  uint64_t d0 = 4000;
  uint64_t d1 = 5000;
  uint64_t d2 = 6000;
  uint64_t d3 = 8000;

  vector<uint64_t> d01 = {d0, d1};
  vector<uint64_t> d12 = {d1, d2};
  vector<uint64_t> d22 = {d2, d2};
  vector<uint64_t> d32 = {d3, d2};
  vector<uint64_t> d33 = {d3, d3};

  dtype_t dtype = dtype_t::f32;

  id_t x = writer.input({bsz,d0,d1}, dtype).view({ {bsz}, d01 });
  id_t w0 = writer.input({d0,d1,d3,d2}, dtype).view({d01,d32});
  id_t w1 = writer.input({d3,d2}, dtype);

  id_t y = writer.matmul(x, w0).view_full({bsz, d3, d2});
  id_t z = writer.matmul(y, w1.transpose(0,1)); // bsz,d3,d3
  y = writer.concat(2, {y, z}); // bsz,d3,d33
  y = writer.reduction("bxy->bx", castable_t::add, y); // bsz,d1
  y = y.to_complex();
  y = y.to_real();

  y = y.save();

  return writer.get_graph();
}

graph_t make_g03()
{
  using id_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  id_t x = writer.input({3000,3000});
  uint64_t prev_size = 3000;
  for(int i = 0; i != 4; ++i) {
    uint64_t next_size = 500*runif(1, 10); // 1000;
    id_t y = writer.input({prev_size,next_size});
    prev_size = next_size;
    x = writer.matmul(x,y);
  }
  x = x.save();

  return writer.get_graph();
}

graph_t make_g04() {
  uint64_t dn = 10000;
  uint64_t dp = 250;
  uint64_t dd = 500;
  vector<uint64_t> dws = {450,550,1000,2500};
  //vector<uint64_t> dws = {450, 5500};
  dtype_t dtype = default_dtype();
  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,0.012,dtype);
  vector<int> outs  = ff_info.wsout;
  outs.push_back(ff_info.sqdiff);
  auto [graph, m_to_g] = ff_info.mgraph.compile(outs);
  return graph;
}

int main() {
  //graph_t graph = make_g01();
  //graph_t graph = make_g03();
  graph_t graph = make_g04();

  int n_compute = 128;
  auto parts = autopartition_for_bytes(graph, n_compute);

  std::ofstream f("g_with_parts.gv");
  graph.print_graphviz(f, parts);
  DOUT("printed g_with_parts.gv");
}
