#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/fill.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/taskgraph.h"

#include "../src/autoplace/autoplace.h"

#include <fstream>

void exp01() {
  scalar_t v1 = scalar_t::negative_inf(dtype_t::f32);
  string v1_str = write_with_ss(v1);
  DOUT(v1_str);

  scalar_t v2;
  DOUT(v2);
  std::stringstream ss(v1_str);
  ss >> v2;
  string v2_str = write_with_ss(v2);
  DOUT(v2_str);


  scalarop_t c = scalarop_t::make_constant(v1);
  DOUT(c);
}

void exp02() {
  fill_t full_fill = fill_t::make_square_lowertri(default_dtype(), 1000);

  fill_t tl = full_fill.select({ {0, 500}, {0, 500} });
  fill_t tr = full_fill.select({ {0, 500}, {500, 1000} });
  fill_t bl = full_fill.select({ {500, 1000}, {0, 500} });
  fill_t br = full_fill.select({ {500, 1000}, {500, 1000} });

  DOUT(full_fill);
  DOUT("");
  DOUT(tl);
  DOUT(tr);
  DOUT(bl);
  DOUT(br);
}

int main() {
  graph_writer_t writer;
//  auto x = writer.input({1000, 1000});
//  //auto y = writer.constant(scalar_t::zero(default_dtype()), {1000, 1000});
//  auto y = writer.fill(fill_t(fill_t::lowertri_t {
//    .lower = scalar_t::one(default_dtype()),
//    .upper = scalar_t::zero(default_dtype()),
//    .nrow = 1000,
//    .ncol = 1000,
//    .start = 0
//  }));
//  auto z = writer.matmul(x,y).save();

  auto a = writer.input({1000, 2000});
  auto b = writer.input({2000, 1000});
  auto x = writer.matmul(a,b);
  //auto y = writer.constant(scalar_t::negative_inf(default_dtype()), {1000,1000});
  auto y = writer.fill(fill_t(fill_t::lowertri_t {
    .lower = scalar_t::zero(default_dtype()),
    .upper = scalar_t::negative_inf(default_dtype()),
    .nrow = 1000,
    .ncol = 1000,
    .start = 0
  }));
  auto z = writer.add(x,y);
  z = writer.softmax(z);

  z.save_inplace();

  graph_t const& graph = writer.get_graph();

  std::ofstream fg("g.gv");
  graph.print_graphviz(fg);
  DOUT("printed g.gv");

  int world_size = 1;
  int num_threads = 8;

  autoplace_config_t config =
    autoplace_config_t::make_default01(world_size, num_threads);

  vector<placement_t> placements;
  for(auto const& node: graph.nodes) {
    auto const& shape = node.op.shape();
    vector<partdim_t> pds;
    for(auto const& d: shape) {
      pds.push_back(partdim_t::split(d, 2));
    }
    placements.emplace_back(partition_t(pds));
  }

  //vector<placement_t> placements = autoplace01(graph, config);
  //for(auto const& pl: placements) {
  //  DOUT(pl.partition);
  //}

  DOUT("building taskgraph...");

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);

  std::ofstream ftg("tg.gv");
  taskgraph.print_graphviz(ftg);
  DOUT("printed tg.gv");
}

