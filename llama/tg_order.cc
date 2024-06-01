#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/mgmake.h"

#include "../src/autoplace/autoplace.h"

#include "modules.h"

#include <fstream>
template <typename T>
void print_graphviz(T const& g, string filename) {
  std::ofstream f(filename);
  g.print_graphviz(f);
  DOUT("printed " << filename);
}

void exp01() {
  graph_writer_t writer;

  auto X = writer.input({100, 100});
  auto Y = writer.ew(scalarop_t::make_exp(), X).save();

  graph_t const& graph = writer.get_graph();
  if(graph.nodes.size() != 2 || X.get_id() != 0) {
    throw std::runtime_error("...");
  }

  partdim_t pd1 = partdim_t::split(100, 1);
  partdim_t pd4 = partdim_t::split(100, 4);

  vector<partition_t> parts {
    partition_t({ pd4, pd4 }),
    partition_t({ pd4, pd1 })
  };
  vector<placement_t> pls;
  for(auto const& part: parts) {
    pls.emplace_back(part);
  }

  print_graphviz(graph, "g.gv");

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

  print_graphviz(taskgraph, "tg.gv");

  DOUT("`order_taskgraph` results");
  for(auto const& x: order_taskgraph(taskgraph)) {
    DOUT(x);
  }
  DOUT("");

  DOUT("`order_taskgraph_priority_min_delta` results");
  for(auto const& x: order_taskgraph_priority_min_delta(taskgraph)) {
    DOUT(x);
  }
}

void exp02() {
  graph_writer_t writer;

  auto X = writer.input({100, 100});
  auto Y = writer.ew(scalarop_t::make_exp(), X).save();

  auto A = writer.input({100, 100});
  auto B = writer.ew(scalarop_t::make_exp(), A).save();

  graph_t const& graph = writer.get_graph();
  if(graph.nodes.size() != 4 || X.get_id() != 0) {
    throw std::runtime_error("...");
  }

  int N = 4;
  partdim_t pd1 = partdim_t::split(100, 1);
  partdim_t pdN = partdim_t::split(100, N);

  vector<partition_t> parts {
    partition_t({ pdN, pdN }),
    partition_t({ pdN, pd1 }),
    partition_t({ pdN, pd1 }),
    partition_t({ pdN, pdN })
  };
  vector<placement_t> pls;
  for(auto const& part: parts) {
    pls.emplace_back(part);
  }

  print_graphviz(graph, "g.gv");

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

  print_graphviz(taskgraph, "tg.gv");

  auto print_order_stuff = [&](vector<_which_op_t> const& xs) {
    for(auto const& x: xs) {
      DOUT(x);
    }
    DOUT(order_taskgraph_memory_usage(taskgraph, xs));
  };

  DOUT("`order_taskgraph` results");
  print_order_stuff(order_taskgraph(taskgraph));
  DOUT("");

  DOUT("`order_taskgraph_priority_min_delta` results");
  print_order_stuff(order_taskgraph_priority_min_delta(taskgraph));
}

void exp03() {
  graph_writer_t writer;
  auto args = model_args_t::llama(1, 1);
  args.n_layers = 3;
  transformer_t model(&writer, model_args_t::llama(1, 1), 0);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(args.batch_size),
    full_dim_t::singleton(args.max_seq_len),
    args.full_dim()
  }));

  tensor_t predictions = model.forward(embeddings);
  predictions.save_inplace();

  for(auto [key,value]: model.get_new_kvs()) {
    key.save_inplace();
    value.save_inplace();
  }
  for(auto [name, weight]: model.weight_map()) {
    weight.save_inplace();
  }

  ///////////////////
  auto const& graph = writer.get_graph();
  auto pls = autoplace01(graph, autoplace_config_t::make_default01(1, 1));

  auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

  auto print_order_stuff = [&](vector<_which_op_t> const& xs) {
    DOUT(order_taskgraph_memory_usage(taskgraph, xs));
  };

  DOUT("`order_taskgraph` results");
  print_order_stuff(order_taskgraph(taskgraph));
  DOUT("");

  DOUT("`order_taskgraph_priority_min_delta` results");
  print_order_stuff(order_taskgraph_priority_min_delta(taskgraph));
}

int main() {
  exp03();
}
