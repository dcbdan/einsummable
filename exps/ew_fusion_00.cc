// #include "misc.h"
// #include "modules.h"

// #include "../src/base/args.h"

// #include "../src/autoplace/autoplace.h"
// #include "../src/autoplace/apart.h"
// #include "../src/autoplace/alocate.h"

// #include <fstream>

// int main(int argc, char** argv) {
//   args_t pargs(argc, argv);

//   model_args_t margs = model_args_t::llama(1, 1);

//   pargs.set_default<uint64_t>("batch_size", 1);
//   margs.batch_size = pargs.get<uint64_t>("batch_size");

//   pargs.set_default<uint64_t>("sequence_length", 4096);
//   margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

//   graph_writer_t writer;

//   uint64_t hidden_dim = 4 * margs.dim;
//   hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
//   hidden_dim =
//     margs.multiple_of * ( (hidden_dim + margs.multiple_of - 1) / margs.multiple_of );

//   feedforward_t model(&writer, "ff", margs.full_dim(), hidden_dim);

//   tensor_t embeddings = writer.input(full_shape_t({
//     full_dim_t::singleton(margs.batch_size),
//     full_dim_t::singleton(margs.max_seq_len),
//     margs.full_dim()
//   }));

//   tensor_t predictions = model.forward(embeddings);
//   predictions.save_inplace();

//   graph_t const& graph = writer.get_graph();

//   pargs.set_default<int>("num_config", 64);
//   int num_config = pargs.get<int>("num_config");
//   vector<partition_t> parts = apart01(graph, num_config, 1);

//   {
//     std::ofstream f("g.gv");
//     graph.print_graphviz(f, parts);
//     DOUT("g.gv");
//   }
// }

#include "../src/einsummable/graph.h"
#include "../src/einsummable/einsummable.h"
#include <ofstream>

int main() {
  graph_t graph;

  // Assuming A, B, and W are all tensors of shape 100 by 100
  int A = graph.insert_input({100, 100});
  int B = graph.insert_input({100, 100});
  int W = graph.insert_input({100, 100});

  // Insert element-wise addition operations
  scalarop_t add_op = scalarop_t::make_add();
  einsummable_t add_einsum({100, 100}, vector<vector<int>>{{0, 1}}, 2, add_op);

  // U = A + W
  int U = graph.insert_einsummable(add_einsum, {A, W});
  U = graph.insert_formation(U); // Insert formation node for U if required

  // V = B + W
  int V = graph.insert_einsummable(add_einsum, {B, W});
  V = graph.insert_formation(V); // Insert formation node for V if required

  // Save results
  graph.nodes[U].op.set_save(true);
  graph.nodes[V].op.set_save(true);

  // Export the graph to a Graphviz file
  {
    std::ofstream f("graph.gv");
    vector<partition_t> parts = graph.make_singleton_partition();
    graph.print_graphviz(f, parts);
  }
  
  return 0;
}
