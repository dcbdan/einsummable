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
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include <fstream>
#include "../src/einsummable/fusion.h"

// int main() {
//   graph_t graph;

//   // Assuming A, B, and W are all tensors of shape 100 by 100
//   int A = graph.insert_input({100, 100});
//   int B = graph.insert_input({100, 100});
//   int W = graph.insert_input({100, 100});

//   // Insert element-wise addition operations
//   scalarop_t add_op = scalarop_t::make_add();
//   einsummable_t add_einsum({100, 100}, vector<vector<int>>{{0},{1}}, 2, add_op);

//   // U = A + W
//   int U = graph.insert_einsummable(add_einsum, {A, W});
//   U = graph.insert_formation(U); // Insert formation node for U if required

//   // V = B + W
//   int V = graph.insert_einsummable(add_einsum, {B, W});
//   V = graph.insert_formation(V); // Insert formation node for V if required

//   // Save results
//   graph.nodes[U].op.set_save(true);
//   graph.nodes[V].op.set_save(true);

//   // Export the graph to a Graphviz file
//   {
//     std::ofstream f("graph.gv");
//     vector<partition_t> parts = graph.make_singleton_partition();
//     graph.print_graphviz(f, parts);
//   }
  
//   return 0;
// }


// int main() {
//   graph_t graph;

//   // Assuming A, B, and W are all tensors of shape 100 by 100
//   int A = graph.insert_input({100, 100});
//   int B = graph.insert_input({100, 100});
//   int W = graph.insert_input({100, 100});

//   // Corrected: The einsummable_t operation is expecting two input shapes of {100, 100}
//   // which matches the shapes of tensors A and W (and B and W for the second operation).
//   scalarop_t add_op = scalarop_t::make_add();
//   einsummable_t add_einsum({100, 100}, vector<vector<int>>{{0,1}, {0,1}}, 2, add_op);

//   // U = A + W
//   int U = graph.insert_einsummable(add_einsum, {A, W});
//   U = graph.insert_formation(U); // Insert formation node for U if required

//   // V = B + W
//   int V = graph.insert_einsummable(add_einsum, {B, W});
//   V = graph.insert_formation(V); // Insert formation node for V if required

//   // Save results
//   graph.nodes[U].op.set_save(true);
//   graph.nodes[V].op.set_save(true);

//   // Export the graph to a Graphviz file
//   std::ofstream f("g.gv");
//   vector<partition_t> parts = graph.make_singleton_partition();
//   graph.print_graphviz(f, parts);
//   f.close();

//   return 0;
// }

// int main()
// {
//   graph_t graph;

//   int x = graph.insert_input({10,20});
//   int y = graph.insert_input({20,30});

//   einsummable_t matmul = einsummable_t::from_matmul(10,20,30);
//   int z = graph.insert_einsummable(matmul, {x, y});
//   z = graph.insert_formation(z);		
  
//   scalarop_t op = scalarop_t::make_relu();
//   einsummable_t ew(
//     {10,30},
//     vector<vector<int>>{ { 0, 1 } },
//     2,
//     op);
//   int w = graph.insert(ew, vector<int>{z});
//   graph.nodes[w].op.set_save(true);

//   // Export the graph to a Graphviz file
//   std::ofstream f("g3.gv");
//   vector<partition_t> parts = graph.make_singleton_partition();
//   graph.print_graphviz(f, parts);
//   f.close();

//   return 0;

// }

vector<placement_t> create_loc0_placements(vector<partition_t> const& parts) 
{
    vector<placement_t> ret;
    ret.reserve(parts.size());
    for(auto const& part: parts) 
    {
        ret.emplace_back(part);
    }
    return ret;
}

int main() {
    graph_t graph;

    // Assuming X is a tensor of shape 100 by 100
    int X = graph.insert_input({100, 100});

    // Create an einsummable for the exponential operation: exp(X)
    scalarop_t exp_op = scalarop_t::make_exp();
    einsummable_t exp_einsum({100, 100}, vector<vector<int>>{{0,1}}, 2, exp_op);
    int Y = graph.insert_einsummable(exp_einsum, std::vector<int>{X});
    std::cout << "exp\n";

    // Create ReLU(Y)
    scalarop_t relu_op = scalarop_t::make_relu();
    einsummable_t relu_einsum({100,100}, vector<vector<int>>{{0,1}}, 2, relu_op);
    int Z = graph.insert_einsummable(relu_einsum, std::vector<int>{Y});
    std::cout << "relu\n";

    graph.nodes[Z].op.set_save(true);

    // Visualizing the ORIGINAL Computation
    std::ofstream g1("g_exp_relu.gv");
    std::vector<partition_t> parts1 = graph.make_singleton_partition();
    graph.print_graphviz(g1, parts1);
    g1.close();

    std::ofstream tg1("tg_exp_relu.gv");
    std::vector<placement_t> placs1 = create_loc0_placements(parts1);
    auto [inns_1, outs_1, taskgraph] = taskgraph_t::make(graph, placs1);
    taskgraph.print_graphviz(tg1);
    tg1.close();
    
    // Do the Fusion
    fusion_t fusion;
    graph_t newGraph = fusion.apply(graph);

    // Visualizing the "OPTIMIZED" Computation 
    std::ofstream g2("g_exp_relu_fused.gv");
    vector<partition_t> parts2 = newGraph.make_singleton_partition();
    newGraph.print_graphviz(g2, parts2);
    g2.close();
    
    std::ofstream tg2("tg_exp_relu_fused.gv");
    std::vector<placement_t> placs2 = create_loc0_placements(parts2);
    auto [inns_2, outs_2, newTaskgraph] = taskgraph_t::make(newGraph, placs2);
    newTaskgraph.print_graphviz(tg2);
    tg2.close();

    return 0;
}

