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

#include "../src/base/setup.h"
#include "../src/base/args.h"
#include "../src/einsummable/graph.h"
#include "../src/engine/communicator.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/apart.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/taskgraph.h"

#include <iostream>
#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <fstream>
#include <sstream>

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

// int main() {
//     graph_t graph;

//     // Assuming X is a tensor of shape 100 by 100
//     int X = graph.insert_input({100, 100});

//     // Create an einsummable for the exponential operation: exp(X)
//     scalarop_t exp_op = scalarop_t::make_exp();
//     einsummable_t exp_einsum({100, 100}, vector<vector<int>>{{0,1}}, 2, exp_op);
//     int Y = graph.insert_einsummable(exp_einsum, std::vector<int>{X});
//     std::cout << "exp\n";

//     // Create ReLU(Y)
//     scalarop_t relu_op = scalarop_t::make_relu();
//     einsummable_t relu_einsum({100,100}, vector<vector<int>>{{0,1}}, 2, relu_op);
//     int Z = graph.insert_einsummable(relu_einsum, std::vector<int>{Y});
//     std::cout << "relu\n";

//     graph.nodes[Z].op.set_save(true);

//     // Visualizing the ORIGINAL Computation
//     std::ofstream g1("g_exp_relu.gv");
//     std::vector<partition_t> parts1 = graph.make_singleton_partition();
//     graph.print_graphviz(g1, parts1);
//     g1.close();

//     std::ofstream tg1("tg_exp_relu.gv");
//     std::vector<placement_t> placs1 = create_loc0_placements(parts1);
//     auto [inns_1, outs_1, taskgraph] = taskgraph_t::make(graph, placs1);
//     taskgraph.print_graphviz(tg1);
//     tg1.close();
    
//     // Do the Fusion
//     fusion_t fusion;
//     graph_t newGraph = fusion.apply(graph);

//     // Visualizing the "OPTIMIZED" Computation 
//     std::ofstream g2("g_exp_relu_fused.gv");
//     vector<partition_t> parts2 = newGraph.make_singleton_partition();
//     newGraph.print_graphviz(g2, parts2);
//     g2.close();
    
//     std::ofstream tg2("tg_exp_relu_fused.gv");
//     std::vector<placement_t> placs2 = create_loc0_placements(parts2);
//     auto [inns_2, outs_2, newTaskgraph] = taskgraph_t::make(newGraph, placs2);
//     newTaskgraph.print_graphviz(tg2);
//     tg2.close();

//     return 0;
// }



void usage() {
  std::cout << "Setup usage: addr_zero is_client world_size memsize(GB)\n";
  std::cout << "Single Machine Example: ./comp522 10.0.0.76 0 1 7\n";
}



int main(int argc, char** argv) {
    if(argc < 4) {
    usage();
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = ( parse_with_ss<int>(argv[2]) == 0 );
  int world_size = parse_with_ss<int>(argv[3]);

//  string addr_zero = "0.0.0.0";
//   bool is_rank_zero = true;
//   int world_size = 1;

//   int num_threads = std::max(1, int(std::thread::hardware_concurrency()));

//   int num_channels = 8;
//   int num_channels_per_move = 2;

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000'000'000;
  mem_size *= GB;
  vector<uint64_t> buffer_size = {mem_size};

  if(is_rank_zero) {
    DOUT("----------------------HARDWARE SETTINGS-----------------------------");
    DOUT("Cluster Size:                                " << world_size << " Machine");
    DOUT("Memory Allocated per Machine:                " << (mem_size/GB) << " GB");
    std::cout << std::endl;
  }

  gpu_mg_server_t server(communicator, buffer_size);

  if(is_rank_zero) {
    args_t args(argc-4, argv+4);
    
    args.set_default("pp", false);
    server.set_parallel_partialize(args.get<bool>("pp"));

    server.set_use_storage(false);


    uint64_t a = 100;
    uint64_t b = 100;

    graph_t graph;

    // Assuming X is a tensor of shape 100 by 100
    int X = graph.insert_input({a, b});

    // Create an einsummable for the exponential operation: exp(X)
    scalarop_t exp_op = scalarop_t::make_exp();
    einsummable_t exp_einsum({a, b}, vector<vector<int>>{{0,1}}, 2, exp_op);
    int Y = graph.insert_einsummable(exp_einsum, std::vector<int>{X});
    std::cout << "exp\n";

    // Create ReLU(Y)
    scalarop_t relu_op = scalarop_t::make_relu();
    einsummable_t relu_einsum({a, b}, vector<vector<int>>{{0,1}}, 2, relu_op);
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
    
                                                     
    {
      std::cout << "----------Initializing Both Matrix and Vector with Random Values---------" << std::endl << std::endl;
      args.set_default<int>("nrep", 1);
      int nrep = args.get<int>("nrep");
      for(int rep = 0; rep != nrep; ++rep) {

        // initialize input tensors and distribute across the cluster
        for(int gid = 0; gid != graph.nodes.size(); ++gid) {
          auto const& node = graph.nodes[gid];
          if(node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.01", "0.01");
            //tensor.ones();
            server.insert_tensor(gid, placs1[gid], tensor);
          }
        }

        // Execution Engine
        std::cout << "----------Executing the Original Graph---------" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        // for(size_t i = 0; i < 20; ++i) {
        //   server.execute_graph(graph, placs1);
        // }
        server.execute_graph(graph, placs1);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        DOUT("Original Graph Finished. Time: " << duration.count() << " ms");
      }
    }

    // Do the Fusion
    fusion_t fusion;
    graph_t newGraph = fusion.apply(graph);
    // fusion_t::build_cuda_graph_for_fused_exp_relu(newGraph.nodes[1].op.get_einsummable);

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
    
    {
      std::cout << "----------Initializing Both Matrix and Vector with Random Values---------" << std::endl << std::endl;
      args.set_default<int>("nrep", 1);
      int nrep = args.get<int>("nrep");
      for(int rep = 0; rep != nrep; ++rep) {

        // initialize input tensors and distribute across the cluster
        for(int gid = 0; gid != newGraph.nodes.size(); ++gid) {
          auto const& node = newGraph.nodes[gid];
          if(node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.01", "0.01");
            //tensor.ones();
            server.insert_tensor(gid, placs2[gid], tensor);
          }
        }

        // Execution Engine
        std::cout << "----------Executing the Optimized Graph ---------" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        server.execute_graph(newGraph, placs2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        DOUT("Optimized Graph Finished. Time: " << duration.count() << " ms");
      }
    }

    server.shutdown();
  } else {
    server.listen();
  }

}

