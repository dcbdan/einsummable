#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/dbuffer.h"

#include "../src/server/base.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/cpu/mg_server.h"

#include "../src/autoplace/autoplace.h"

map<int, dbuffer_t>
run(
  communicator_t& communicator,
  graph_t const& graph,
  string which_server,
  map<int, dbuffer_t> input_data,
  args_t& args)
{
  args.set_default<int>("num_threads", 4);
  int num_threads = args.get<int>("num_threads");
  uint64_t mem_size;
  std::unique_ptr<server_base_t> server;
  if(which_server == "mg") {
    // mem_size = args.get<uint64_t>("mem_size_mg");
    server = std::unique_ptr<server_base_t>(
      new cpu_mg_server_t(communicator, 12800, num_threads));
  } else if(which_server == "tg") {
    // uint64_t mem_size = args.get<uint64_t>("mem_size_tg");
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(communicator, 20000000, num_threads));
  } else {
    throw std::runtime_error("invalid server arg");
  }

  vector<placement_t> placements = autoplace01(
    graph,
    autoplace_config_t::make_default01(1, num_threads));

  for(auto const& id: graph.get_inputs()) {
    server->insert_tensor(id, placements[id], input_data.at(id));
  }

  server->execute_graph(graph, placements);

  map<int, dbuffer_t> output_data;
  for(int id = 0; id != graph.nodes.size(); ++id) {
    auto const& node = graph.nodes[id];
    if(node.op.is_save()) {
      dbuffer_t d = server->get_tensor_from_gid(id);
      output_data.insert({id, d});
    }
  }

  return output_data;
}

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f64); // more precision with f64 than f32

  args_t args(argc, argv);

  graph_t graph;
  {
    graph_writer_t writer;
    auto X0 = writer.input({20,20});
    auto X1 = writer.input({20,20});
    auto X2 = writer.input({20,20});
    auto X3 = writer.input({20,20});
    auto Y0 = writer.matmul(X0, X1);
    auto Y1 = writer.matmul(X2, X3);
    auto Z1 = writer.matmul(Y0, Y1).save();

    graph = writer.get_graph();
  }

  map<int, dbuffer_t> input_data;
  // int scale = 1;
  for(auto const& id: graph.get_inputs()) {
    dtype_t dtype = graph.out_dtype(id);
    auto shape = graph.out_shape(id);

    dbuffer_t d = make_dbuffer(dtype, product(shape));
    d.rnorm();
    d.scale(scalar_t(dtype, "0.0001"));
    // d.iota();
    // d.scale(scalar_t(dtype, "0.0001"));
    // d.ones();
    // d.scale(scalar_t(dtype, std::to_string(scale)));

    input_data.insert({id, d});
    // scale += 1;
  }

  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);
  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args);

  if (tg_tensor_results == mg_tensor_results) {
    std::cout << "The maps are equal." << std::endl;
  } else {
    std::cout << "The maps are not equal." << std::endl;
  }
  // DOUT("mg_tensor_results: ")
  // for (auto iter = mg_tensor_results.begin(); iter != mg_tensor_results.end(); ++iter) {
  //   auto buffer = iter->second;
  //   DOUT(buffer);
  // }
  
  // DOUT("tg_tensor_results: ")
  // for (auto iter = tg_tensor_results.begin(); iter != tg_tensor_results.end(); ++iter) {
  //   auto buffer = iter->second;
  //   DOUT(buffer);
  // }
}
