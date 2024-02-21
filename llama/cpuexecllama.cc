#include "../src/base/args.h"

#include "../src/einsummable/gwriter.h"
#include "../src/einsummable/dbuffer.h"

#include "../src/server/base.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/cpu/mg_server.h"

#include "../src/autoplace/autoplace.h"
#include "../llama/modules.h"
#include "../llama/builder.h"

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
      new cpu_mg_server_t(communicator, 1338033000, num_threads));
  } else if(which_server == "tg") {
    // uint64_t mem_size = args.get<uint64_t>("mem_size_tg");
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(communicator, 13380321280000, num_threads));
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


int feed_forward_main(int argc, char** argv){
  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(512));

  model_args_t model_args = model_args_t {
    .dim             = 4096,
    .n_layers        = 1,
    .n_heads         = 32,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
  };

  graph_writer_t writer;
  graph_t graph;


  uint64_t hidden_dim = 4 * model_args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    model_args.multiple_of * ( (hidden_dim + model_args.multiple_of - 1) / model_args.multiple_of );

  feedforward_t feedforward(&writer, "feed_forward.", model_args.full_dim(), hidden_dim);
  rms_norm_t feedforward_norm = rms_norm_t(&writer, "ffn_norm.", model_args.full_dim(), model_args.norm_eps);
  tensor_t scores = feedforward.forward();
  scores.save_inplace();

  graph = writer.get_graph();

  {
    std::cout << "g.gv" << std::endl;
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
  }

  vector<int> inputs = graph.get_inputs();
  map<int, dbuffer_t> input_data;
  for (int input_id: inputs){
    dtype_t dtype = graph.out_dtype(input_id);
    auto shape = graph.out_shape(input_id);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.0001"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 
  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);

  if (tg_tensor_results == mg_tensor_results) {
    std::cout << "The maps are equal." << std::endl;
  } else {
    std::cout << "The maps are not equal." << std::endl;
  }

}

int llama_main(int argc, char** argv) {

  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(512));

  /* Create the llama first token graph using builder_t */
  model_args_t model_args = model_args_t {
    .dim             = 4096,
    .n_layers        = 1,
    .n_heads         = 32,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
  };

  builder_t builder = builder_t::make_first_token(model_args, uint64_t(128));
  graph_t graph = builder.graph;

  {
    std::cout << "g.gv" << std::endl;
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
  }

  vector<int> inputs = graph.get_inputs();
  map<int, dbuffer_t> input_data;
  for (int input_id: inputs){
    dtype_t dtype = graph.out_dtype(input_id);
    auto shape = graph.out_shape(input_id);
    DOUT(shape);
    DOUT(dtype);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.0001"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 
  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);

  if (tg_tensor_results == mg_tensor_results) {
    std::cout << "The maps are equal." << std::endl;
  } else {
    std::cout << "The maps are not equal." << std::endl;
  }

  DOUT("mg_tensor_results: ")
  for (auto iter = mg_tensor_results.begin(); iter != mg_tensor_results.end(); ++iter) {
    auto buffer = iter->second;
    DOUT(buffer);
  }
  
  DOUT("tg_tensor_results: ")
  for (auto iter = tg_tensor_results.begin(); iter != tg_tensor_results.end(); ++iter) {
    auto buffer = iter->second;
    DOUT(buffer);
  }
}


int main(int argc, char** argv) {
  feed_forward_main(argc, argv);
}