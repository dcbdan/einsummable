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
main_rank_zero(
  // graph_t graph,
  // model_args_t model_args,
  // map<int, dbuffer_t> input_data,
  std::unique_ptr<server_base_t>& server,
  args_t& args)
{

  args.set_default("seed", int(-1));
  int seed = args.get<int>("seed");
  if(seed >= 0) {
    set_seed(args.get<int>("seed"));
  }

  DOUT("here?");


  /* Create the llama first token graph using builder_t */
  model_args_t model_args = model_args_t {
    .dim             = 256, //was 4096
    .n_layers        = 1,
    .n_heads         = 4, //32
    .multiple_of     = 64, //256
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 1024,
  };
  DOUT("here2?");

  builder_t builder = builder_t::make_first_token(model_args, uint64_t(512));
  graph_t graph = builder.graph;
  DOUT("here3?");

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
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }


  std::cout << "Inputs: " << inputs << std::endl;

  vector<placement_t> placements = autoplace01(
  graph,
  autoplace_config_t::make_default01(args.get<int>("world_size"), args.get<int>("num_threads")));

  std::cout << "input size: " << graph.get_inputs().size() << std::endl;

  for(auto const& id: graph.get_inputs()) {
    std::cout << "id: " << id << "  placements: " << placements[id].locations << std::endl;
    server->insert_tensor(id, placements[id], input_data.at(id));
    DOUT("finish inserting?");
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
    mem_size = args.get<uint64_t>("mem_size_mg");
    server = std::unique_ptr<server_base_t>(
      new cpu_mg_server_t(communicator, mem_size, num_threads));
  } else if(which_server == "tg") {
    mem_size = args.get<uint64_t>("mem_size_tg");
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(communicator, mem_size, num_threads));
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

  // model_args_t model_args = model_args_t {
  //   .dim             = 4096, //was 4096
  //   .n_layers        = 1,
  //   .n_heads         = 32, //32
  //   .multiple_of     = 256, //256
  //   .norm_eps        = 1e-6,
  //   .batch_size      = args.get<uint64_t>("batch_size"),
  //   .max_seq_len     = 2048, //was 2048
  //   .vocab_size      = 32000,
  // };

    model_args_t model_args = model_args_t {
    .dim             = 256, //was 4096
    .n_layers        = 1,
    .n_heads         = 4, //32
    .multiple_of     = 64, //256
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 1024,
  };

  graph_writer_t writer;
  graph_t graph;


  uint64_t hidden_dim = 4 * model_args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    model_args.multiple_of * ( (hidden_dim + model_args.multiple_of - 1) / model_args.multiple_of );

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(model_args.batch_size),
    full_dim_t::singleton(args.get<uint64_t>("seq_len")),
    model_args.full_dim()
  }));
  feedforward_t feedforward(&writer, "feed_forward.", model_args.full_dim(), hidden_dim);
  rms_norm_t feedforward_norm = rms_norm_t(&writer, "ffn_norm.", model_args.full_dim(), model_args.norm_eps);
  tensor_t scores = feedforward.forward(embeddings);
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
    DOUT(dtype);
    DOUT(shape);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);

  DOUT("taskgraph done, starting memgraph");
  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 

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

  return(1);

}

int llama_main(int argc, char** argv) {

  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(512));

  /* Create the llama first token graph using builder_t */
  model_args_t model_args = model_args_t {
    .dim             = 4096, //was 4096
    .n_layers        = 1,
    .n_heads         = 32, //32
    .multiple_of     = 256, //256
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 32000,
  };


  builder_t builder = builder_t::make_first_token(model_args, uint64_t(512));
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
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);
  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 


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
  return(1);
}

int attention_main(int argc, char** argv){
  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(128));

  model_args_t model_args = model_args_t {
    .dim             = 256, //was 4096
    .n_layers        = 1,
    .n_heads         = 4, //32
    .multiple_of     = 64, //256
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 1024,
  };

  graph_writer_t writer;
  graph_t graph;

  tensor_t full_freqs_cis = writer.input(
    { 2*model_args.max_seq_len, uint64_div(model_args.head_dim(), 2) },
    dtype_t::c64);
  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(model_args.batch_size),
    full_dim_t::singleton(args.get<uint64_t>("seq_len")),
    model_args.full_dim()
  }));

  uint64_t start_pos = 0;
  
  /* get_freqs_cis from modules.cc*/
  using _all = graph_writer_t::idx_t::all;
  using _rng = graph_writer_t::idx_t::rng;
  using idx_t = graph_writer_t::idx_t;

  vector<uint64_t> shape = full_freqs_cis.get_shape()();

  vector<idx_t> subset(shape.size(), _all{});

  int64_t b0 = int64_t(start_pos);
  int64_t e0 = int64_t(start_pos + args.get<uint64_t>("seq_len"));
  subset[0] = _rng{ b0, e0 };

  tensor_t freqs_cis = full_freqs_cis.subset(subset);
  

  attention_t attention = attention_t(&writer, "attention.", model_args, start_pos, std::nullopt);
  tensor_t scores = attention.forward(embeddings, freqs_cis, std::nullopt);
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
    DOUT(dtype);
    DOUT(shape);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);

  DOUT("taskgraph done, starting memgraph");
  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 

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

  return(1);

}

int partial_attention_main(int argc, char** argv){
  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(128));

  model_args_t model_args = model_args_t {
    .dim             = 128, //was 4096
    .n_layers        = 1,
    .n_heads         = 4, //32
    .multiple_of     = 32, //256
    .norm_eps        = 1e-6,
    .batch_size      = args.get<uint64_t>("batch_size"),
    .max_seq_len     = 128, //was 2048
    .vocab_size      = 32000,
  };

  graph_writer_t writer;
  graph_t graph;

  tensor_t full_freqs_cis = writer.input(
    { 2*model_args.max_seq_len, uint64_div(model_args.head_dim(), 2) },
    dtype_t::c64);
  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(model_args.batch_size),
    full_dim_t::singleton(args.get<uint64_t>("seq_len")),
    model_args.full_dim()
  }));

  uint64_t start_pos = 0;
  
  /* get_freqs_cis from modules.cc*/
  using _all = graph_writer_t::idx_t::all;
  using _rng = graph_writer_t::idx_t::rng;
  using idx_t = graph_writer_t::idx_t;

  vector<uint64_t> shape = full_freqs_cis.get_shape()();

  vector<idx_t> subset(shape.size(), _all{});

  int64_t b0 = int64_t(start_pos);
  int64_t e0 = int64_t(start_pos + args.get<uint64_t>("seq_len"));
  subset[0] = _rng{ b0, e0 };

  tensor_t freqs_cis = full_freqs_cis.subset(subset);
  

  attention_t attention = attention_t(&writer, "attention.", model_args, start_pos, std::nullopt);
  tensor_t score = attention.forward_test(embeddings, freqs_cis, std::nullopt);
  score.save_inplace();

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
    DOUT(dtype);
    DOUT(shape);
    dbuffer_t d = make_dbuffer(dtype, product(shape));
    if (dtype == dtype_t::c64) {
      d.random();
    } else {
      d.rnorm();
      d.scale(scalar_t(dtype, "0.1"));
    }
    input_data.insert({input_id, d});
  }

  std::cout << "Inputs: " << inputs << std::endl;
  

  /* Initializing some input data*/
  communicator_t communicator("0.0.0.0", true, 1);

  map<int, dbuffer_t> tg_tensor_results = run(communicator, graph, "tg", input_data, args);

  DOUT(tg_tensor_results.size());

  DOUT("taskgraph done, starting memgraph");
  map<int, dbuffer_t> mg_tensor_results = run(communicator, graph, "mg", input_data, args); 

  if (tg_tensor_results == mg_tensor_results) {
    std::cout << "The maps are equal." << std::endl;
  } else {
    std::cout << "The maps are not equal." << std::endl;
  }

  // DOUT("mg_tensor_results: ")
  // for (auto iter = mg_tensor_results.begin(); iter != mg_tensor_results.end(); ++iter) {
  //   auto buffer = iter->second;
  //   DOUT(iter->first);
  //   DOUT(buffer);
  // }
  
  // DOUT("tg_tensor_results: ")
  // for (auto iter = tg_tensor_results.begin(); iter != tg_tensor_results.end(); ++iter) {
  //   auto buffer = iter->second;
  //   DOUT(iter->first);
  //   DOUT(buffer);
  // }

  DOUT("Done");

  return(1);

}

int llama_world_size_main(int argc, char** argv) {
  // To run this function:
  // ./cpuexecllama mem_size_tg 100000 mem_size_mg 6000 rank n world_size n

  args_t args(argc, argv);
  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(512));
  args.set_default("addr_zero", "0.0.0.0");
  bool is_rank_zero = args.get<int>("rank") == 0;
  int world_size = args.get<int>("world_size");
  string which_server = args.get<string>("which");
  int num_threads = args.get<int>("num_threads");
  int num_channels = 4;
  int num_channels_per_move = 1;
  DOUT("n_locs " << world_size << " | num_threads_per_loc " << num_threads);

  communicator_t communicator(args.get<string>("addr_zero"), is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  uint64_t mem_size;
  std::unique_ptr<server_base_t> server;
  if(which_server == "mg") {
    mem_size = args.get<uint64_t>("mem_size_mg");
    server = std::unique_ptr<server_base_t>(
      new cpu_mg_server_t(communicator, mem_size, num_threads));
  } else if(which_server == "tg") {
    mem_size = args.get<uint64_t>("mem_size_tg");
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(communicator, mem_size, num_threads));
  } else {
    throw std::runtime_error("invalid server arg");
  }

  if(!is_rank_zero) {
    DOUT("befire listen");
    server->listen();
    DOUT("after listen");
    return 0;
  }

  DOUT("before calling main_rank_zero");
  map<int, dbuffer_t> results = main_rank_zero(server, args);
  DOUT("results: ")
  for (auto iter = results.begin(); iter != results.end(); ++iter) {
    auto buffer = iter->second;
    DOUT(buffer);
  }

  server->shutdown();
  return(1);
}

int main(int argc, char** argv) {
  return llama_world_size_main(argc, argv);
  // feed_forward_main(argc, argv);
  // return attention_main(argc, argv);
  // return partial_attention_main(argc, argv);
  // return llama_main(argc, argv);
  return(1);
}