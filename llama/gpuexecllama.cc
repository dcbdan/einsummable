#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

void main_rank_zero_matmul(
  gpu_mg_server_t& server,
  args_t& args)
{
  int this_rank = 0;

  // llama gpu parameters here
  args.set_default<int>("gpus", 1);
  args.set_default<int>("computes", 1);
  args.set_default<int>("nseq", 4096);
  args.set_default<int>("nbatch", 1);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  int nseq = args.get<int>("nseq");
  int nbatch = args.get<int>("nbatch");

  // print parameters
  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);
  DOUT("nseq:                            " << nseq);
  DOUT("nbatch:                          " << nbatch);


  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  // string register_cmd = server.get_registered_cmd();

  model_args_t model_args = model_args_t {
    .dim             = 4096, //was 4096
    .n_layers        = 1,
    .n_heads         = 32, //32
    .multiple_of     = 256, //256
    .norm_eps        = 1e-6,
    .batch_size      = 1,
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 32000,
  };

  //build the graph for feedforward only
  graph_writer_t writer;
  graph_t graph;
  uint64_t hidden_dim = 4 * model_args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    model_args.multiple_of * ( (hidden_dim + model_args.multiple_of - 1) / model_args.multiple_of );
  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(model_args.batch_size),
    full_dim_t::singleton(args.get<uint64_t>("nseq")),
    model_args.full_dim()
  }));
  feedforward_t feedforward(&writer, "feed_forward.", model_args.full_dim(), hidden_dim);
  // rms_norm_t feedforward_norm = rms_norm_t(&writer, "ffn_norm.", model_args.full_dim(), model_args.norm_eps);
  tensor_t scores = feedforward.forward(embeddings);
  scores.save_inplace();
  graph = writer.get_graph();
  {
    std::cout << "g.gv" << std::endl;
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
  }

  //make random input values 
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

  // if (nseq > margs.max_seq_len) {
  //   throw std::runtime_error("The sequence length is too long for the model parameters.");
  // }

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      model_args.n_layers = std::min(model_args.n_layers, n_layers);
    }
  }
  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);

  std::cout << "Inputs: " << inputs << std::endl;
  for(auto const& id: graph.get_inputs()) {
    std::cout << "id: " << id << "  placements: " << placements[id].locations << std::endl;
    server.insert_tensor(id, placements[id], input_data.at(id));
  }
  server.execute_graph(graph, placements);
}

void main_rank_zero(
  gpu_mg_server_t& server,
  args_t& args)
{
  int this_rank = 0;

  // llama gpu parameters here
  args.set_default<int>("gpus", 1);
  args.set_default<int>("computes", 1);
  args.set_default<int>("nseq", 4096);
  args.set_default<int>("nbatch", 1);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  int nseq = args.get<int>("nseq");
  int nbatch = args.get<int>("nbatch");

  // print parameters
  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);
  DOUT("nseq:                            " << nseq);
  DOUT("nbatch:                          " << nbatch);


  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  // string register_cmd = server.get_registered_cmd();

  model_args_t model_args = model_args_t {
    .dim             = 4096, //was 4096
    .n_layers        = 1,
    .n_heads         = 32, //32
    .multiple_of     = 256, //256
    .norm_eps        = 1e-6,
    .batch_size      = 1,
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 32000,
  };

  //build the graph for feedforward only
  graph_writer_t writer;
  graph_t graph;
  uint64_t hidden_dim = 4 * model_args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    model_args.multiple_of * ( (hidden_dim + model_args.multiple_of - 1) / model_args.multiple_of );
  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(model_args.batch_size),
    full_dim_t::singleton(args.get<uint64_t>("nseq")),
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

  //make random input values 
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

  // if (nseq > margs.max_seq_len) {
  //   throw std::runtime_error("The sequence length is too long for the model parameters.");
  // }

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      model_args.n_layers = std::min(model_args.n_layers, n_layers);
    }
  }
  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);

  std::cout << "Inputs: " << inputs << std::endl;
  for(auto const& id: graph.get_inputs()) {
    std::cout << "id: " << id << "  placements: " << placements[id].locations << std::endl;
    server.insert_tensor(id, placements[id], input_data.at(id));
  }
  server.execute_graph(graph, placements);
}

int main_rank_zero_llama(gpu_mg_server_t& server,
  args_t& args) {
  int this_rank = 0;

  // llama gpu parameters here
  args.set_default<int>("gpus", 4);
  args.set_default<int>("computes", 1);
  args.set_default<int>("nseq", 4096);
  args.set_default<int>("nbatch", 1);
  int num_gpus = args.get<int>("gpus");
  int num_computes_per_loc = args.get<int>("computes");
  int nseq = args.get<int>("nseq");
  int nbatch = args.get<int>("nbatch");

  // print parameters
  DOUT("num_gpus:                        " << num_gpus);
  DOUT("num_computes_per_loc:            " << num_computes_per_loc);
  DOUT("nseq:                            " << nseq);
  DOUT("nbatch:                          " << nbatch);

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  model_args_t model_args = model_args_t {
    .dim             = 4096, //was 4096
    .n_layers        = 32,
    .n_heads         = 32, //32
    .multiple_of     = 256, //256
    .norm_eps        = 1e-6,
    .batch_size      = 1,
    .max_seq_len     = 2048, //was 2048
    .vocab_size      = 32000,
  };

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      model_args.n_layers = std::min(model_args.n_layers, n_layers);
    }
  }

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

   
  autoplace_config_t config = autoplace_config_t::make_default01(
    num_gpus, num_computes_per_loc);
    vector<placement_t> placements = autoplace01(graph, config);

  std::cout << "Inputs: " << inputs << std::endl;
  for(auto const& id: graph.get_inputs()) {
    std::cout << "id: " << id << "  placements: " << placements[id].locations << std::endl;
    server.insert_tensor(id, placements[id], input_data.at(id));
  }
  server.execute_graph(graph, placements);

  return 0;
}

// ./gpu_llama 7B 1 max_n_layers n
int main(int argc, char** argv) {

  set_default_dtype(dtype_t::f32);

  if(argc < 3) {
    DOUT("argc " << argc);
    throw std::runtime_error("required args: "
       "(1)base_data_file       (2)num_data_files");
  }

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;

  if(is_rank_zero) {
    DOUT("world size:                      " << world_size);
  }

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  int this_rank = communicator.get_this_rank();

  args_t args(argc, argv);

  vector<uint64_t> buffer_sizes;
  // NOTE: 4 is hardcoded here since each anton has 4 gpus
  for (int i = 0; i < 4; ++i) {
    buffer_sizes.push_back(1lu * 1000lu * 1000lu * 1000lu);
  }

  gpu_mg_server_t server(communicator, buffer_sizes);

  // auto reader_process = [&](map<int, buffer_t> const& data_) {
  //   map<int, tuple<int, buffer_t>> data;
  //   for(auto const& [tid, buffer]: data_) {
  //     data.insert({tid, {this_rank, buffer}});
  //   }
  //   server.local_insert_tensors(data);
  // };

  // tensor_reader_t reader(
  //   communicator,
  //   reader_process,
  //   this_rank, world_size,
  //   base_data_file, num_data_files);

  args.set_default("parallel_partialize", false);
  server.set_parallel_partialize(args.get<bool>("parallel_partialize"));

  args.set_default("use_storage", true);
  server.set_use_storage(args.get<bool>("use_storage"));

  args.set_default("split_off_inputs", true);
  server.set_split_off_inputs(args.get<bool>("split_off_inputs"));

  // DOUT("parallel_partialize:             " << server.parallel_partialize_);
  DOUT("use_storage:                     " << server.use_storage_);
  DOUT("split_off_inputs:                " << server.split_off_inputs_);

  if(is_rank_zero) {
    main_rank_zero_matmul(server, args);
    
    server.shutdown();
  } else {
    // server.register_listen(
    //   reader.read_cmd(),
    //   [&]{ reader.listen_read(); });
    // server.register_listen(
    //   reader.shutdown_cmd(),
    //   [&]{ reader.listen_shutdown(); });

    server.listen();
  }
  return 0;
}