#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/cpu/mg_server.h"

#include "../src/autoplace/autoplace.h"
#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"

void main_rank_zero(
  std::unique_ptr<server_base_t>& server,
  tensor_reader_t& model_loader,
  args_t& pargs);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);

  int expected_argc = 12;
  if(argc < expected_argc) {
    return 1;
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  string which_server = parse_with_ss<string>(argv[5]);

  string base_data_file(argv[6]);
  int num_data_files = parse_with_ss<int>(argv[7]);

  int num_threads = parse_with_ss<int>(argv[8]);
  int num_contraction_threads = parse_with_ss<int>(argv[9]);

  int num_channels = parse_with_ss<int>(argv[10]);
  int num_channels_per_move = parse_with_ss<int>(argv[11]);

  DOUT("num_threads: " << num_threads);
  DOUT("num_contraction_threads: " << num_contraction_threads);
  DOUT("num_channels: " << num_channels);
  DOUT("num_channels_per_move: " << num_channels_per_move);

  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);
  int this_rank = communicator.get_this_rank();

  std::unique_ptr<server_base_t> server;
  if(which_server == "mg") {
    server = std::unique_ptr<server_base_t>(
      new cpu_mg_server_t(
        communicator, mem_size, num_threads, num_channels_per_move));
    DOUT("using mg server");
  } else if(which_server == "tg") {
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(
        communicator, mem_size, num_threads, num_contraction_threads));
    DOUT("using tg server and ignoring num_channels_per_move");
  } else {
    throw std::runtime_error("invalid server arg");
  }

  auto reader_process = [&](map<int, buffer_t> const& data_) {
    map<int, tuple<int, buffer_t>> data;
    for(auto const& [tid, buffer]: data_) {
      data.insert({tid, {this_rank, buffer}});
    }
    server->local_insert_tensors(data);
  };

  tensor_reader_t reader(
    communicator,
    reader_process,
    this_rank, world_size,
    base_data_file, num_data_files);

  if(!is_rank_zero) {
    server->register_listen(
      reader.read_cmd(),
      [&]{ reader.listen_read(); });
    server->register_listen(
      reader.shutdown_cmd(),
      [&]{ reader.listen_shutdown(); });

    server->listen();

    return 0;
  }

  args_t args(argc-(expected_argc-1), argv+(expected_argc-1));
  args.set_default<string>("filename", argv[0]);

  //args.set_default("config_threads", 8);
  //int num_config_threads_per_machine = args.get<int>("config_threads");
  //DOUT("num config threads per machine " << num_config_threads_per_machine);
  ////autoplace_config_t config = autoplace_config_t::make_default02(
  ////  world_size, num_config_threads_per_machine);
  //autoplace_config_t config = autoplace_config_t::make_default01(
  //  world_size, num_config_threads_per_machine);

  main_rank_zero(server, reader, args);

  server->shutdown();
}

void main_rank_zero(
  std::unique_ptr<server_base_t>& server,
  tensor_reader_t& model_loader,
  args_t& pargs)
{
  //
  pargs.set_default("simplify_tg", true);
  set_tg_do_simplify(pargs.get<bool>("simplify_tg"));
  //

  dtype_t dtype = default_dtype();

  model_args_t margs = model_args_t::llama(model_loader.num_files(), 1);

  pargs.set_default<int>("max_n_layers", -1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    DLINEOUT("n_layers " << n_layers);
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("batch_size", 1);
  margs.batch_size = pargs.get<uint64_t>("batch_size");

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  tensor_t predictions = model.forward(embeddings);
  predictions.save_inplace();

  string register_cmd = server->get_registered_cmd();

  dbuffer_t embedding_matrix;
  vector<uint64_t> embedding_matrix_shape { margs.vocab_size, margs.dim };
  {
    relation_t rel = model_loader(
      register_cmd,
      "tok_embeddings.weight",
      embedding_matrix_shape,
      server->get_max_tid() + 1);
    embedding_matrix = server->get_tensor(rel);
  }

  // Load the permanant weights and the lora weights
  auto weight_map = model.weight_map();
  for(auto const& [name, tensor]: weight_map) {
    int next_tid = server->get_max_tid() + 1;
    relation_t relation = model_loader(
      register_cmd, name, tensor.get_shape().full(), next_tid);
    server->insert_gid_without_data(tensor.get_id(), relation);
  }

  model_loader.shutdown(register_cmd);

  tensor_t full_freqs_cis = model.full_freqs_cis;
  server->insert_tensor(
    full_freqs_cis.get_id(),
    full_freqs_cis.get_shape().full(),
    //transformer_t::form_full_freqs_cis(margs));
    transformer_t::form_position_interpolation_full_freqs_cis(margs, 2048));

  string tokenizer_file = pargs.get<string>("tokenizer");

  just_tokenizer_t tokenizer(tokenizer_file);

  // TODO: need to insert the embeddings data, though...
}


