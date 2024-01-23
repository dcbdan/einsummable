#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/trainer.h"

#include "../src/autoplace/autoplace.h"

void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize "
               "base_data_file num_data_files (Args)\n"
               "Args:\n";
}

void main_rank_zero(
  std::unique_ptr<server_base_t>& server,
  tensor_reader_t& model_loader,
  args_t& pargs,
  autoplace_config_t config)
{
  auto f_autoplace = [&config](
    graph_t const& graph,
    map<int, placement_t> const& fixed_pls,
    vector<tuple<int,int>> const& equal_pls)
  {
    return autoplace02(graph, config, fixed_pls, equal_pls);
  };

  pargs.set_default<int>("lora_rank", 32);
  int lora_rank = pargs.get<int>("lora_rank");

  model_args_t margs = model_args_t::llama(model_loader.num_files(), 1);

  pargs.set_default<int>("max_n_layers", -1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("batch_size", 1);
  margs.batch_size = pargs.get<uint64_t>("batch_size");

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0, lora_rank);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  // prediction: batch size, vocab size
  tensor_t prediction = model.forward(embeddings);
  tensor_t labels = writer.input(
    vector<uint64_t>{margs.batch_size, margs.vocab_size},
    prediction.get_dtype());

  // Compute the loss
  //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
  //   Loss = sum_n (l{n}) / N
  // Note, shift by c for numerical stability;
  //   where c{n} = max_c v{n,c}
  tensor_t loss;
  {
    dtype_t dtype = prediction.get_dtype();
    tensor_t v = prediction;
    tensor_t c = writer.reduction("bv->b", castable_t::max, v);
    // v = v - c
    v = writer.ew("bv,b->bv", scalarop_t::make_sub(dtype), v, c);
    // ev = exp(v)
    tensor_t ev = writer.ew(scalarop_t::make_exp(dtype), v);
    // evsubset{b} = sum_v ev{b,v}*labels{b,v}
    tensor_t evsubset = writer.contraction("bv,bv->b", ev, labels);
    tensor_t evsum    = writer.reduction("bv->b", castable_t::add, ev);

    tensor_t lll = writer.ew(
      "b,b->b",
      scalarop_t::make_div(dtype),
      evsubset, evsum);

    lll = writer.ew(scalarop_t::make_log(dtype), lll);

    // (would like to use unsqueeze here but it is not implemented)

    double one_over_bsz = 1.0 / double(margs.batch_size);
    loss = lll.scale(scalar_t(dtype, write_with_ss(one_over_bsz)));
  }

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
    if(name.find("lora") != string::npos) {
      // For the lora, we have (X*L0)*L1 where L0 needs to be
      // initialized gaussiann and L1 needs to be initialized with zeros
      auto shape = tensor.get_shape().full();
      dbuffer_t dbuffer = make_dbuffer(tensor.get_dtype(), product(shape));

      if(name.find("lora0") != string::npos) {
        dbuffer.rnorm();
      } else if(name.find("lora1") != string::npos) {
        dbuffer.zeros();
      } else {
        throw std::runtime_error("should not reach");
      }

      server->insert_tensor(tensor.get_id(), shape, dbuffer);
    } else {
      int next_tid = server->get_max_tid() + 1;
      relation_t relation = model_loader(
        register_cmd, name, tensor.get_shape().full(), next_tid);
      server->insert_gid_without_data(tensor.get_id(), relation);
    }
  }

  model_loader.shutdown(register_cmd);

  vector<int> trainer_weight_ids;
  vector<int> trainer_constant_ids;
  for(auto const& [name, tensor]: weight_map) {
    int id = tensor.get_id();
    if(name.find("lora") != string::npos) {
      trainer_weight_ids.push_back(id);
    } else if(name.find("norm") != string::npos) {
      trainer_constant_ids.push_back(id);
    } else {
      trainer_constant_ids.push_back(id);
    }
  }

  tensor_t full_freqs_cis = model.full_freqs_cis;
  server->insert_tensor(
    full_freqs_cis.get_id(),
    full_freqs_cis.get_shape().full(),
    transformer_t::form_full_freqs_cis(margs));

  trainer_constant_ids.push_back(full_freqs_cis.get_id());

  trainer_t trainer(
    server.get(),
    writer.get_graph(),
    loss.get_id(),
    vector<int>{loss.get_id()},                        // inspect
    vector<int>{embeddings.get_id(), labels.get_id()}, // data
    trainer_constant_ids,
    trainer_weight_ids,
    f_autoplace,
    dtype_t::f32,
    update_type_t::adamw);

  trainer.init();

  string tokenizer_file = pargs.get<string>("tokenizer");
  string dataset_file   = pargs.get<string>("dataset");
  dataset_reader_t data_loader(tokenizer_file, dataset_file);

  map<string, scalar_t> vars {
    { "beta1", scalar_t(dtype_t::f32, "0.9") },
    { "beta2", scalar_t(dtype_t::f32, "0.999") },
    { "eta", scalar_t(pargs.get<float>("learning_rate")) }
  };

  pargs.set_default<int>("niter", 2);
  int niter = pargs.get<int>("niter");
  for(int iter = 1; iter != niter + 1; ++iter) {
    // Insert the actual (embeddings,label) data
    // Note that embeddings will need to be selected from the embedding matrix
    // and the labels will need to be one-hot encoded
    auto [data_tokens, label_tokens] = data_loader.random_data(
      margs.batch_size, margs.max_seq_len);
    server->insert_tensor(
      embeddings.get_id(),
      embeddings.get_shape().full(),
      data_loader.make_embedding(
        embedding_matrix,
        vector_flatten(data_tokens)));

    server->insert_tensor(
      labels.get_id(),
      labels.get_shape().full(),
      data_loader.one_hot_encode(labels.get_dtype(), label_tokens));

    trainer(vars);

    double loss_val = server->get_tensor_from_gid(loss.get_id()).sum_to_f64();
    DOUT("loss: " << loss_val);
  }
}

int main(int argc, char** argv) {
  if(argc < 7) {
    usage();
    return 1;
  }
  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  string base_data_file(argv[5]);
  int num_data_files = parse_with_ss<int>(argv[6]);

  int num_threads = 8;//2; //std::max(1, int(std::thread::hardware_concurrency()));
  int num_channels = 4;
  int num_channels_per_move = 1;

  DOUT("n_locs " << world_size << " | num_threads_per_loc " << num_threads);

  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  std::unique_ptr<server_base_t> server;
  server = std::unique_ptr<server_base_t>(
    new cpu_tg_server_t(communicator, mem_size, num_threads));

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

  args_t args(argc-6, argv+6);

  int num_config_threads_per_machine = 8;
  DOUT("num config threads per machine " << num_config_threads_per_machine);
  autoplace_config_t config = autoplace_config_t::make_default02(
    world_size, num_config_threads_per_machine);

  main_rank_zero(server, reader, args, config);

  server->shutdown();
}

