#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/server/cpu/mg_server.h"

#include "../src/misc/checkpoint.h"
#include "../src/misc/update.h"

#include "../src/autoplace/autoplace.h"

void usage() {
  std::cout << "Usage: addr_zero is_client world_size memsize "
               "base_data_file num_data_files (Args)\n"
               "Args:\n";
}

void main_rank_zero(
  server_base_t* server,
  tensor_reader_t& model_loader,
  args_t& pargs,
  autoplace_config_t config);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f16);

  int expected_argc = 10;
  if(argc < expected_argc) {
    usage();
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

  DOUT("n_locs " << world_size << " | num_threads_per_loc " << num_threads);

  // TODO: num channels, num channels per move?
  communicator_t communicator(addr_zero, is_rank_zero, world_size);
  int this_rank = communicator.get_this_rank();

  std::unique_ptr<server_base_t> server;
  if(which_server == "mg") {
    server = std::unique_ptr<server_base_t>(
      new cpu_mg_server_t(communicator, mem_size, num_threads));
  } else if(which_server == "tg") {
    server = std::unique_ptr<server_base_t>(
      new cpu_tg_server_t(communicator, mem_size, num_threads, num_contraction_threads));
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

  args.set_default("config_threads", 8);
  int num_config_threads_per_machine = args.get<int>("config_threads");
  DOUT("num config threads per machine " << num_config_threads_per_machine);
  autoplace_config_t config = autoplace_config_t::make_default01(
    world_size, num_config_threads_per_machine);

  main_rank_zero(server.get(), reader, args, config);

  server->shutdown();

  return 0;
}

struct graph_setup_t {
  model_args_t margs;
  checkpoint_graphs_t checkpoint_graphs;
  updater_desc_t updater_desc;
  vector<tuple<int, int>> next_iter_remap;
  vector<tuple<int, fill_t>> init_fills;
  int embeddings_id;
  int predictions_id;
  int labels_id;
  int loss_id;
  int full_freqs_cis_id;
  vector<tuple<string, int>> model_weight_map;

  vector<uint64_t> get_shape(int id) const {
    return checkpoint_graphs.full_graph.out_shape(id);
  }
};

graph_setup_t make_graph(
  args_t& pargs,
  int llama_num_files,
  uint64_t batch_size)
{
  dtype_t dtype = default_dtype();

  pargs.set_default("seed", int(-1));
  int seed = pargs.get<int>("seed");
  if(seed >= 0) {
    set_seed(pargs.get<int>("seed"));
  }

  pargs.set_default<int>("lora_rank", 32);
  int lora_rank = pargs.get<int>("lora_rank");

  model_args_t margs = model_args_t::llama(llama_num_files, batch_size);

  pargs.set_default<int>("max_n_layers", -1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    DOUT("n_layers " << n_layers);
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  graph_t graph;
  vector<int> checkpoints;
  vector<int> weight_ids;
  vector<int> grad_ids;
  set<int> forward_ids;

  int embeddings_id = -1;
  int predictions_id = -1;
  int labels_id = -1;
  int loss_id = -1;
  int full_freqs_cis_id = -1;
  vector<int> constant_ids;
  vector<tuple<string, int>> model_weight_map;
  {
    graph_writer_t writer;
    transformer_t model(&writer, margs, 0, lora_rank);

    tensor_t embeddings = writer.input(full_shape_t({
      full_dim_t::singleton(margs.batch_size),
      full_dim_t::singleton(margs.max_seq_len),
      margs.full_dim()
    }));

    // predictions: batch size, vocab size
    tensor_t predictions = model.forward(embeddings);
    tensor_t labels = writer.input(
      vector<uint64_t>{margs.batch_size, margs.vocab_size},
      dtype);

    // Compute the loss
    //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
    //   Loss = sum_n (l{n}) / N
    // Note, shift by c for numerical stability;
    //   where c{n} = max_c v{n,c}
    tensor_t loss;
    {
      tensor_t v = predictions;
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

    vector<tensor_t> ws;
    vector<tensor_t> cs;
    for(auto [name, tensor]: model.weight_map()) {
      model_weight_map.emplace_back(name, tensor.get_id());

      if(name.find("lora") != string::npos) {
        ws.push_back(tensor);
      } else {
        if(lora_rank) {
          // we're doing lora, so explicitly save all the weights
          tensor.save_inplace();
          // and add them as a constant
          cs.push_back(tensor);
        }
      }
    }

    {
      vector<int> fs = vector_iota<int>(writer.get_graph().nodes.size());
      forward_ids = set<int>(fs.begin(), fs.end());
    }

    vector<tensor_t> grads = writer.backprop(loss, ws);

    checkpoints = vector_from_each_method(model.checkpoints, int, get_id);
    weight_ids  = vector_from_each_method(ws, int, get_id);
    grad_ids    = vector_from_each_method(grads, int, get_id);

    embeddings_id = embeddings.get_id();
    predictions_id = predictions.get_id();
    labels_id = labels.get_id();
    loss_id = loss.get_id();

    full_freqs_cis_id = model.full_freqs_cis.get_id();
    model.full_freqs_cis.save_inplace();

    constant_ids = vector_from_each_method(cs, int, get_id);

    graph = std::move(writer.get_graph());
  }

  pargs.set_default<bool>("is_adamw", true);
  bool is_adamw = pargs.get<bool>("is_adamw");

  updater_desc_t::vanilla_t vanilla {};
  updater_desc_t::adamw_t adamw { .min_precision = dtype_t::f64 };
  updater_desc_t updater_desc = is_adamw ?
    updater_desc_t { dtype, adamw   }    :
    updater_desc_t { dtype, vanilla }    ;

  vector<tuple<int, int>> old_news;
  for(auto const& constant_id: constant_ids) {
    old_news.emplace_back(constant_id, constant_id);
  }

  old_news.emplace_back(full_freqs_cis_id, full_freqs_cis_id);

  vector<tuple<int, fill_t>> init_fills = update_weights(
    updater_desc, graph, old_news, weight_ids, grad_ids);

  // Note that update_weights may add input nodes, which must belong
  // to the forward graph
  for(auto const& [new_input_id, _]: init_fills) {
    forward_ids.insert(new_input_id);
  }

  // Make sure that every new is a save
  for(auto const& [old_id, new_id]: old_news) {
    graph.nodes[new_id].op.set_save(true);
  }

  checkpoint_graphs_t checkpoint_graphs(
    std::move(graph),
    checkpoints,
    forward_ids);

  return graph_setup_t {
    .margs = margs,
    .checkpoint_graphs = checkpoint_graphs,
    .updater_desc = updater_desc,
    .next_iter_remap = old_news,
    .init_fills = init_fills,
    .embeddings_id = embeddings_id,
    .predictions_id = predictions_id,
    .labels_id = labels_id,
    .loss_id = loss_id,
    .full_freqs_cis_id = full_freqs_cis_id,
    .model_weight_map = model_weight_map
  };
}

void main_rank_zero(
  server_base_t* server,
  tensor_reader_t& model_loader,
  args_t& pargs,
  autoplace_config_t config)
{
  dtype_t dtype = default_dtype();

  //
  pargs.set_default("simplify_tg", true);
  set_tg_do_simplify(pargs.get<bool>("simplify_tg"));
  //

  pargs.set_default("which", vector<int>());
  vector<int> which_data = pargs.get<vector<int>>("which");

  uint64_t batch_size;
  if(which_data.size() > 0) {
    batch_size = which_data.size();
  } else {
    pargs.set_default<uint64_t>("batch_size", 1);
    batch_size = pargs.get<uint64_t>("batch_size");
  }

  auto info = make_graph(pargs, model_loader.num_files(), batch_size);

  // Used to figure out required shapes
  auto const& margs = info.margs;

  // Use graphs to get checkpoint_taskgraphs_t
  auto const& graphs = info.checkpoint_graphs;

  // updater_desc will be used to set learning rate scalar variables
  auto const& updater_desc = info.updater_desc;

  // these fills need to be added before executing
  auto const& init_fills = info.init_fills;

  // used for the next remap
  auto const& next_iter_remap = info.next_iter_remap;

  DLINEOUT("autoplace01..");
  vector<placement_t> full_pls = autoplace01(graphs.full_graph, config);
  DLINEOUT("making the taskgraphs..");
  checkpoint_taskgraphs_t taskgraphs(graphs, full_pls);

  /////////////////////////////////////////////////////////////////////////////
  // Read in all the tensors
  DLINEOUT("init tensors");
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

  for(auto const& [name, id]: info.model_weight_map) {
    auto shape = info.get_shape(id);
    if(name.find("lora") != string::npos) {
      // For the lora, we have (X*L0)*L1 where L0 needs to be
      // initialized gaussiann and L1 needs to be initialized with zeros
      dbuffer_t dbuffer = make_dbuffer(dtype, product(shape));

      if(name.find("lora0") != string::npos) {
        dbuffer.rnorm();
        dbuffer.scale(scalar_t(dtype, write_with_ss(float(1e-3))));
      } else if(name.find("lora1") != string::npos) {
        dbuffer.zeros();
      } else {
        throw std::runtime_error("should not reach");
      }

      server->insert_tensor(id, shape, dbuffer);
    } else {
      int next_tid = server->get_max_tid() + 1;
      relation_t relation = model_loader(
        register_cmd, name, shape, next_tid);
      server->insert_gid_without_data(id, relation);
    }
  }

  model_loader.shutdown(register_cmd);

  for(auto const& [gid, fill]: init_fills) {
    if(!fill.is_constant()) {
      throw std::runtime_error("not implemented");
    }
    scalar_t const& value = fill.get_constant().value;
    server->insert_constant(gid, full_pls[gid], value);
  }

  server->insert_tensor(
    info.full_freqs_cis_id,
    info.get_shape(info.full_freqs_cis_id),
    transformer_t::form_position_interpolation_full_freqs_cis(margs, 2048));
  // TODO: this is how form_position_interpolation works?

  /////////////////////////////////////////////////////////////////////////////

  string tokenizer_file = pargs.get<string>("tokenizer");
  string dataset_file   = pargs.get<string>("dataset");

  dataset_reader_t data_loader(tokenizer_file, dataset_file);

  scalar_t _lr(dtype, write_with_ss(pargs.get<float>("learning_rate")));
  map<string, scalar_t> vars {
    { "beta1", scalar_t(dtype, "0.9") },
    { "beta2", scalar_t(dtype, "0.999") },
    { "eta", _lr },
    { "learning_rate", _lr },
  };

  pargs.set_default<int>("niter", 2);
  int niter = pargs.get<int>("niter");
  for(int iter = 1; iter != niter + 1; ++iter) {
    // Insert the actual (embeddings,label) data
    // Note that embeddings will need to be selected from the embedding matrix
    // and the labels will need to be one-hot encoded

    auto [data_tokens, label_tokens] = [&] {
      if(which_data.size() > 0) {
        vector<vector<int>> data_tokens;
        vector<int> label_tokens;
        for(auto const& which_datum: which_data) {
          auto [datum_tokens, label_token] =
            data_loader.datum(which_datum, margs.max_seq_len);
          data_tokens.push_back(datum_tokens);
          label_tokens.push_back(label_token);
        }
        return tuple<vector<vector<int>>, vector<int>>(data_tokens, label_tokens);
      }
      return data_loader.random_data(margs.batch_size, margs.max_seq_len);
    }();

    server->insert_tensor(
      info.embeddings_id,
      info.get_shape(info.embeddings_id),
      data_loader.make_embedding(
        embedding_matrix,
        vector_flatten(data_tokens)));

    server->insert_tensor(
      info.labels_id,
      info.get_shape(info.labels_id),
      data_loader.one_hot_encode(dtype, label_tokens));

    update_vars(updater_desc, iter, vars);
    for(int which = 0; which != taskgraphs.infos.size(); ++which) {
      server->remap_gids(graphs.remaps[which]);
      auto const& [init_rels, taskgraph, save_rels] = taskgraphs.infos[which];
      server->remap(init_rels);
      server->execute(taskgraph, save_rels, vars);
    }
    server->remap_gids(graphs.remaps.back());

    double loss_val = server->get_tensor_from_gid(info.loss_id).sum_to_f64();
    DOUT("loss: " << loss_val);

    server->remap_gids(next_iter_remap);
  }
}



