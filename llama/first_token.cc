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
  args_t& pargs,
  int world_size);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);

  for(int i = 0; i != argc; ++i) {
    DOUT(i << ": " << argv[i]);
  }

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
    throw std::runtime_error("invalid server arg: " + which_server);
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
  args.set_default<string>("data_filename", argv[0]);

  main_rank_zero(server, reader, args, world_size);

  server->shutdown();
}

struct layer_ids_t {
  int w1;
  int w2;
  int w3;
  int wq;
  int wk;
  int wv;
  int wo;
  int fn;
  int an;
};

template <typename T>
void vector_repeat_resize(vector<T>& ret, int n) {
  if(ret.size() > n) {
    ret.resize(n);
    return;
  }
  if(ret.size() == n) {
    return;
  }

  if(ret.size() == 0) {
    throw std::runtime_error("cannot repeat; nothing in there");
  }

  ret.reserve(n);
  int i = 0;
  while(ret.size() < n) {
    ret.push_back(ret[i++]);
  }
}

void main_rank_zero(
  std::unique_ptr<server_base_t>& server,
  tensor_reader_t& model_loader,
  args_t& pargs,
  int world_size)
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

  bool actually_run = true;
  if(actually_run) {
    string register_cmd = server->get_registered_cmd();

    // fill out embeddings by reading the contents of data_filename
    // and creating a batchsize by embed size matrix
    {
      dbuffer_t embedding_matrix;
      vector<uint64_t> embedding_matrix_shape { margs.vocab_size, margs.dim };

      relation_t rel = model_loader(
        register_cmd,
        "tok_embeddings.weight",
        embedding_matrix_shape,
        server->get_max_tid() + 1);
      embedding_matrix = server->get_tensor(rel);

      string tokenizer_file = pargs.get<string>("tokenizer");
      just_tokenizer_t tokenizer(tokenizer_file);

      vector<int> tokens;
      {
        string filename = pargs.get<string>("data_filename");
        std::ifstream t(filename);
        std::stringstream buffer;
        buffer << t.rdbuf();
        tokens = tokenizer(buffer.str());
        DLINEOUT("total number of tokens in " << filename << " is " << tokens.size());
        vector_repeat_resize(tokens, margs.max_seq_len * margs.batch_size);
      }

      server->insert_tensor(
        embeddings.get_id(),
        embeddings.get_shape().full(),
        make_embedding(
          tokenizer.vocab_size(),
          embedding_matrix,
          tokens));
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
      transformer_t::form_position_interpolation_full_freqs_cis(margs, 2048));
  }

  graph_t const& graph = writer.get_graph();
  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("g.gv");
  }

  vector<placement_t> pls;
  {
    int num_config = pargs.get<int>("config_threads");

    pargs.set_default<string>("partitioner", "auto");
    string which = pargs.get<string>("partitioner");
    vector<partition_t> parts;

    if(which == "auto") {
      parts = apart01(graph, world_size * num_config, 1);
    } else if(which == "data" || which == "dim" || which == "seq") {
      // w1: hidden_dim, args.full_dim()
      // w2: args.full_dim(), hidden_dim
      // w3: hidden_dim, args.full_dim()
      //
      // wq: args.full_dim(), args.full_dim()
      // wk: args.full_dim(), args.full_dim()
      // wv: args.full_dim(), args.full_dim()
      // wo: args.full_dim(), args.full_dim()
      //
      // fn, an: args.full_dim()
      vector<layer_ids_t> layer_ids;
      for(auto const& layer: model.layers) {
        auto const& ff = layer.feedforward;
        auto const& aa = layer.attention;
        layer_ids.push_back(layer_ids_t {
          .w1 = ff.w1.get_id(),
          .w2 = ff.w2.get_id(),
          .w3 = ff.w3.get_id(),
          .wq = aa.wq.get_id(),
          .wk = aa.wk.get_id(),
          .wv = aa.wv.get_id(),
          .wo = aa.wo.get_id(),
          .fn = layer.attention_norm.weight.get_id(),
          .an = layer.feedforward_norm.weight.get_id()
        });
      }

      map<tuple<int, int>, partdim_t> pds;
      if(which == "data") {
        int id = embeddings.get_id();
        pds.insert({ {id,0}, partdim_t::split(margs.batch_size, num_config) });
      } else if(which == "dim") {
        int split_a = num_config;
        int split_b = 1;
        while(split_a > margs.n_heads) {
          if(split_a % 2 != 0) {
            throw std::runtime_error("make num config more even..");
          }
          split_a /= 2;
          split_b *= 2;
        }

        partdim_t pda = partdim_t::split(margs.n_heads, split_a);
        partdim_t pdb = partdim_t::split(margs.head_dim(), split_b);

        partdim_t pdb2 = partdim_t::split(margs.head_dim()/2, split_b);
        pds.insert({ { model.full_freqs_cis.get_id(), 1 }, pdb2});

        pds.insert({ {embeddings.get_id(), 2}, pda });
        pds.insert({ {embeddings.get_id(), 3}, pdb });
        pds.insert({ {model.norm.weight.get_id(), 0}, pda });
        pds.insert({ {model.norm.weight.get_id(), 1}, pdb });
        pds.insert({ {model.w_vocab.get_id(), 1}, pda });
        pds.insert({ {model.w_vocab.get_id(), 2}, pdb });
        for(auto const& [w1,w2,w3,wq,wk,wv,wo,fn,an]: layer_ids) {
          pds.insert({ {w1,1}, pda });  pds.insert({ {w1,2}, pdb });
          pds.insert({ {w2,0}, pda });  pds.insert({ {w2,1}, pdb });
          pds.insert({ {w3,1}, pda });  pds.insert({ {w3,2}, pdb });

          pds.insert({ {wq,0}, pda });  pds.insert({ {wq,1}, pdb });
          pds.insert({ {wk,0}, pda });  pds.insert({ {wk,1}, pdb });
          pds.insert({ {wv,0}, pda });  pds.insert({ {wv,1}, pdb });
          pds.insert({ {wo,0}, pda });  pds.insert({ {wo,1}, pdb });

          pds.insert({ {wq,2}, pda });  pds.insert({ {wq,3}, pdb });
          pds.insert({ {wk,2}, pda });  pds.insert({ {wk,3}, pdb });
          pds.insert({ {wv,2}, pda });  pds.insert({ {wv,3}, pdb });
          pds.insert({ {wo,2}, pda });  pds.insert({ {wo,3}, pdb });

          pds.insert({ {fn,0}, pda });  pds.insert({ {fn,1}, pdb });
          pds.insert({ {an,0}, pda });  pds.insert({ {an,1}, pdb });
        }
      } else if(which == "seq") {
        partdim_t pd = partdim_t::split(margs.max_seq_len, num_config);
        pds.insert({ { embeddings.get_id(), 1 }, pd });
        pds.insert({ { model.full_freqs_cis.get_id(), 0 }, pd});
        pds.insert({ { model.mask.value().get_id(), 0 }, pd});
        pds.insert({ { model.mask.value().get_id(), 1 }, pd});
      } else {
        throw std::runtime_error("missing case");
      }

      parts = apart03(graph, pds);
    } else {
      throw std::runtime_error("missing partitioner");
    }

    uint64_t flops_per_byte_moved = 1000;
    pls = alocate01(graph, parts, world_size, flops_per_byte_moved);
  }

  if(actually_run) {
    server->execute_graph(graph, pls);
  } else {
    auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph, pls);

    map<string, einsummable_t> es;
    cpu_kernel_executor_t ke;
    for(auto const& node: taskgraph.nodes) {
      if(node.op.is_apply()) {
        auto const& e = node.op.get_apply().einsummable;
        auto maybe = ke.build(e);
        if(!maybe) {
          es.insert({write_with_ss(e), e});
        }
      }
    }
    DOUT("MISSING KERNELS:");
    for(auto const& [_, e]: es) {
      DOUT("  " << e);
      DOUT("  " << std::get<0>(e.join.to_cpp_bytes()));
      DOUT("");
    }
  }
}

