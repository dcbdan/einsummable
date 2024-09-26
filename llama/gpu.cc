#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

//#include <fstream>

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

struct token_maker_t {
  token_maker_t(vector<vector<int>> const ps):
    prompts(ps)
  {
    int bsz = prompts.size();
    if(bsz == 0) {
      throw std::runtime_error("must give atleast one prompt");
    }

    int seqlen = prompts[0].size();
    for(auto const& prompt: prompts) {
      seqlen = std::min(std::size_t(seqlen), prompt.size());
    }
    if(seqlen == 0) {
      throw std::runtime_error("cannot have empty prompt");
    }

    tokens = vtensor_t<int>({bsz, seqlen});
    for(int i = 0; i != bsz;    ++i) {
    for(int j = 0; j != seqlen; ++j) {
      tokens.at({i,j}) = prompts[i][j];
    }}
  }

  int batch_size() const {
    return tokens.get_shape()[0];
  }

  vtensor_t<int> const& get_tokens() const {
    return tokens;
  }

  vtensor_t<int> operator()(int start_pos, int end_pos) const {
    auto shape = tokens.get_shape();
    return tokens.subset({ {0, shape[0]}, {start_pos, end_pos} });
  }

  vtensor_t<int> last_column() const {
    auto shape = tokens.get_shape();
    return this->operator()(shape[1]-1, shape[1]);
  }

  void add_next_tokens(vtensor_t<int> const& next_tokens) {
    int startpos = tokens.get_shape()[0];

    tokens = vtensor_t<int>::concat(1, {tokens, next_tokens});

    // now apply the mask
    auto shape = tokens.get_shape();

    int bsz    = shape[0];
    int seqlen = shape[1] - startpos;

    for(int i = 0; i != bsz; ++i) {
      for(int j = startpos; j < prompts[i].size() && j < shape[1]; ++j) {
        tokens.at({i,j}) = prompts[i][j];
      }
    }
  }

private:
  vector<vector<int>> prompts;
  vtensor_t<int> tokens;
};

dbuffer_t lookup_embeddings(
  int nvocab,
  int nembed,
  dbuffer_t const& data,
  vtensor_t<int> tokens)
{
  if(data.nelem() != nvocab * nembed) {
    throw std::runtime_error("incorrectly sized data matrix");
  }

  int ntokens = product(tokens.get_shape());
  dbuffer_t ret = make_dbuffer(data.dtype, nembed * ntokens);

  char const* data_raw = reinterpret_cast<char const*>(data.raw());
  char      * ret_raw  = reinterpret_cast<char      *>(ret.raw());

  uint64_t stride = dtype_size(data.dtype) * nembed;

  for(int i = 0; i != ntokens; ++i) {
    int const& token = tokens.get()[i];
    if(token < 0 || token >= nvocab) {
      throw std::runtime_error("invalid token");
    }
    std::copy(
      data_raw + token    *stride,
      data_raw + (token+1)*stride,
      ret_raw  + i        *stride);
  }

  return ret;
}

template <typename T>
vtensor_t<int> _get_top_choices(
  T const* data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  topn = std::min(topn, ncol);
  vtensor_t<int> ret({int(nrow), int(topn)});
  int* ret_vec = ret.get().data();
  for(int i = 0; i != nrow; ++i) {
    T const* d = data    + i*ncol;
    int*     r = ret_vec + i*topn;
    vector<T const*> tops = select_topn(d, d + ncol, topn);
    for(int j = 0; j != topn; ++j) {
      r[j] = std::distance(d, tops[j]);
    }
  }
  return ret;
}

vtensor_t<int> get_top_choices(
  dbuffer_t const& data,
  uint64_t nrow,
  uint64_t ncol,
  uint64_t topn)
{
  if(data.dtype == dtype_t::f16) {
    return _get_top_choices(data.f16(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f32) {
    return _get_top_choices(data.f32(), nrow, ncol, topn);
  } else if(data.dtype == dtype_t::f64) {
    return _get_top_choices(data.f64(), nrow, ncol, topn);
  }
  throw std::runtime_error("get_top_choices: no dtype support here");
}

void _print_pl_info(
  string msg,
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto [_0, _1, taskgraph] =
    taskgraph_t::make(graph, placements);

  if(msg.size() < 45) {
    msg.resize(45, ' ');
  }

  int num_input_msgs = 0;
  uint64_t num_input_bytes = 0;
  int num_core_msgs = 0;
  uint64_t num_core_bytes = 0;
  set<int> inputs_everywhere = taskgraph.get_input_everywhere_ids();
  for(int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
    auto const& node = taskgraph.nodes[tid];
    if(node.op.is_move()) {
      uint64_t sz = node.op.get_move().size;
      if(inputs_everywhere.count(tid) > 0) {
        num_input_msgs++;
        num_input_bytes += sz;
      } else {
        num_core_msgs++;
        num_core_bytes += sz;
      }
    }
  }

  auto to_mb = [](uint64_t n) { return double(n)/1e6; };
  DOUT("(" << msg << ") input "
      << num_input_msgs << "#, " << to_mb(num_input_bytes) << "MB | core "
      << num_core_msgs << "#, " << to_mb(num_core_bytes) << "MB ");
}

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

// prompts = [
//   # For these prompts, the expected answer is the natural continuation of the prompt
//   "I believe the meaning of life is",
//   "Simply put, the theory of relativity states that ",
//   "Building a website can be done in 10 simple steps:\n",
//   """Translate English to French:
//      sea otter => loutre de mer
//      plush girafe => girafe peluche
//      cheese =>"""
// ]
vector<vector<int>> const _init_tokens {
  {1, 306, 4658, 278, 6593, 310, 2834, 338},
  {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
  {1, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 13},
  {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 268, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 268, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 268, 923, 968, 1149}
};

token_maker_t make_token_maker_with_shape(int nbatch, int nseq) {
  vector<vector<int>> tokens = _init_tokens;
  for(auto& ts: tokens) {
    vector_repeat_resize(ts, nseq);
  }
  vector_repeat_resize(tokens, nbatch);

  return token_maker_t(tokens);
}

token_maker_t make_default_token_maker() {
  return token_maker_t(_init_tokens);
}


void main_rank_zero(
  gpu_mg_server_t& server,
  tensor_reader2_t& reader,
  args_t& args)
{
  int this_rank = 0;

  // llama gpu parameters here
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

  token_maker_t token_maker = make_token_maker_with_shape(nbatch, nseq);

  vtensor_t<int> init_tokens = token_maker.get_tokens();
  // DOUT(init_tokens.get());

  uint64_t bsz    = init_tokens.get_shape()[0];
  uint64_t seqlen = init_tokens.get_shape()[1];

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  dtype_t dtype = default_dtype();

  model_args_t margs = model_args_t::llama(reader.num_files(), bsz);

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  margs.batch_size = bsz;
  margs.max_seq_len = seqlen;

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0);


  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  tensor_t predictions = model.forward(embeddings);
  predictions.save_inplace();

  graph_t graph1 = writer.get_graph();
  graph_t graph2 = writer.get_graph();
  for(auto const& [name, t]: model.weight_map()) {
    graph1.nodes[t.get_id()].op.set_save(true);
    graph2.nodes[t.get_id()].op.set_save(true);
  }
  {
    int const& gid = model.full_freqs_cis.get_id();
    graph1.nodes[gid].op.set_save(true);
    graph2.nodes[gid].op.set_save(true);
  }

  dbuffer_t embedding_matrix;
  dbuffer_t embeddings_data;
  {
    map<int, relation_t> relations;
    map<int, tuple<int, buffer_t>> local_data;
    int current_tid = 0;

    auto read_into = [&](
      string const& name, 
      vector<uint64_t> const& shape,
      map<int, tuple<int, buffer_t>>& local)
    {
      auto [rel, ds] = reader(name, shape, current_tid);
      current_tid += rel.placement.num_parts();

      vector<int> const& locs = rel.placement.locations.get();
      vector<int> const& tids = rel.tids.get();

      if(locs.size() != ds.size() || tids.size() != ds.size()) {
        throw std::runtime_error("bwoah.");
      }

      for(int i = 0; i != ds.size(); ++i) {
        int const& tid = tids[i];
        int const& loc = locs[i];
        auto& d        = ds[i];
        local.insert({ tid, { loc, d } });
      }

      return rel;
    };


    auto read_into_local_data = [&](string const& name, vector<uint64_t> const& shape) {
      return read_into(name, shape, local_data);
    };

    auto insert_name = [&](int gid, string const& name) {
      relation_t rel = read_into_local_data(name, graph1.nodes[gid].op.out_shape());
      relations.insert({gid, rel});
    };

    auto insert_local_buffer = [&](int gid, buffer_t data) {
      local_data.insert({ current_tid, { 0, data }});

      relation_t rel = relation_t::make_singleton(
        graph1.nodes[gid].op.out_dtype(),
        graph1.nodes[gid].op.out_shape(),
        current_tid);
      relations.insert({gid, rel});
      current_tid += 1;
    };


    {
      vector<uint64_t> shape{ margs.vocab_size, margs.dim };
      map<int, tuple<int, buffer_t>> ds;
      relation_t rel = read_into("tok_embeddings.weight", shape, ds);
      server.local_insert_tensors(ds);
      embedding_matrix = server.get_tensor(rel);
      server.local_erase_tensors(rel.tids.get());
    }

    for(auto const& [name, tensor]: model.weight_map()) {
      int gid = tensor.get_id();
      insert_name(gid, name);
    }

    {
      // TODO check this:
      // int const& gid = model.full_freqs_cis;
      int const& gid = model.full_freqs_cis.get_id();
      buffer_t freqs_cis = transformer_t::form_full_freqs_cis(margs).data;
      insert_local_buffer(gid, freqs_cis);
    }

    {
      int const& gid = embeddings.get_id();
      embeddings_data = lookup_embeddings(
        margs.vocab_size,
        margs.dim,
        embedding_matrix,
        init_tokens);
      insert_local_buffer(gid, embeddings_data.data);
    }

    server.local_insert_tensors(local_data);

    // At this point we've called local_insert_tensors on the server directly
    // or via the reader, so tell the server how it maps to gids
    for(auto const& [gid, rel]: relations) {
      server.insert_gid_without_data(gid, rel);
    }
  }

  vector<placement_t> pls;
  {
    args.set_default<string>("partitioner", "auto");
    string which = args.get<string>("partitioner");
    vector<partition_t> parts;

    int num_computes = num_computes_per_loc;

    if(which == "auto") {
      parts = apart01(graph1, num_gpus * num_computes, 1, 1, parts_space_t::contraction);
      //parts = apart01(graph, num_gpus, 1);
      // parts = apart01(graph, num_gpus, 1, 1, parts_space_t::all_range);
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
        pds.insert({ {id,0}, partdim_t::split(margs.batch_size, num_gpus * num_computes) });
      } else if(which == "dim") {
        int split_a = num_gpus * num_computes;
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
        partdim_t pd = partdim_t::split(margs.max_seq_len, num_gpus * num_computes);
        pds.insert({ { embeddings.get_id(), 1 }, pd });
        pds.insert({ { model.full_freqs_cis.get_id(), 0 }, pd});
        pds.insert({ { model.mask.value().get_id(), 0 }, pd});
        pds.insert({ { model.mask.value().get_id(), 1 }, pd});
      } else {
        throw std::runtime_error("missing case");
      }

      parts = apart03(graph1, pds);
    } else {
      throw std::runtime_error("missing partitioner");
    }

    //uint64_t flops_per_byte_moved = 1000;
    //pls = alocate01(graph, parts, num_gpus, flops_per_byte_moved);

    //vector<partition_t> parts = apart01(graph1, num_gpus, 1, 1, parts_space_t::contraction);

    if(which == "auto" && num_computes == 1) {
      pls = alocate03(graph1, parts, num_gpus, true);
    } else {
      uint64_t flops_per_byte_moved = 1000;
      pls = alocate01(graph1, parts, num_gpus, flops_per_byte_moved);
    }
  }

  {
    std::ofstream out_pls("pls.txt");
    for(auto const pl: pls) {
      out_pls << pl.locations.get() << std::endl;
    }
  }

  DLINEOUT("TIME TO RUN GRAPH1");
  server.execute_graph(graph1, pls);

//  server.insert_tensor(
//    embeddings.get_id(), 
//    pls[embeddings.get_id()],
//    embeddings_data);
//
//  DLINEOUT("TIME TO RUN GRAPH2");
//  server.execute_graph(graph2, pls);
}

// ./gpu_llama 7B 1 max_n_layers n
int main(int argc, char** argv) {

  set_default_dtype(dtype_t::f16);

  if(argc < 2) {
    DOUT("argc " << argc);
    throw std::runtime_error("required arg: base_data_file");
  }

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;

  string which_model = argv[1];
  string base_data_file = "/home/dcb/LLaMA/es/" + which_model;

  int num_data_files;
  if(which_model == "7B") {
    num_data_files = 1;
  } else if(which_model == "65B") {
    num_data_files = 8;
  } else {
    throw std::runtime_error("not sure how many data files...");
  }

  if(is_rank_zero) {
    DOUT("base data file                   " << base_data_file);
  }

  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  int this_rank = communicator.get_this_rank();

  args_t args(argc-1, argv+1);

  //args.set_default<int>("gpus", 8);
  int num_gpus = args.get<int>("gpus");

  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i) {
    buffer_sizes.push_back(args.get<uint64_t>("memsize") * 1000lu * 1000lu * 1000lu);
  }

  args.set_default("use_cudagraph", false);
  bool use_cudagraph = args.get<bool>("use_cudagraph");
  args.set_default<uint64_t>("storage", 4);
  auto storage_size = args.get<uint64_t>("storage") * 1000lu * 1000lu * 1000lu;

  gpu_mg_server_t server = storage_size > 0                                  ?
    gpu_mg_server_t(communicator, use_cudagraph, buffer_sizes, storage_size) :
    gpu_mg_server_t(communicator, use_cudagraph, buffer_sizes)               ;

  args.set_default("parallel_partialize", false);
  server.set_parallel_partialize(args.get<bool>("parallel_partialize"));

  args.set_default("split_off_inputs", true);
  server.set_split_off_inputs(args.get<bool>("split_off_inputs"));

  tensor_reader2_t reader(num_gpus, base_data_file, num_data_files);

  if(is_rank_zero) {
    main_rank_zero(server, reader, args);
  } else {
    server.listen();
  }

  server.shutdown();
}
