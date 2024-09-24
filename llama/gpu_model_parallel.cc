#include "misc.h"
#include "modules.h"
#include "reader.h"
#include "dataset_reader.h"

#include "../src/base/args.h"
#include "../src/server/gpu/server.h"

#include "../src/einsummable/taskgraph.h"

#include <fstream>
#include <cuda_runtime_api.h>
#include <stdexcept>

#define MRZ
#define CVR
#define CVROUT(x) 

using pl_t = model_parallel_placement_t;

tuple<
  map<int, int>, // convert init tids to these
  taskgraph_t,   // run this taskgraph
  map<int, int>> // convert output tids to these
convert_data_to_model_parallel(
  int nlocs,
  map<int, relation_t> const& init_rels,        // oid -> rels
  map<int, pl_t> const& model_pls,              // oid -> new representation
  map<int, vector<int>> const& oid_to_new_tids) // oid -> expected representation
{
  CVR;
  graph_t convert;
  vector<placement_t> pls;
  map<int, int> oid_to_cid;
  map<int, vector<int>> oid_to_out_cids;

  // insert the init state
  for(auto const& [oid, rel]: init_rels) {
    int cid = convert.insert_input(rel.placement.total_shape(), rel.dtype);
    oid_to_cid.insert({oid, cid});
    pls.push_back(rel.placement);
    CVROUT(oid << "," << cid);
  }
  CVR;

  // for each tensor in model_pls, convert it
  for(auto const& [oid, pl]: model_pls) {
    // Note: model_pls includes all inputs but also all constants
    auto iter = oid_to_cid.find(oid);
    if(iter == oid_to_cid.end()) {
      continue;
    }
    int cid_inn = iter->second;
    CVROUT(oid << "," << cid_inn);
    vector<uint64_t> shape = convert.out_shape(cid_inn);
    dtype_t dtype = convert.out_dtype(cid_inn);

    if(pl.split()) {
      partition_t part = pl.make_partition(shape, nlocs);
      for(int loc = 0; loc != nlocs; ++loc) {
        auto hrect = part.get_hrect(loc);

        int cid_out = convert.insert_subset(hrect, cid_inn);
        convert.nodes[cid_out].op.set_save(cid_out);

        placement_t pl(partition_t::singleton(hrect_shape(hrect)));
        pl.locations.get()[0] = loc;

        pls.push_back(pl);

        oid_to_out_cids[oid].push_back(cid_out);
      }
    } else {
      placement_t singleton_pl(partition_t::singleton(shape));
      for(int loc = 0; loc != nlocs; ++loc) {
        int cid_out = convert.insert_formation(cid_inn, true);

        placement_t copy = singleton_pl;
        copy.locations.get()[0] = loc;

        pls.push_back(copy);

        oid_to_out_cids[oid].push_back(cid_out);
      }
    }
  }
  CVR;

  auto [
    cid_to_inn_tids,
    cid_to_out_tids,
    convert_tg] = taskgraph_t::make(convert, pls);
  CVR;

  map<int, int> inn_tid_to_convert_tid;
  for(auto const& [oid, rel]: init_rels) {
    vector<int> const& inn_tids = rel.tids.get();

    int cid = oid_to_cid.at(oid);
    vector<int> const& convert_tids = cid_to_inn_tids.at(cid).get();

    if(inn_tids.size() != convert_tids.size()) {
      throw std::runtime_error("invalid number of tids...");
    }
    for(int i = 0; i != inn_tids.size(); ++i) {
      inn_tid_to_convert_tid.insert({inn_tids[i], convert_tids[i]});
    }
  }
  CVR;

  map<int, int> convert_tid_to_out_tid;
  for(auto const& [oid, out_tids]: oid_to_new_tids) {
    vector<int> const& out_cids = oid_to_out_cids.at(oid);

    vector<int> convert_tids;
    for(int const& out_cid: out_cids) {
      vector_concatenate_into(
        convert_tids, 
        cid_to_out_tids.at(out_cid).get());
    }

    if(convert_tids.size() != out_tids.size()) {
      throw std::runtime_error("invalid number of tids!");
    }

    for(int i = 0; i != out_tids.size(); ++i) {
      inn_tid_to_convert_tid.insert({convert_tids[i], out_tids[i]});
    }
  }

  CVR;

  return {
    inn_tid_to_convert_tid,
    convert_tg,
    convert_tid_to_out_tid
  };
}

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

void main_rank_zero(
  gpu_mg_server_t& server,
  tensor_reader2_t& reader,
  int num_files,
  uint64_t batch_size,
  uint64_t seqlen,
  int n_layers,
  int ngpus);

int main(int argc, char** argv) {
  set_default_dtype(dtype_t::f32);
  args_t pargs(argc, argv);

  string which_model = pargs.get<string>("model");
  int num_files;
  if(which_model == "7B") {
    num_files = 1;
  } else if(which_model == "13B") {
    num_files = 2;
  } else if(which_model == "30B") {
    num_files = 4;
  } else if(which_model == "65B") {
    num_files = 8;
  } else {
    throw std::runtime_error("BWOAH");
  }

  string base_data_file = "/home/dcb/LLaMA/es/" + which_model;

  int max_n_layers = pargs.get<int>("max_n_layers");
  uint64_t batch_size = pargs.get<uint64_t>("batch_size");
  int num_gpus = pargs.get<int>("gpus");

  string addr_zero = "0.0.0.0";
  bool is_rank_zero = true;
  int world_size = 1;
  communicator_t communicator(addr_zero, is_rank_zero, world_size);

  vector<uint64_t> buffer_sizes;
  for (int i = 0; i < num_gpus; ++i) {
    buffer_sizes.push_back(pargs.get<uint64_t>("memsize") * 1000lu * 1000lu * 1000lu);
  }

  bool use_cudagraph = false;
  gpu_mg_server_t server(communicator, use_cudagraph, buffer_sizes);

  pargs.set_default<uint64_t>("seq", 4096);
  uint64_t seqlen = pargs.get<uint64_t>("seq");
  DOUT("sequence length: " << seqlen);

  tensor_reader2_t reader(num_gpus, base_data_file, num_files);

  if(is_rank_zero) {
    main_rank_zero(server, reader, num_files, batch_size, seqlen, max_n_layers, num_gpus);
  } else {
    server.listen();
  }
}

void main_rank_zero(
  gpu_mg_server_t& server,
  tensor_reader2_t& reader,
  int num_files,
  uint64_t batch_size,
  uint64_t seqlen,
  int n_layers,
  int ngpus)
{
  MRZ;
  model_args_t margs = model_args_t::llama(num_files, batch_size);
  if(n_layers >= 0) {
    margs.n_layers = std::min(margs.n_layers, n_layers);
  }
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

  graph_t const& graph = writer.get_graph();
  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    DOUT("g.gv");
  }
  MRZ;

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

  map<int, pl_t> pls;

  DOUT("full freqs cis: " << model.full_freqs_cis.get_id());
  DOUT("mask          : " << model.mask.value().get_id());
  DOUT("embeddings:     " << embeddings.get_id());
  DOUT("norm:           " << model.norm.weight.get_id());
  DOUT("vocab           " << model.w_vocab.get_id());

  pls.insert({ model.full_freqs_cis.get_id(), pl_t::make_replicate() });
  if(model.mask) {
    pls.insert({ model.mask.value().get_id(), pl_t::make_replicate() });
  }

  pls.insert({embeddings.get_id(),        pl_t::make_replicate() });
  pls.insert({model.norm.weight.get_id(), pl_t::make_replicate() });
  pls.insert({model.w_vocab.get_id(),     pl_t::make_split(0)    });

  pls.insert({model.norm.weight.get_id(), pl_t::make_replicate()});

  for(auto const& [w1,w2,w3,wq,wk,wv,wo,fn,an]: layer_ids) {
    // w1: Col
    // w2: Row
    // w3: Col
    //
    // wq: Col
    // wk: Col
    // wv: Col
    // wo: Row
    //
    // 13 shape: hidden, full dim
    // 2  shape  full dim, hidden
    //
    // qkvo shape ( full dim x full dim )
    //
    // full_dim = (_,_)

    //DOUT("w1/w2/w3: " << w1 << "/" << w2 << "/" << w3);
    pls.insert({w1,pl_t::make_split(0)});
    pls.insert({w2,pl_t::make_split(2)});
    pls.insert({w3,pl_t::make_split(0)});

    //DOUT("wq/wk/wv/wo: " << wq << "/" << wk << "/" << wv << "/" << wo);
    pls.insert({wq,pl_t::make_split(0)});
    pls.insert({wk,pl_t::make_split(0)});
    pls.insert({wv,pl_t::make_split(0)});
    pls.insert({wo,pl_t::make_split(2)});

    pls.insert({fn,pl_t::make_replicate()});
    pls.insert({an,pl_t::make_replicate()});
  }

  MRZ;
  int nlocs = ngpus;
  auto [inns_model_parallel_tids, outs_model_parallel_tids, model_parallel_tg] =
    taskgraph_t::make_model_parallel(graph, nlocs, pls);
  MRZ;

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
      relation_t rel = read_into_local_data(name, graph.nodes[gid].op.out_shape());
      relations.insert({gid, rel});
    };

    auto insert_local_buffer = [&](int gid, buffer_t data) {
      local_data.insert({ current_tid, { 0, data }});

      relation_t rel = relation_t::make_singleton(
        graph.nodes[gid].op.out_dtype(),
        graph.nodes[gid].op.out_shape(),
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

    token_maker_t token_maker = make_token_maker_with_shape(batch_size, seqlen);
    vtensor_t<int> init_tokens = token_maker.get_tokens();
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
  MRZ;

  // At this point the server has all the gids in some format... We need
  // to duplicate some tensors and remap others...
  auto [tid_remap_init, convert_taskgraph, tid_remap_fini] =
    convert_data_to_model_parallel(
      nlocs, server.get_gid_map(), pls, inns_model_parallel_tids);
  MRZ;

  map<string, scalar_t> scalar_vals;

  server._remap_tids(tid_remap_init);
  MRZ;
  server.execute_tg_server(convert_taskgraph, scalar_vals);
  MRZ;
  server._remap_tids(tid_remap_fini);
  MRZ;

  server.execute_tg_server(model_parallel_tg, scalar_vals);
}


