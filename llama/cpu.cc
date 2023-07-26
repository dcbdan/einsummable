#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/manager.h"
#include "../src/execution/cpu/kernels.h"
#include "../src/execution/cpu/permute.h"
#include "../src/execution/cpu/contraction.h"

#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/repartition.h"

#include "../src/autoplace/fsmcmc.h"
#include "../src/autoplace/rwmcmc.h"
#include "../src/autoplace/loadbalanceplace.h"
#include "../src/autoplace/autopart.h"

#include "reader.h"

#include <fstream>

#include <iomanip> // setprecision

#include <fstream>

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

int num_threads_per_node = 8;
int num_real_threads_per_node = 12;
int num_steps = 300000;
int nlocs = -1;
double beta = 10000.0;

vector<placement_t> autoplace(graph_t const& graph) {
  if(nlocs == -1) {
    throw std::runtime_error("need to set nlocs");
  }
  DOUT("num threads per node " << num_threads_per_node)
  auto kernel_coster = kernel_coster_t::for_cpu_cluster(nlocs);
  kernel_coster.touch_start = 1e-2;

  int max_blocks = num_threads_per_node * nlocs * 2;

  relationwise_mcmc_t mcmc(
    graph, kernel_coster,
    nlocs, num_threads_per_node, max_blocks,
    equal_items_t<int>());

  DOUT("single thread cost " << mcmc.cost());

  {
    uint64_t min_sizing = 1;
    vector<partition_t> parts = autopartition(
      graph, min_sizing, nlocs * num_threads_per_node, mcmc.get_equal_gids());

    // this ignores the equal gids
    vector<placement_t> pls = load_balanced_placement(graph, parts, nlocs, false);

    mcmc.set_placements(pls);
  }

  DOUT("balanced cost " << mcmc.cost());

  for(int i = 0; i != num_steps; ++i) {
    if(i % 5000 == 0) {
      DOUT( i << " / " << num_steps << "    " << mcmc.cost() << " " << mcmc.get_best_cost() );
    }
    mcmc.step(beta);
  }

  DOUT(num_steps << " / " << num_steps << "   " << mcmc.get_best_cost() );
  return mcmc.get_best_placements();
}

void main_(loc_manager_t& manager, tensor_reader_t& reader, string filename) {
  {
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  set_default_dtype(dtype_t::f16);

  auto convert_f16_to_default = [](buffer_t buffer) {
    if(default_dtype() != dtype_t::f16) {
      buffer = dbuffer_t(dtype_t::f16, buffer).copy(default_dtype()).data;
    }
    return buffer;
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
  token_maker_t token_maker({
    {1, 306, 4658, 278, 6593, 310, 2834, 338},
    {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
    {1, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 13},
    {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 268, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 268, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 268, 923, 968, 1149}
  });

  vtensor_t<int> init_tokens = token_maker.get_tokens();
  DOUT(init_tokens.get());

  uint64_t bsz    = init_tokens.get_shape()[0];
  uint64_t seqlen = init_tokens.get_shape()[1];

  model_args_t args = model_args_t::llama(reader.num_files(), bsz);

  int niter = 0; // 256-seqlen-1;

  DLINEOUT("getting the embedding matrix");

  dbuffer_t embedding_matrix = [&] {
    int starting_tid = manager.get_max_tid() + 1;
    vector<uint64_t> shape{ args.vocab_size, args.dim };
    relation_t relation = reader(
      manager.get_registered_cmd(), manager.mpi, manager.data,
      "tok_embeddings.weight", shape, starting_tid);
    buffer_t ret = convert_f16_to_default(manager.unpartition(relation).data);
    for(auto const& tid: relation.tids.get()) {
      manager.data.erase(tid);
    }
    return dbuffer_t(default_dtype(), ret);
  }();

  DOUT("making builder");

  builder_t builder = builder_t::make_first_token(args, seqlen, autoplace);
  uint64_t total = 0;
  for(auto const& node: builder.taskgraph.nodes) {
    if(node.op.is_partialize()) {
      for(auto const& [_, touch]: node.op.get_partialize().as_touches_from_flat()) {
        auto szs = vector_from_each_member(touch.selection, uint64_t, size);
        total += dtype_size(touch.dtype) * product(szs);
      }
    }
  }
  DOUT("total touch bytes: " << (double(total)*1.0e-9) << " GB");
  DOUT("taskgraph number of locs " << builder.taskgraph.num_locs());

  // TODO: tensor_reader_t reads everything in f16. Support when
  //       f16 isn't the default dtype by converting f16 read tensors
  //       into different default dtype then remapping the relations

  // need all weights, mask, freqcis, embeddings

  {
    remap_relations_t init_remap;

    DLINEOUT("getting weights");
    for(auto const& [name, usage_tinfo]: builder.weights) {
      auto shape = usage_tinfo.placement.total_shape();
      int starting_tid = manager.get_max_tid() + 1;
      init_remap.insert(
        reader(manager.get_registered_cmd(), manager.mpi, manager.data,
               name, shape, starting_tid),
        usage_tinfo);
    }

    DLINEOUT("got weights");
    auto insert_local = [&](relation_t const& tinfo, buffer_t buffer) {
      int id = manager.get_max_tid() + 1;
      manager.data.insert({ id, buffer });
      init_remap.insert(
        tinfo.as_singleton(id),
        tinfo);
    };


    if(builder.mask) {
      auto const& tinfo = builder.mask.value();
      buffer_t mask = transformer_t::form_start_mask(seqlen).data;
      insert_local(tinfo, mask);
    }

    {
      auto const& tinfo = builder.freqs_cis;
      buffer_t freqs_cis = transformer_t::form_full_freqs_cis(args).data;
      insert_local(tinfo, freqs_cis);
    }

    {
      buffer_t embeddings = lookup_embeddings(
        args.vocab_size,
        args.dim,
        embedding_matrix,
        init_tokens).data;

      auto const& tinfo = builder.embeddings;
      insert_local(tinfo, embeddings);
    }

    manager.remap_data(init_remap);
  }

  // all data has been read, so shut down the reader
  reader.shutdown(manager.get_registered_cmd(), manager.mpi);

  int world_size = bool(manager.mpi) ? manager.mpi->world_size : 1;
  if(builder.taskgraph.num_locs() > world_size) {
    throw std::runtime_error("the taskgraph has more locs than mpi rank");
  }

  DOUT("---");
  manager.execute(builder.taskgraph);

  {
    dbuffer_t scores = manager.unpartition(builder.scores);

    uint64_t top_n = 5;
    vtensor_t<int> top_choices = get_top_choices(
      scores, bsz, args.vocab_size, top_n);
    //DOUT(top_choices.get());

    token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
    DOUT(token_maker.get_tokens().get());
  }

  for(int i = 0; i != niter; ++i) {
    builder = builder_t::make_next_token(builder, (i + 1 == niter));

    //builder.print_info();

    manager.remap_data(builder.remap.value());

    {
      dbuffer_t embeddings = lookup_embeddings(
        args.vocab_size,
        args.dim,
        embedding_matrix,
        token_maker.last_column());

      manager.partition_into_data(builder.embeddings, embeddings);
    }

    DOUT("---");
    manager.execute(builder.taskgraph);

    {
      dbuffer_t scores = manager.unpartition(builder.scores);

      uint64_t top_n = 5;
      vtensor_t<int> top_choices = get_top_choices(
        scores, bsz, args.vocab_size, top_n);
      //DOUT(top_choices.get());

      token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
      DOUT(token_maker.get_tokens().get());
    }
  }

  DOUT(token_maker.get_tokens().get());
}

int main(int argc, char** argv) {
  //auto args = model_args_t::llama_30B();
  //builder_t builder = builder_t::make_first_token(args, 256, autoplace);

  //for(auto const& [name, rel]: builder.weights) {
  //  DOUT(name);
  //  DOUT(rel.placement.total_shape());
  //  DOUT("");
  //}

  auto settings = execute_taskgraph_settings_t::default_settings();
  settings.num_apply_runner = num_real_threads_per_node;

  mpi_t mpi(argc, argv);
  nlocs = mpi.world_size;

  loc_manager_t manager(&mpi, settings);

  if(argc != 3) {
    throw std::runtime_error("usage: base_filename n_files");
  }

  // reader holds data by reference
  tensor_reader_t reader(
    mpi.this_rank, mpi.world_size,
    argv[1], parse_with_ss<int>(argv[2]));

  if(mpi.this_rank != 0) {
    manager.register_listen(reader.read_cmd(),
      [&](loc_manager_t& m) {
        reader.listen_read(m.mpi, m.data);
      }
    );
    manager.register_listen(reader.shutdown_cmd(),
      [&](loc_manager_t& m) {
        reader.listen_shutdown();
      }
    );
    manager.listen();
  } else {
    main_(manager, reader, argv[1]);
    manager.shutdown();
  }
}
