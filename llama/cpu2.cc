#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/cpu/server.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"

//#include <fstream>

//
#include "../src/engine/cpu/kernel_executor.h"

void do_a_matmul() {
  DLINEOUT("doin a matmul...");
  dbuffer_t x = make_dbuffer(dtype_t::f32, 1000*1000);
  dbuffer_t y = make_dbuffer(dtype_t::f32, 1000*1000);
  dbuffer_t z = make_dbuffer(dtype_t::f32, 1000*1000);

  x.random("-0.00001", "0.00001");
  y.random("-0.00001", "0.00001");

  matrix_multiply(
    dtype_t::f32,
    1000, 1000, 1000,
    false, false,
    z.raw(), x.raw(), y.raw());
}
//

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

exec_state_t::priority_t parse_priority_type(string const& val) {
  if(val == "given") {
    return exec_state_t::priority_t::given;
  } else if(val == "bfs") {
    return exec_state_t::priority_t::bfs;
  } else if(val == "dfs") {
    return exec_state_t::priority_t::dfs;
  } else if(val == "random") {
    return exec_state_t::priority_t::random;
  } else {
    throw std::runtime_error("invalid exec state value");
  }
}

parts_space_t parse_parts_space(string const& space_) {
  if(space_ == "contraction") {
    return parts_space_t::contraction;
  } else if(space_ == "all") {
    return parts_space_t::all;
  } else if(space_ == "all_range") {
    return parts_space_t::all_range;
  } else {
    throw std::runtime_error("invalid space_");
  }
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

struct llama_autoplacer_t {
  llama_autoplacer_t(int world_size, int num_threads_per, args_t& args)
    : world_size(world_size), num_threads_per(num_threads_per)
  {
    args.set_default("space", "contraction");
    space = parse_parts_space(args.get<string>("space"));

    args.set_default("double_workers", false);
    double_workers = args.get<bool>("double_workers");

    args.set_default("max_branching", int(1));
    max_branching = args.get<int>("max_branching");

    args.set_default("flops_per_byte_moved", 100);
    flops_per_byte_moved = args.get<int>("flops_per_byte_moved");
  }

  vector<placement_t> operator()(graph_t const& graph) const {
    int multiplier = double_workers ? 2 : 1 ;

    gremlin_t* gremlin_parts = new gremlin_t("parts");
    auto parts = apart01(
      graph,
      multiplier * world_size * num_threads_per,
      max_branching,
      space);
    delete gremlin_parts;

    //{
    //  std::ofstream f("g_parts.gv");
    //  graph.print_graphviz(f, parts);
    //  DOUT("printed g_parts.gv");
    //}

    gremlin_t gremlin_locate("locate");
    uint64_t flops_per_byte_moved = 100;
    auto ret = alocate01(
      graph, parts, world_size, flops_per_byte_moved);
    _print_pl_info("agg-at-a-time-from-inns 100", graph, ret);
    return ret;
  }

  int world_size;
  int num_threads_per;
  parts_space_t space;
  bool double_workers;
  int max_branching;
  uint64_t flops_per_byte_moved;
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

// run llama to compute
// (1) the first token and
// (2) the second token
void main_rank_zero_experiments(
  cpu_mg_server_t& server,
  tensor_reader_t& reader,
  llama_autoplacer_t const& autoplacer,
  args_t& args)
{
  int this_rank = server.comm.get_this_rank();

  uint64_t bsz    = args.get<uint64_t>("batch_size");
  uint64_t seqlen = args.get<uint64_t>("seq_len");
  DLINEOUT("bsz " << bsz << "  | seqlen " << seqlen);

  token_maker_t token_maker = make_token_maker_with_shape(bsz, seqlen);

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  set_default_dtype(dtype_t::f32);

  string register_cmd = server.get_registered_cmd();

  model_args_t margs = model_args_t::llama(reader.num_files(), bsz);

  // TODO: this may be off by one but I don't think it matters
  margs.max_seq_len = seqlen + 2;

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  builder_t builder = builder_t::make_first_token(margs, seqlen);

  dbuffer_t embedding_matrix;
  {
    map<int, relation_t> relations;
    int current_tid = 0;
    map<int, tuple<int, buffer_t>> local_data;

    {
      vector<uint64_t> shape{ margs.vocab_size, margs.dim };
      relation_t rel = reader(register_cmd, "tok_embeddings.weight", shape, current_tid);
      current_tid += rel.placement.num_parts();
      embedding_matrix = server.get_tensor(rel);
    }

    auto insert_reader_rel = [&relations, &current_tid](int gid, relation_t const& rel) {
      relations.insert({gid, rel});
      current_tid += rel.placement.num_parts();
    };
    auto insert_local_buffer = [&](int gid, buffer_t data) {
      local_data.insert({ current_tid, { this_rank, data }});

      relation_t rel = relation_t::make_singleton(
        builder.input_dtype(gid),
        builder.input_shape(gid),
        current_tid);
      relations.insert({gid, rel});
      current_tid += 1;
    };

    for(auto const& [name, gid]: builder.weights) {
      auto shape = builder.input_shape(gid);
      relation_t rel = reader(register_cmd, name, shape, current_tid);
      insert_reader_rel(gid, rel);
    }

    if(builder.mask) {
      int const& gid = builder.mask.value();
      buffer_t mask = transformer_t::form_start_mask(seqlen).data;
      insert_local_buffer(gid, mask);
    }

    {
      int const& gid = builder.freqs_cis;
      buffer_t freqs_cis = transformer_t::form_full_freqs_cis(margs).data;
      insert_local_buffer(gid, freqs_cis);
    }

    {
      int const& gid = builder.embeddings;
      dbuffer_t embeddings = lookup_embeddings(
        margs.vocab_size,
        margs.dim,
        embedding_matrix,
        token_maker.get_tokens());
      insert_local_buffer(gid, embeddings.data);
    }

    server.local_insert_tensors(local_data);

    // At this point we've called local_insert_tensors on the server directly
    // or via the reader, so tell the server how it maps to gids
    for(auto const& [gid, rel]: relations) {
      server.insert_gid_without_data(gid, rel);
    }
  }

  reader.shutdown(register_cmd);

  {
    vector<placement_t> pls = autoplacer(builder.graph);
    server.execute_graph(builder.graph, pls);
  }

  {
    dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

    uint64_t top_n = 1;
    vtensor_t<int> top_choices = get_top_choices(
      scores, bsz, margs.vocab_size, top_n);

    token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
  }

  for(int i = 0; i != 1; ++i) {
    builder = builder_t::make_next_token(builder);
    server.remap_gids(builder.remap.value());

    vector<placement_t> pls = autoplacer(builder.graph);

    {
      dbuffer_t embeddings = lookup_embeddings(
        margs.vocab_size,
        margs.dim,
        embedding_matrix,
        token_maker.last_column());

      server.insert_tensor(builder.embeddings, pls[builder.embeddings], embeddings);
    }

    server.execute_graph(builder.graph, pls);

    {
      dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

      uint64_t top_n = 1;
      vtensor_t<int> top_choices = get_top_choices(
        scores, bsz, margs.vocab_size, top_n);

      token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
    }
  }
}

// TODO: this is mostly just a copy of main_rank_zero_experiments
void main_rank_zero(
  cpu_mg_server_t& server,
  tensor_reader_t& reader,
  llama_autoplacer_t const& autoplacer,
  args_t& args)
{
  int this_rank = server.comm.get_this_rank();

  token_maker_t token_maker = make_default_token_maker();

  vtensor_t<int> init_tokens = token_maker.get_tokens();
  DOUT(init_tokens.get());

  uint64_t bsz    = init_tokens.get_shape()[0];
  uint64_t seqlen = init_tokens.get_shape()[1];

  {
    // Note: Assuming all is this being set?
    int seed = 99;//runif(10000);
    DOUT("Seed: " << seed);
    set_seed(seed);
  }

  set_default_dtype(dtype_t::f32);

  string register_cmd = server.get_registered_cmd();

  model_args_t margs = model_args_t::llama(reader.num_files(), bsz);
  // TODO: set the max seq len

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  builder_t builder = builder_t::make_first_token(margs, seqlen);

  dbuffer_t embedding_matrix;
  {
    map<int, relation_t> relations;
    int current_tid = 0;
    map<int, tuple<int, buffer_t>> local_data;

    {
      vector<uint64_t> shape{ margs.vocab_size, margs.dim };
      relation_t rel = reader(register_cmd, "tok_embeddings.weight", shape, current_tid);
      current_tid += rel.placement.num_parts();
      embedding_matrix = server.get_tensor(rel);
    }

    auto insert_reader_rel = [&relations, &current_tid](int gid, relation_t const& rel) {
      relations.insert({gid, rel});
      current_tid += rel.placement.num_parts();
    };
    auto insert_local_buffer = [&](int gid, buffer_t data) {
      local_data.insert({ current_tid, { this_rank, data }});

      relation_t rel = relation_t::make_singleton(
        builder.input_dtype(gid),
        builder.input_shape(gid),
        current_tid);
      relations.insert({gid, rel});
      current_tid += 1;
    };

    for(auto const& [name, gid]: builder.weights) {
      auto shape = builder.input_shape(gid);
      relation_t rel = reader(register_cmd, name, shape, current_tid);
      insert_reader_rel(gid, rel);
    }

    if(builder.mask) {
      int const& gid = builder.mask.value();
      buffer_t mask = transformer_t::form_start_mask(seqlen).data;
      insert_local_buffer(gid, mask);
    }

    {
      int const& gid = builder.freqs_cis;
      buffer_t freqs_cis = transformer_t::form_full_freqs_cis(margs).data;
      insert_local_buffer(gid, freqs_cis);
    }

    {
      int const& gid = builder.embeddings;
      dbuffer_t embeddings = lookup_embeddings(
        margs.vocab_size,
        margs.dim,
        embedding_matrix,
        init_tokens);
      insert_local_buffer(gid, embeddings.data);
    }

    server.local_insert_tensors(local_data);

    // At this point we've called local_insert_tensors on the server directly
    // or via the reader, so tell the server how it maps to gids
    for(auto const& [gid, rel]: relations) {
      server.insert_gid_without_data(gid, rel);
    }
  }

  reader.shutdown(register_cmd);

  {
    vector<placement_t> pls = autoplacer(builder.graph);
    server.execute_graph(builder.graph, pls);
  }

  {
    dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

    uint64_t top_n = 1;
    vtensor_t<int> top_choices = get_top_choices(
      scores, bsz, margs.vocab_size, top_n);

    token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
  }

  args.set_default("niter", int(100));
  int niter = args.get<int>("niter");
  for(int i = 0; i != niter; ++i) {
    builder = builder_t::make_next_token(builder);
    server.remap_gids(builder.remap.value());

    vector<placement_t> pls = autoplacer(builder.graph);

    {
      dbuffer_t embeddings = lookup_embeddings(
        margs.vocab_size,
        margs.dim,
        embedding_matrix,
        token_maker.last_column());

      server.insert_tensor(builder.embeddings, pls[builder.embeddings], embeddings);
    }

    server.execute_graph(builder.graph, pls);

    {
      dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

      uint64_t top_n = 1;
      vtensor_t<int> top_choices = get_top_choices(
        scores, bsz, margs.vocab_size, top_n);

      token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
    }
  }

  vtensor_t<int> const& tokens = token_maker.get_tokens();
  int nrow = tokens.get_shape()[0];
  for(int row = 0; row != nrow; ++row) {
    DOUT(tokens.index_subtensor(row).get());
  }
}

int main(int argc, char** argv) {
  // Sometimes mkl is slow on the first call. So
  // "wakeup" mkl by doing a matmul
  do_a_matmul();

  if(argc < 9) {
    DOUT("argc " << argc);
    throw std::runtime_error("required args: "
       "addr_zero is_client world_size "
       "memory_size(GB) "
       "num_channels num_channels_per_move "
       "base_data_file num_data_files");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
  uint64_t GB = 1000000000;
  mem_size *= GB;

  int num_channels = parse_with_ss<int>(argv[5]);
  int num_channels_per_move = parse_with_ss<int>(argv[6]);

  int num_threads = std::max(1, int(std::thread::hardware_concurrency()));

  string base_data_file(argv[7]);
  int num_data_files = parse_with_ss<int>(argv[8]);

  if(is_rank_zero) {
    DOUT("world size:                      " << world_size);
    DOUT("memory allocated:                " << (mem_size/GB) << " GB");
    DOUT("number of threads in threadpool: " << num_threads);
    DOUT("number of channels per move:     " << num_channels_per_move);
    DOUT("number of channels               " << num_channels);
    DOUT("base data file                   " << base_data_file);
    DOUT("num data files                   " << num_data_files);
  }

  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);

  int this_rank = communicator.get_this_rank();

  args_t args(argc-8, argv+8);

  args.set_default("priority_type", "given");
  exec_state_t::priority_t priority_type =
    parse_priority_type(args.get<string>("priority_type"));

  cpu_mg_server_t server(
    communicator, mem_size, num_threads, num_channels_per_move, priority_type);

  auto reader_process = [&](map<int, buffer_t> const& data_) {
    map<int, tuple<int, buffer_t>> data;
    for(auto const& [tid, buffer]: data_) {
      data.insert({tid, {this_rank, buffer}});
    }
    server.local_insert_tensors(data);
  };
  
  tensor_reader_t reader(
    communicator,
    reader_process,
    this_rank, world_size,
    base_data_file, num_data_files);

  args.set_default("parallel_partialize", false);
  server.set_parallel_partialize(args.get<bool>("parallel_partialize"));

  args.set_default("use_storage", false);
  server.set_use_storage(args.get<bool>("use_storage"));

  if(is_rank_zero) {
    // Assumption: all nodes have num_threads many threads
    llama_autoplacer_t autoplacer(world_size, num_threads, args);

    args.set_default("experiments", true);
    if(args.get<bool>("experiments")) {
      main_rank_zero_experiments(server, reader, autoplacer, args);
    } else {
      main_rank_zero(server, reader, autoplacer, args);
    }

    server.shutdown();
  } else {
    server.register_listen(
      reader.read_cmd(),
      [&]{ reader.listen_read(); });
    server.register_listen(
      reader.shutdown_cmd(),
      [&]{ reader.listen_shutdown(); });

    server.listen();
  }
}
