#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "einsummable.pb.h"

#include "../src/base/args.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/memgraph.h"

#include "../src/server/cpu/server.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/autolinns.h"

#include "../src/engine/repartition.h"
#include "../src/engine/communicator.h"
#include "../src/engine/threadpool.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/notifier.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/exec_state.h"
#include "../src/engine/channel_manager.h"
#include "../src/engine/cpu/kernel_executor.h"
#include "../src/engine/cpu/workspace_manager.h"

string get_model_base_filename(int num_files) {
  if(num_files == 1) {
    return ""; // TODO
  } else if(num_files == 2) {
    return ""; // TODO
  } else if(num_files == 4) {
    return ""; // TODO
  } else if(num_files == 8) {
    return ""; // TODO
  } else {
    throw std::runtime_error("invalid number of files");
  }
}

struct event_t {
  event_t(){}

  template <typename T>
  event_t(T && t)
   : op(std::move(t))
  {}

  struct init_t {
    int world_size;
    uint64_t mem_size;
    int num_threads;
    int num_files;
    uint64_t batch_size;
    uint64_t seq_len;

    model_args_t make_model_args() const {
      model_args_t ret = model_args_t::llama(num_files, batch_size);
      // TODO: this may be off by one but I don't think it matters
      ret.max_seq_len = seq_len + 2;
      return ret;
    }
  };
  struct close_readers_t {
  };
  struct load_weight_t {
    string name;
    vector<memloc_t> data_locs;
  };
  struct load_data_matrix_t {
    uint64_t batch_size;
    uint64_t seq_len;
    uint64_t d_embed;
    mem_t mem;
  };
  struct load_mask_t {
    uint64_t seq_len;
    mem_t mem;
  };
  struct load_freqs_cis_t {
    uint64_t dim;
    uint64_t heads;
    uint64_t max_seq_len;
    mem_t mem;
  };
  struct execute_t {
    string message;
    memgraph_t memgraph;
  };
  struct build_next_data_matrix_t {
    partition_t src_partition;
    vector<memloc_t> src_data_locs;
    partition_t dst_partition;
    vector<memloc_t> dst_data_locs;
  };

  std::variant<
    init_t, close_readers_t,
    load_weight_t, load_data_matrix_t, load_mask_t, load_freqs_cis_t,
    execute_t, build_next_data_matrix_t
  > op;

  string to_wire() const;
  void to_proto(es_proto::InferenceEvent& e) const;

  static event_t from_wire(string const& str);
  static event_t from_proto(es_proto::InferenceEvent const& ie);

  bool is_init()                   const {
    return std::holds_alternative<init_t>(                  op); }
  bool is_close_readers()          const {
    return std::holds_alternative<close_readers_t>(         op); }
  bool is_load_weight()            const {
    return std::holds_alternative<load_weight_t>(           op); }
  bool is_load_data_matrix()       const {
    return std::holds_alternative<load_data_matrix_t>(      op); }
  bool is_load_mask()              const {
    return std::holds_alternative<load_mask_t>(             op); }
  bool is_load_freqs_cis()         const {
    return std::holds_alternative<load_freqs_cis_t>(        op); }
  bool is_execute()                const {
    return std::holds_alternative<execute_t>(               op); }
  bool is_build_next_data_matrix() const {
    return std::holds_alternative<build_next_data_matrix_t>(op); }

  init_t                   const& get_init()                   const {
    return std::get<init_t>(                  op); }
  close_readers_t          const& get_close_readers()          const {
    return std::get<close_readers_t>(         op); }
  load_weight_t            const& get_load_weight()            const {
    return std::get<load_weight_t>(           op); }
  load_data_matrix_t       const& get_load_data_matrix()       const {
    return std::get<load_data_matrix_t>(      op); }
  load_mask_t              const& get_load_mask()              const {
    return std::get<load_mask_t>(             op); }
  load_freqs_cis_t         const& get_load_freqs_cis()         const {
    return std::get<load_freqs_cis_t>(        op); }
  execute_t                const& get_execute()                const {
    return std::get<execute_t>(               op); }
  build_next_data_matrix_t const& get_build_next_data_matrix() const {
    return std::get<build_next_data_matrix_t>(op); }
};

struct recorder_t {
  recorder_t(event_t::init_t const& event_init)
    : _max_tid(0),
      world_size(event_init.world_size),
      mem_size(event_init.mem_size),
      num_files(event_init.num_files),
      alloc_settings(allocator_settings_t::default_settings())
  {
    events.emplace_back(event_init);
  }

  void read_data_matrix(int gid, uint64_t bsz, uint64_t seq_len, uint64_t d0, uint64_t d1);

  void read_weight(int gid, string name, vector<uint64_t> const& shape);

  void insert_mask(int gid, uint64_t seq_len);

  void insert_freqs_cis(
    int gid,
    uint64_t margs_dim, uint64_t margs_n_heads, uint64_t margs_max_seq_len);

  void execute_graph(
    graph_t const& graph,
    vector<placement_t> const& pls,
    string message);

  void build_next_relation(
    int scores_gids,
    builder_t const& next_builder,
    vector<placement_t> const& next_pls);

  void close_readers() { events.emplace_back(event_t::close_readers_t{}); }

  vector<event_t> const& get_events() const { return events; }

private:
  vector<memloc_t> insert_relation(int gid, dtype_t dtype, placement_t const& pl);

  vector<memloc_t> insert_relation(int gid, relation_t const& relation);

  // this loc is the given loc, so it could just return mem_t, but the memloc_t
  // is needed for insert_relation where this method is used
  memloc_t insert_data(int tid, int loc, uint64_t size);

  allocator_t& get_allocator(int loc);

  relation_t const& get_relation(int gid) const { return relations.at(gid); }

  void remap(remap_relations_t const& remap_relations, string message);

  void execute_taskgraph(taskgraph_t const& taskgraph, string message);

  vector<memloc_t> get_data_locs(vector<int> const& tids);

  void remap_gids(vector<tuple<int,int>> const& remap);

private:
  map<int, relation_t> relations;
  map<int, memloc_t> tensors;

  int _max_tid;

  int world_size;
  uint64_t mem_size;
  int num_files;

  vector<event_t> events;

  optional<vector<allocator_t>> _allocators;

  allocator_settings_t alloc_settings;
};

struct executor_t {
  executor_t(
    communicator_t& c,
    int n,
    exec_state_t::priority_t p)
    : comm(c),
      threadpool("_", std::max(1, int(std::thread::hardware_concurrency()))),
      num_channels_per_move(n), priority_type(p)
  {
    this_rank = comm.get_this_rank();
    world_size = comm.get_world_size();
  }

  // only for location zero to do
  void operator()(event_t const& event);
  void shutdown();

  // for every non zero location to do
  void listen();

private:
  void copy_into_data(buffer_t buffer, mem_t mem);

  void execute(memgraph_t const& memgraph, string message);

  void read_weight_chunk_server(
    int chunk,
    string name,
    memloc_t const& memloc);
  void read_weight_chunk(
    int chunk,
    string name,
    mem_t const& mem);

  void fill_embedding_matrix_server();

  map<int, buffer_t> fill_embedding_matrix(remap_relations_t const& remap);

  dbuffer_t get_tensor_server(
    partition_t const& src_part,
    vector<memloc_t> const& src_mems);

  void push_tensor_server(
    dbuffer_t buffer,
    partition_t dst_part,
    vector<memloc_t> const& dst_mems);

  map<int, buffer_t> get_data(
    partition_t const& src_part,
    vector<memloc_t> const& src_mems,
    relation_t const& dst_rel);

  void push_data(
    map<int, buffer_t> data,
    relation_t const& src_rel,
    partition_t const& dst_part,
    vector<memloc_t> const& dst_mems);

private:
  enum class cmd_t {
    execute,
    read_weight_chunk,
    read_embedding,
    get_data,
    push_data,
    open_reader,
    close_readers,
    alloc,
    shutdown
  };

  static vector<string> const& cmd_strs() {
    static vector<string> ret {
      "execute",
      "read_weight_chunk",
      "read_embedding",
      "get_data",
      "push_data",
      "open_reader",
      "close_reader",
      "alloc",
      "shutdown"
    };
    return ret;
  }

  friend std::ostream& operator<<(std::ostream& out, cmd_t const& c);
  friend std::istream& operator>>(std::istream& inn, cmd_t& c);

  cmd_t recv_cmd() {
    return parse_with_ss<cmd_t>(comm.recv_string(0));
  }

  void broadcast_cmd(cmd_t cmd) {
    comm.broadcast_string(write_with_ss(cmd));
  }

  void send_cmd(int dst, cmd_t cmd) {
    comm.send_string(dst, write_with_ss(cmd));
  }

private:
  communicator_t& comm;
  int this_rank;
  int world_size;

  buffer_t data;

  threadpool_t threadpool;

  int num_channels_per_move;

  cpu_kernel_executor_t kernel_executor;

  exec_state_t::priority_t priority_type;

  map<int, local_tensor_reader_t> readers;

  // only at rank 0 {{{
  dbuffer_t embedding_matrix;
  int num_files;
  model_args_t margs;
  // }}}

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
    auto parts = autopartition_for_bytes(
      graph,
      multiplier * world_size * num_threads_per,
      max_branching,
      space);
    delete gremlin_parts;

    gremlin_t gremlin_locate("locate");
    uint64_t flops_per_byte_moved = 100;
    auto ret = autolocate_agg_at_a_time_from_inns(
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

string events_to_wire(vector<event_t> const& events) {
  es_proto::InferenceEvents e;

  for(auto const& event: events) {
    event.to_proto(*e.add_event());
  }

  string ret;
  e.SerializeToString(&ret);
  return ret;
}

vector<event_t> events_from_wire(string const& str) {
  es_proto::InferenceEvents e;
  if(!e.ParseFromString(str)) {
    throw std::runtime_error("could not parse inference events!");
  }

  vector<event_t> ret;
  int n = e.event_size();
  for(int i = 0; i != n; ++i) {
    ret.push_back(event_t::from_proto(e.event(i)));
  }

  return ret;
}

void main_build(int argc, char** argv) {
  set_default_dtype(dtype_t::f32); // just in case

  args_t args(argc, argv);

  string model = args.get<string>("model");
  int world_size = args.get<int>("world_size");
  int num_threads = args.get<int>("num_threads"); // per machine

  uint64_t mem_size   = args.get<uint64_t>("mem_size");
  uint64_t batch_size = args.get<uint64_t>("batch_size");
  uint64_t seq_len    = args.get<uint64_t>("seq_len");

  uint64_t GB = 1000000000;
  mem_size *= GB;

  string filename = args.get<string>("filename");

  llama_autoplacer_t autoplacer(world_size, num_threads, args);

  int num_files;
  if(model == "7B") {
    num_files = 1;
  } else if(model == "13B") {
    num_files = 2;
  } else if(model == "30B") {
    num_files = 4;
  } else if(model == "65B") {
    num_files = 8;
  }

  event_t::init_t event_init {
    .world_size  = world_size,
    .mem_size    = mem_size,
    .num_threads = num_threads,
    .num_files   = num_files,
    .batch_size  = batch_size,
    .seq_len     = seq_len
  };

  model_args_t margs = event_init.make_model_args();

  builder_t builder = builder_t::make_first_token(margs, seq_len);

  recorder_t recorder(event_init);

  recorder.read_data_matrix(builder.embeddings,
    batch_size, seq_len, margs.n_heads, margs.head_dim());

  for(auto const& [name, gid]: builder.weights) {
    auto shape = builder.input_shape(gid);
    recorder.read_weight(gid, name, shape);
  }

  recorder.close_readers();

  if(builder.mask) {
    int const& gid = builder.mask.value();
    recorder.insert_mask(gid, seq_len);
  }

  {
    int const& gid = builder.freqs_cis;
    recorder.insert_freqs_cis(
      gid, margs.dim, margs.n_heads, margs.max_seq_len);
  }

  {
    vector<placement_t> pls = autoplacer(builder.graph);
    recorder.execute_graph(builder.graph, pls, "init");
  }

  {
    int scores_gid = builder.scores;
    builder = builder_t::make_next_token(builder);
    vector<placement_t> pls = autoplacer(builder.graph);

    recorder.build_next_relation(scores_gid, builder, pls);
  }

  std::ofstream out(filename);
  out << events_to_wire(recorder.get_events());
}

void main_execute(int argc, char** argv) {
  string usage = "addr_zero is_client world_size "
                 "num_channels num_channels_per_move "
                 "event_filename args";
  if(argc <= 4) {
    throw std::runtime_error(usage);
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);

  int num_channels = parse_with_ss<int>(argv[4]);
  int num_channels_per_move = parse_with_ss<int>(argv[5]);

  communicator_t communicator(addr_zero, is_rank_zero, world_size, num_channels);

  string events_filename = argv[6];

  args_t args(argc-6, argv+6);

  args.set_default("priority_type", "given");
  exec_state_t::priority_t priority_type =
    parse_priority_type(args.get<string>("priority_type"));

  executor_t executor(communicator, num_channels_per_move, priority_type);

  if(is_rank_zero) {
    es_proto::InferenceEvents es;
    std::fstream f(events_filename, std::ios::in | std::ios::binary);
    if(!es.ParseFromIstream(&f)) {
      throw std::runtime_error("could not parse from " + events_filename);
    }
    int n = es.event_size();
    for(int i = 0; i != n; ++i) {
      event_t event = event_t::from_proto(es.event(i));
      executor(event);
    }

    executor.shutdown();
  } else {
    executor.listen();
  }
}

int main(int argc, char** argv) {
  string usage = "usage: " + string(argv[0]) + " [execute|build]";
  if(argc <= 1) {
    throw std::runtime_error(usage);
  }
  if(string(argv[1]) == "execute") {
    main_execute(argc-1, argv+1);
  } else if(string(argv[1]) == "build") {
    main_build(argc-1, argv+1);
  } else {
    throw std::runtime_error(usage);
  }
}

void recorder_t::read_data_matrix(int gid,
  uint64_t bsz, uint64_t seq_len, uint64_t d0, uint64_t d1)
{
  auto data_locs = insert_relation(
    gid,
    relation_t::make_singleton(dtype_t::f32, {bsz, seq_len, d0, d1}, _max_tid+1));

  events.emplace_back(
    event_t::load_data_matrix_t { bsz, seq_len, d0*d1, data_locs[0].as_mem() });
}

void recorder_t::read_weight(int gid, string name, vector<uint64_t> const& shape)
{
  auto rel = tensor_reader_t::get_placement(name, shape, world_size, num_files);

  auto data_locs = insert_relation(gid, dtype_t::f32, rel);

  events.emplace_back(
    event_t::load_weight_t { name, data_locs });
}

void recorder_t::insert_mask(int gid, uint64_t seq_len)
{
  auto data_locs = insert_relation(
    gid,
    relation_t::make_singleton(dtype_t::f32, {seq_len, seq_len}, _max_tid+1));
  events.emplace_back(
    event_t::load_mask_t { seq_len, data_locs[0].as_mem() });
}

void recorder_t::insert_freqs_cis(
  int gid,
  uint64_t args_dim, uint64_t args_n_heads, uint64_t args_max_seq_len)
{
  uint64_t dim  = uint64_div(args_dim, args_n_heads);
  uint64_t hdim = uint64_div(dim, 2);
  uint64_t end  = 2*args_max_seq_len;

  auto data_locs = insert_relation(
    gid,
    relation_t::make_singleton(dtype_t::c64, {end, hdim}, _max_tid+1));
  events.emplace_back(event_t::load_freqs_cis_t {
    args_dim, args_n_heads, args_max_seq_len, data_locs[0].as_mem()
  });
}

void recorder_t::execute_graph(
  graph_t const& graph,
  vector<placement_t> const& placements,
  string message)
{
  // it's all about to get wonky, so just reset the allocators
  _allocators = std::nullopt;

  auto make_relation = [&](int gid, vtensor_t<int> const& tids) {
    return relation_t {
      .dtype = graph.out_dtype(gid),
      .placement = placements[gid],
      .tids = tids
    };
  };

  auto [inn_g_to_t, out_g_to_t, taskgraph] =
    taskgraph_t::make(graph, placements);

  remap_relations_t r;

  for(auto const& [gid, dst_tids]: inn_g_to_t) {
    relation_t const& src_rel = get_relation(gid);
    relation_t        dst_rel = make_relation(gid, dst_tids);

    r.insert(src_rel, dst_rel);
  }

  remap(r, message);

  execute_taskgraph(taskgraph, message);

  relations.clear();
  _max_tid = 0;
  for(auto const& [gid, tids]: out_g_to_t) {
    relations.insert({gid, make_relation(gid, tids)});
    _max_tid = std::max(
      *std::max_element(tids.get().begin(), tids.get().end()),
      _max_tid);
  }
}

void recorder_t::build_next_relation(
  int scores_gid,
  builder_t const& next_builder,
  vector<placement_t> const& next_pls)
{
  // Must copy scores_rel since it will be deleted
  // from relations in remap_gids
  relation_t scores_rel = relations.at(scores_gid);
  auto src_data_locs = get_data_locs(scores_rel.tids.get());

  remap_gids(next_builder.remap.value());

  int const& embedding_gid = next_builder.embeddings;
  auto const& embedding_pl = next_pls[embedding_gid];

  auto embedding_data_locs = insert_relation(
    embedding_gid, dtype_t::f32, embedding_pl);

  events.emplace_back(event_t::build_next_data_matrix_t {
    .src_partition = scores_rel.placement.partition,
    .src_data_locs = src_data_locs,
    .dst_partition = embedding_pl.partition,
    .dst_data_locs = embedding_data_locs
  });

  execute_graph(next_builder.graph, next_pls, "next");
}


void recorder_t::remap(remap_relations_t const& remap_relations, string message) {
  map<int, memstoloc_t> tensors_;
  for(auto const& [tid, memloc]: tensors) {
    tensors_.insert({tid, memstoloc_t(memloc)});
  }

  tensors.clear();

  auto [remap_gid, g] = create_remap_graph_constructor(remap_relations);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  // before: full_data_locs is with respect to the remap inn tids
  _update_map_with_new_tg_inns(
    tensors_, remap_gid, gid_to_inn, remap_relations, std::nullopt);
  // after: full_data_locs is with respect to the tasgkraph inns

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, {}, vector<uint64_t>(world_size, mem_size),
      tensors_, alloc_settings, false);

  tensors_.clear();

  events.emplace_back(event_t::execute_t {
    .message = message + ":remap",
    .memgraph = std::move(memgraph)
  });

  _update_map_with_new_tg_outs(
    out_tg_to_loc, remap_gid, gid_to_out, remap_relations, std::nullopt);

  for(auto const& [tid, memstoloc]: out_tg_to_loc) {
    tensors.insert({tid, memstoloc.get_memloc()});
  }
}

void recorder_t::execute_taskgraph(taskgraph_t const& taskgraph, string message) {
  map<int, memstoloc_t> tensors_;
  for(auto const& [tid, memloc]: tensors) {
    tensors_.insert({tid, memstoloc_t(memloc)});
  }

  tensors.clear();

  auto [inn_tg_to_loc, out_tg_to_loc, inputs_mg, core_mg] =
    memgraph_t::make_(
      taskgraph, {}, vector<uint64_t>(world_size, mem_size),
      tensors_, alloc_settings, false, true);

  tensors_.clear();

  events.emplace_back(event_t::execute_t {
    .message = message + ":inputs",
    .memgraph = inputs_mg.value()
  });
  events.emplace_back(event_t::execute_t {
    .message = message + ":core",
    .memgraph = std::move(core_mg)
  });

  for(auto const& [tid, memstoloc]: out_tg_to_loc) {
    tensors.insert({tid, memstoloc.get_memloc()});
  }
}

vector<memloc_t>
recorder_t::insert_relation(int gid, dtype_t dtype, placement_t const& pl)
{
  relation_t rel {
    .dtype = dtype,
    .placement = pl,
    .tids = vtensor_t<int>(pl.block_shape())
  };
  vector<int>& locs = rel.tids.get();
  std::iota(locs.begin(), locs.end(), _max_tid + 1);

  return insert_relation(gid, rel);
}

vector<memloc_t>
recorder_t::insert_relation(int gid, relation_t const& relation)
{
  auto const& tids = relation.tids.get();

  for(auto const& tid: tids) {
    _max_tid = std::max(tid, _max_tid);
  }

  auto [_, did_insert] = relations.insert({gid, relation});
  if(!did_insert) {
    throw std::runtime_error("this gid already has a relation!");
  }

  auto const& locs = relation.placement.locations.get();
  vector<uint64_t> nelems = relation.placement.partition.all_block_sizes().get();
  uint64_t dsz = dtype_size(relation.dtype);

  int nbid = tids.size();
  vector<memloc_t> ret;
  ret.reserve(nbid);
  for(int bid = 0; bid != nbid; ++bid) {
    int const& tid = tids[bid];
    int const& loc = locs[bid];
    uint64_t const& nelem = nelems[bid];
    uint64_t size = nelem * dsz;
    ret.push_back(insert_data(tid, loc, size));
  }

  return ret;
}

memloc_t
recorder_t::insert_data(int tid, int loc, uint64_t size)
{
  auto& allocator = get_allocator(loc);
  auto maybe = allocator.try_to_allocate_without_deps(size);

  if(maybe) {
    uint64_t const& offset = maybe.value();
    memloc_t memloc {
      .offset = offset,
      .size = size,
      .loc = loc
    };
    auto [_, did_insert] = tensors.insert({tid, memloc});
    if(!did_insert) {
      throw std::runtime_error("this tid already exists!");
    }
    return memloc;
  } else {
    throw std::runtime_error("ran out of memory in recorder insert_data");
  }
}

allocator_t& recorder_t::get_allocator(int loc)
{
  if(!_allocators) {
    _allocators = vector<allocator_t>(world_size, allocator_t(mem_size, alloc_settings));
    auto& allocs = _allocators.value();

    for(auto const& [_, memloc]: tensors) {
      auto const& [offset, size, loc] = memloc;
      allocs[loc].allocate_at_without_deps(offset, size);
    }
  }

  return _allocators.value()[loc];
}

void recorder_t::remap_gids(vector<tuple<int,int>> const& remap)
{
  map<int, relation_t> ret;

  for(auto const& [src,dst]: remap) {
    ret.insert({dst, relations.at(src)});
  }

  relations = ret;
}

vector<memloc_t>
recorder_t::get_data_locs(vector<int> const& tids)
{
  vector<memloc_t> ret;
  ret.reserve(tids.size());
  for(auto const& tid: tids) {
    ret.push_back(tensors.at(tid));
  }
  return ret;
}

void executor_t::operator()(event_t const& event) {
  if(this_rank != 0) {
    throw std::runtime_error("executing on non rank zero!");
  }

  if(event.is_load_weight()) {
    auto const& [name, memlocs] = event.get_load_weight();
    int num_files = memlocs.size();
    for(int chunk = 0; chunk != num_files; ++chunk) {
      auto const& memloc = memlocs[chunk];
      if(memloc.loc != (chunk % world_size)) {
        throw std::runtime_error("read weight: this loc is probably wrong");
      }
      read_weight_chunk_server(chunk, name, memloc);
    }
  } else if(event.is_load_data_matrix()) {
    auto const& [bsz, seq_len, d_embed, mem] = event.get_load_data_matrix();
    if(bsz != margs.batch_size) {
      throw std::runtime_error("bsz does not line up");
    }
    if(d_embed != margs.n_heads * margs.head_dim()) {
      throw std::runtime_error("invalid embed dim");
    }
    token_maker_t token_maker = make_token_maker_with_shape(bsz, seq_len);
    dbuffer_t embeddings = lookup_embeddings(
      margs.vocab_size,
      margs.dim,
      embedding_matrix,
      token_maker.get_tokens());
    if(embeddings.size() != mem.size) {
      throw std::runtime_error("invalid size in embeddings");
    }
    copy_into_data(embeddings.data, mem);
  } else if(event.is_load_mask()) {
    auto const& [seq_len, mem] = event.get_load_mask();
    buffer_t mask = transformer_t::form_start_mask(seq_len, dtype_t::f32).data;
    copy_into_data(mask, mem);
  } else if(event.is_load_freqs_cis()) {
    auto const& [dim, heads, max_seq_len, mem] = event.get_load_freqs_cis();
    buffer_t freqs_cis =
      transformer_t::form_full_freqs_cis(dim, heads, max_seq_len).data;
    copy_into_data(freqs_cis, mem);
  } else if(event.is_execute()) {
    auto const& [message, memgraph] = event.get_execute();
    broadcast_cmd(cmd_t::execute);
    comm.broadcast_string(message);
    comm.broadcast_string(memgraph.to_wire());
    execute(memgraph, message);
  } else if(event.is_init()) {
    auto const& e = event.get_init();
    if(e.world_size != world_size) {
      throw std::runtime_error("invalid world size provided by init");
    }
    if(e.num_threads > threadpool.num_runners()) {
      throw std::runtime_error("insufficient number of threads");
    }
    if(e.num_threads != threadpool.num_runners()) {
      DOUT("warning: num plan threads < num threadpool threads");
    }

    broadcast_cmd(cmd_t::alloc);
    comm.broadcast_contig_obj(e.mem_size);
    data = make_buffer(e.mem_size);

    margs = e.make_model_args();
    num_files = e.num_files;

    string base_filename = get_model_base_filename(num_files);
    for(int i = 0; i != num_files; ++i) {
      string si = write_with_ss(i);
      if(i < 10) {
        si = "0" + si;
      }
      string filename = base_filename + "_" + si;
      int loc = i % world_size;
      if(loc == 0) {
        readers.insert({i, local_tensor_reader_t(filename)});
      } else {
        send_cmd(loc, cmd_t::open_reader);
        comm.send_int(loc, i);
        comm.send_string(loc, filename);
      }
    }

    fill_embedding_matrix_server();
  } else if(event.is_close_readers()) {
    broadcast_cmd(cmd_t::close_readers);
    readers.clear();
  } else if(event.is_build_next_data_matrix()) {
    auto const& [src_part, src_data_locs, dst_part, dst_data_locs] =
      event.get_build_next_data_matrix();

    dbuffer_t scores = get_tensor_server(src_part, src_data_locs);

    uint64_t top_n = 1;
    vtensor_t<int> top_choices = get_top_choices(
      scores, margs.batch_size, margs.vocab_size, top_n);
    dbuffer_t embeddings = lookup_embeddings(
      margs.vocab_size,
      margs.dim,
      embedding_matrix,
      top_choices);

    push_tensor_server(embeddings, dst_part, dst_data_locs);
  } else {
    throw std::runtime_error("missing case: event_t");
  }
}

void executor_t::shutdown() {
  if(this_rank != 0) {
    throw std::runtime_error("must shutdown on rank 0!");
  }
  broadcast_cmd(cmd_t::shutdown);
}

void executor_t::listen() {
  if(this_rank == 0) {
    throw std::runtime_error("listening on rank zero!");
  }

  while(true) {
    cmd_t cmd = recv_cmd();
    if(cmd == cmd_t::execute) {
      string message = comm.recv_string(0);
      memgraph_t memgraph = memgraph_t::from_wire(comm.recv_string(0));
      execute(memgraph, message);
    } else if(cmd == cmd_t::read_weight_chunk) {
      int chunk = comm.recv_int(0);
      string name = comm.recv_string(0);
      mem_t mem = comm.recv_contig_obj<mem_t>(0);
      read_weight_chunk(chunk, name, mem);
    } else if(cmd == cmd_t::read_embedding) {
      remap_relations_t remap = remap_relations_t::from_wire(comm.recv_string(0));
      fill_embedding_matrix(remap);
    } else if(cmd == cmd_t::get_data) {
      partition_t src_part = partition_t::from_wire(comm.recv_string(0));
      vector<memloc_t> src_mems = comm.recv_vector<memloc_t>(0);
      relation_t dst_rel = relation_t::from_wire(comm.recv_string(0));
      get_data(src_part, src_mems, dst_rel);
    } else if(cmd == cmd_t::push_data) {
      relation_t src_rel = relation_t::from_wire(comm.recv_string(0));
      partition_t dst_part = partition_t::from_wire(comm.recv_string(0));
      vector<memloc_t> dst_mems = comm.recv_vector<memloc_t>(0);
      push_data(map<int, buffer_t>(), src_rel, dst_part, dst_mems);
    } else if(cmd == cmd_t::open_reader) {
      int chunk = comm.recv_int(0);
      string filename = comm.recv_string(0);
      readers.insert({chunk, local_tensor_reader_t(filename)});
    } else if(cmd == cmd_t::close_readers) {
      readers.clear();
    } else if(cmd == cmd_t::alloc) {
      uint64_t memsize = comm.recv_contig_obj<uint64_t>(0);
      data = make_buffer(memsize);
    } else if(cmd == cmd_t::shutdown) {
      return;
    } else {
      throw std::runtime_error("missing command!");
    }
  }
}

void executor_t::copy_into_data(buffer_t buffer, mem_t mem)
{
  if(mem.size != buffer->size) {
    throw std::runtime_error("invalid buffer given");
  }
  std::copy(
    buffer->data,
    buffer->data + buffer->size,
    data->data + mem.offset);
}

void executor_t::execute(memgraph_t const& memgraph, string message)
{
  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(
      memgraph,
      this_rank,
      kernel_executor,
      num_channels_per_move);

  rm_ptr_t rcm_ptr(new recv_channel_manager_t(comm));
  recv_channel_manager_t& rcm = *static_cast<recv_channel_manager_t*>(rcm_ptr.get());

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(data->raw())),
      rm_ptr_t(new notifier_t(comm, rcm)),
      rm_ptr_t(new send_channel_manager_t(comm)),
      rcm_ptr,
      rm_ptr_t(new threadpool_manager_t(threadpool))
    }
  ));

  exec_state_t state(graph, resource_manager, priority_type);

  if(this_rank == 0) {
    gremlin_t gremlin(message);
    state.event_loop();
  } else {
    state.event_loop();
  }
}

void executor_t::read_weight_chunk_server(
  int chunk,
  string name,
  memloc_t const& memloc)
{
  if(this_rank != 0) {
    throw std::runtime_error("this not the server: read weight");
  }

  if(memloc.loc == this_rank) {
    read_weight_chunk(chunk, name, memloc.as_mem());
  } else {
    int const& loc = memloc.loc;
    send_cmd(loc, cmd_t::read_weight_chunk);
    comm.send_int(loc, chunk);
    comm.send_string(loc, name);
    comm.send_contig_obj(loc, memloc.as_mem());
  }
}

void executor_t::read_weight_chunk(
  int chunk,
  string name,
  mem_t const& mem)
{
  dbuffer_t weight_f16 = dbuffer_t(
    dtype_t::f16,
    readers.at(chunk).read(name));
  dbuffer_t weight = dbuffer_t(
    dtype_t::f32,
    make_buffer_reference(data->data, mem.size));
  std::copy(
    weight_f16.f16(),
    weight_f16.f16() + weight_f16.nelem(),
    weight.f32());
}

void executor_t::fill_embedding_matrix_server() {
  remap_relations_t remap;

  {
    vector<uint64_t> shape{ margs.vocab_size, margs.dim };

    placement_t pl = tensor_reader_t::get_placement(
      "tok_embeddings.weight",
      shape,
      world_size,
      num_files);

    relation_t rel {
      .dtype = dtype_t::f32,
      .placement = pl,
      .tids = vtensor_t<int>(pl.block_shape())
    };
    rel.tids.get() = vector_iota<int>(pl.num_parts());

    remap.insert(
      rel,
      rel.as_singleton(99, 0));
  }

  broadcast_cmd(cmd_t::read_embedding);
  comm.broadcast_string(remap.to_wire());

  map<int, buffer_t> data_map = fill_embedding_matrix(remap);

  embedding_matrix = dbuffer_t(dtype_t::f32, data_map.at(99));
}

map<int, buffer_t>
executor_t::fill_embedding_matrix(remap_relations_t const& remap)
{
  map<int, buffer_t> ret;

  auto const& [src_rel, _] = remap.remap[0];
  int nbid = src_rel.placement.num_parts();
  for(int chunk = 0; chunk != nbid; ++chunk) {
    if(this_rank == (chunk % world_size)) {
      int const& tid = src_rel.tids.get()[chunk];
      ret.insert({
        tid,
        readers.at(chunk)("tok_embeddings.weight", dtype_t::f32)
      });
    }
  }

  repartition(comm, remap, ret);

  return ret;
}

dbuffer_t executor_t::get_tensor_server(
  partition_t const& src_part,
  vector<memloc_t> const& src_mems)
{
  partition_t dst_part = partition_t::singleton(src_part.total_shape());
  relation_t dst_rel {
    .dtype = dtype_t::f32,
    .placement = placement_t(dst_part),
    .tids = vtensor_t<int>(dst_part.block_shape(), 99)
  };

  broadcast_cmd(cmd_t::get_data);
  comm.broadcast_string(src_part.to_wire());
  comm.broadcast_vector(src_mems);
  comm.broadcast_string(dst_rel.to_wire());

  return dbuffer_t(
    dtype_t::f32,
    get_data(src_part, src_mems, dst_rel).at(99));
}

relation_t _build_relation(
  dtype_t dtype,
  partition_t const& part,
  vector<memloc_t> const& mems)
{
  int nbid = part.num_parts();
  if(mems.size() != nbid) {
    throw std::runtime_error("invalid src mems length");
  }

  relation_t rel {
    .dtype     = dtype,
    .placement = placement_t(part),
    .tids      = vtensor_t<int>(part.block_shape())
  };

  {
    vector<int>& locs = rel.placement.locations.get();
    for(int bid = 0; bid != nbid; ++bid) {
      auto const& memloc = mems[bid];
      int& loc = locs[bid];
      loc = memloc.loc;
    }
  }

  rel.tids.get() = vector_iota<int>(nbid);

  vector<uint64_t> nelems_per_block =
    rel.placement.partition.all_block_sizes().get();
  if(nelems_per_block.size() != mems.size()) {
    throw std::runtime_error("invalid num nelems_per_block");
  }

  uint64_t dsz = dtype_size(dtype);

  for(int bid = 0; bid != mems.size(); ++bid) {
    if(dsz * nelems_per_block[bid] != mems[bid].size) {
      throw std::runtime_error("incorrect sized memory");
    }
  }

  return rel;
}

map<int, buffer_t> executor_t::get_data(
  partition_t const& src_part,
  vector<memloc_t> const& src_mems,
  relation_t const& dst_rel)
{
  relation_t src_rel = _build_relation(dst_rel.dtype, src_part, src_mems);

  map<int, buffer_t> data_map;
  for(int tid = 0; tid != src_mems.size(); ++tid) {
    auto const& memloc = src_mems[tid];
    uint8_t* ptr = data->data + memloc.offset;
    data_map.insert({
      tid,
      make_buffer_reference(ptr, memloc.size)
    });
  }

  remap_relations_t remap;
  remap.insert(src_rel, dst_rel);

  repartition(comm, remap, data_map);

  return data_map;
}

void executor_t::push_tensor_server(
  dbuffer_t dbuffer,
  partition_t dst_part,
  vector<memloc_t> const& dst_mems)
{
  partition_t src_part = partition_t::singleton(dst_part.total_shape());
  relation_t src_rel {
    .dtype = dbuffer.dtype,
    .placement = placement_t(src_part),
    .tids = vtensor_t<int>(src_part.block_shape(), 99)
  };

  broadcast_cmd(cmd_t::push_data);
  comm.broadcast_string(src_rel.to_wire());
  comm.broadcast_string(dst_part.to_wire());
  comm.broadcast_vector(dst_mems);

  map<int, buffer_t> data_map;
  data_map.insert({99, dbuffer.data});

  push_data(data_map, src_rel, dst_part, dst_mems);
}

void executor_t::push_data(
  map<int, buffer_t> data_map,
  relation_t const& src_rel,
  partition_t const& dst_part,
  vector<memloc_t> const& dst_mems)
{
  relation_t dst_rel = _build_relation(src_rel.dtype, dst_part, dst_mems);

  remap_relations_t remap;
  remap.insert(src_rel, dst_rel);

  repartition(comm, remap, data_map);

  int nbid = dst_part.num_parts();
  for(int tid = 0; tid != nbid; ++tid) {
    auto const& memloc = dst_mems[tid];
    if(memloc.loc == this_rank) {
      buffer_t const& buffer = data_map.at(tid);
      copy_into_data(buffer, memloc.as_mem());
    }
  }
}

std::ostream& operator<<(std::ostream& out, executor_t::cmd_t const& c)
{
  auto const& items = executor_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, executor_t::cmd_t& c)
{
  c = executor_t::cmd_t(istream_expect_or(inn, executor_t::cmd_strs()));
  return inn;
}

string event_t::to_wire() const
{
  es_proto::InferenceEvent e;
  to_proto(e);
  string ret;
  e.SerializeToString(&ret);
  return ret;
}

void event_t::to_proto(es_proto::InferenceEvent& ret) const
{
  if(is_init()) {
    auto const& init = get_init();
    auto* i = ret.mutable_init();
    i->set_world_size(init.world_size);
    i->set_mem_size(init.mem_size);
    i->set_num_threads(init.num_threads);
    i->set_num_files(init.num_files);
    i->set_batch_size(init.batch_size);
    i->set_seq_len(init.seq_len);
  } else if(is_close_readers()) {
    auto* i = ret.mutable_close_readers();
    i->set_dummy(99);
  } else if(is_load_weight()) {
    auto const& load_weight = get_load_weight();
    auto* i = ret.mutable_load_weight();
    i->set_name(load_weight.name);
    for(auto const& mem: load_weight.data_locs) {
      mem.to_proto(*i->add_data_locs());
    }
  } else if(is_load_data_matrix()) {
    auto const& load_data_matrix = get_load_data_matrix();
    auto* i = ret.mutable_load_data_matrix();
    i->set_batch_size(load_data_matrix.batch_size);
    i->set_seq_len(load_data_matrix.seq_len);
    i->set_d_embed(load_data_matrix.d_embed);
    load_data_matrix.mem.to_proto(*i->mutable_mem());
  } else if(is_load_mask()) {
    auto const& load_mask = get_load_mask();
    auto* i = ret.mutable_load_mask();
    i->set_seq_len(load_mask.seq_len);
    load_mask.mem.to_proto(*i->mutable_mem());
  } else if(is_load_freqs_cis()) {
    auto const& load_freqs_cis = get_load_freqs_cis();
    auto* i = ret.mutable_load_freqs_cis();
    i->set_dim(load_freqs_cis.dim);
    i->set_heads(load_freqs_cis.heads);
    i->set_max_seq_len(load_freqs_cis.max_seq_len);
    load_freqs_cis.mem.to_proto(*i->mutable_mem());
  } else if(is_execute()) {
    auto const& execute = get_execute();
    auto* i = ret.mutable_execute();
    i->set_msg(execute.message);
    execute.memgraph.to_proto(*i->mutable_memgraph());
  } else if(is_build_next_data_matrix()) {
    auto const& build_next = get_build_next_data_matrix();
    auto* i = ret.mutable_build_next();
    build_next.src_partition.to_proto(*i->mutable_src_part());
    for(auto const& mem: build_next.src_data_locs) {
      mem.to_proto(*i->add_src_data_locs());
    }
    build_next.dst_partition.to_proto(*i->mutable_dst_part());
    for(auto const& mem: build_next.dst_data_locs) {
      mem.to_proto(*i->add_dst_data_locs());
    }
  } else {
    throw std::runtime_error("missing case: event to proto");
  }
}

event_t event_t::from_wire(string const& str)
{
  es_proto::InferenceEvent ie;
  if(!ie.ParseFromString(str)) {
    throw std::runtime_error("could not parse inference event");
  }
  return from_proto(ie);
}

event_t event_t::from_proto(es_proto::InferenceEvent const& ie)
{
  if(ie.has_init()) {
    auto const& i = ie.init();
    return event_t(init_t {
      .world_size  = i.world_size(),
      .mem_size    = i.mem_size(),
      .num_threads = i.num_threads(),
      .num_files   = i.num_files(),
      .batch_size  = i.batch_size(),
      .seq_len     = i.seq_len()
    });
  } else if(ie.has_close_readers()) {
    return event_t(close_readers_t{});
  } else if(ie.has_load_weight()) {
    auto const& l = ie.load_weight();
    vector<memloc_t> data_locs;
    int num = l.data_locs_size();
    data_locs.reserve(num);
    for(int i = 0; i != num; ++i) {
      data_locs.push_back(memloc_t::from_proto(l.data_locs(i)));
    }
    return event_t(load_weight_t {
      .name = l.name(),
      .data_locs = data_locs
    });
  } else if(ie.has_load_data_matrix()) {
    auto const& l = ie.load_data_matrix();
    return event_t(load_data_matrix_t {
      .batch_size = l.batch_size(),
      .seq_len = l.seq_len(),
      .d_embed = l.d_embed(),
      .mem = mem_t::from_proto(l.mem())
    });
  } else if(ie.has_load_mask()) {
    auto const& l = ie.load_mask();
    return event_t(load_mask_t {
      .seq_len = l.seq_len(),
      .mem = mem_t::from_proto(l.mem())
    });
  } else if(ie.has_load_freqs_cis()) {
    auto const& l = ie.load_freqs_cis();
    return event_t(load_freqs_cis_t {
      .dim = l.dim(),
      .heads = l.heads(),
      .max_seq_len = l.max_seq_len(),
      .mem = mem_t::from_proto(l.mem())
    });
  } else if(ie.has_execute()) {
    auto const& e = ie.execute();
    return event_t(execute_t {
      .message = e.msg(),
      .memgraph = memgraph_t::from_proto(e.memgraph())
    });
  } else if(ie.has_build_next()) {
    auto const& b = ie.build_next();

    vector<memloc_t> src_data_locs;
    {
      int num = b.src_data_locs_size();
      src_data_locs.reserve(num);
      for(int i = 0; i != num; ++i) {
        src_data_locs.push_back(memloc_t::from_proto(b.src_data_locs(i)));
      }
    }

    vector<memloc_t> dst_data_locs;
    {
      int num = b.dst_data_locs_size();
      dst_data_locs.reserve(num);
      for(int i = 0; i != num; ++i) {
        dst_data_locs.push_back(memloc_t::from_proto(b.dst_data_locs(i)));
      }
    }

    return event_t(build_next_data_matrix_t {
      .src_partition = partition_t::from_proto(b.src_part()),
      .src_data_locs = src_data_locs,
      .dst_partition = partition_t::from_proto(b.dst_part()),
      .dst_data_locs = dst_data_locs
    });
  } else {
    throw std::runtime_error("missing case");
  }
}


