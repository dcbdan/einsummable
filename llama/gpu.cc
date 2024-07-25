#include "misc.h"
#include "modules.h"
#include "builder.h"
#include "reader.h"

#include "../src/base/args.h"

#include "../src/server/gpu/server.h"

#include "../src/autoplace/autoplace.h"

// #include <fstream>

struct token_maker_t {
    token_maker_t(vector<vector<int>> const ps) : prompts(ps)
    {
        int bsz = prompts.size();
        if (bsz == 0) {
            throw std::runtime_error("must give atleast one prompt");
        }

        int seqlen = prompts[0].size();
        for (auto const& prompt : prompts) {
            seqlen = std::min(std::size_t(seqlen), prompt.size());
        }
        if (seqlen == 0) {
            throw std::runtime_error("cannot have empty prompt");
        }

        tokens = vtensor_t<int>({bsz, seqlen});
        for (int i = 0; i != bsz; ++i) {
            for (int j = 0; j != seqlen; ++j) {
                tokens.at({i, j}) = prompts[i][j];
            }
        }
    }

    int batch_size() const
    {
        return tokens.get_shape()[0];
    }

    vtensor_t<int> const& get_tokens() const
    {
        return tokens;
    }

    vtensor_t<int> operator()(int start_pos, int end_pos) const
    {
        auto shape = tokens.get_shape();
        return tokens.subset({{0, shape[0]}, {start_pos, end_pos}});
    }

    vtensor_t<int> last_column() const
    {
        auto shape = tokens.get_shape();
        return this->operator()(shape[1] - 1, shape[1]);
    }

    void add_next_tokens(vtensor_t<int> const& next_tokens)
    {
        int startpos = tokens.get_shape()[0];

        tokens = vtensor_t<int>::concat(1, {tokens, next_tokens});

        // now apply the mask
        auto shape = tokens.get_shape();

        int bsz = shape[0];
        int seqlen = shape[1] - startpos;

        for (int i = 0; i != bsz; ++i) {
            for (int j = startpos; j < prompts[i].size() && j < shape[1]; ++j) {
                tokens.at({i, j}) = prompts[i][j];
            }
        }
    }

private:
    vector<vector<int>> prompts;
    vtensor_t<int>      tokens;
};

dbuffer_t lookup_embeddings(int nvocab, int nembed, dbuffer_t const& data, vtensor_t<int> tokens)
{
    if (data.nelem() != nvocab * nembed) {
        throw std::runtime_error("incorrectly sized data matrix");
    }

    int       ntokens = product(tokens.get_shape());
    dbuffer_t ret = make_dbuffer(data.dtype, nembed * ntokens);

    char const* data_raw = reinterpret_cast<char const*>(data.raw());
    char*       ret_raw = reinterpret_cast<char*>(ret.raw());

    uint64_t stride = dtype_size(data.dtype) * nembed;

    for (int i = 0; i != ntokens; ++i) {
        int const& token = tokens.get()[i];
        if (token < 0 || token >= nvocab) {
            throw std::runtime_error("invalid token");
        }
        std::copy(data_raw + token * stride, data_raw + (token + 1) * stride, ret_raw + i * stride);
    }

    return ret;
}

template <typename T>
vtensor_t<int> _get_top_choices(T const* data, uint64_t nrow, uint64_t ncol, uint64_t topn)
{
    topn = std::min(topn, ncol);
    vtensor_t<int> ret({int(nrow), int(topn)});
    int*           ret_vec = ret.get().data();
    for (int i = 0; i != nrow; ++i) {
        T const*         d = data + i * ncol;
        int*             r = ret_vec + i * topn;
        vector<T const*> tops = select_topn(d, d + ncol, topn);
        for (int j = 0; j != topn; ++j) {
            r[j] = std::distance(d, tops[j]);
        }
    }
    return ret;
}

vtensor_t<int> get_top_choices(dbuffer_t const& data, uint64_t nrow, uint64_t ncol, uint64_t topn)
{
    if (data.dtype == dtype_t::f16) {
        return _get_top_choices(data.f16(), nrow, ncol, topn);
    } else if (data.dtype == dtype_t::f32) {
        return _get_top_choices(data.f32(), nrow, ncol, topn);
    } else if (data.dtype == dtype_t::f64) {
        return _get_top_choices(data.f64(), nrow, ncol, topn);
    }
    throw std::runtime_error("get_top_choices: no dtype support here");
}

void _print_pl_info(string msg, graph_t const& graph, vector<placement_t> const& placements)
{
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);

    if (msg.size() < 45) {
        msg.resize(45, ' ');
    }

    int      num_input_msgs = 0;
    uint64_t num_input_bytes = 0;
    int      num_core_msgs = 0;
    uint64_t num_core_bytes = 0;
    set<int> inputs_everywhere = taskgraph.get_input_everywhere_ids();
    for (int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
        auto const& node = taskgraph.nodes[tid];
        if (node.op.is_move()) {
            uint64_t sz = node.op.get_move().size;
            if (inputs_everywhere.count(tid) > 0) {
                num_input_msgs++;
                num_input_bytes += sz;
            } else {
                num_core_msgs++;
                num_core_bytes += sz;
            }
        }
    }

    auto to_mb = [](uint64_t n) { return double(n) / 1e6; };
    DOUT("(" << msg << ") input " << num_input_msgs << "#, " << to_mb(num_input_bytes)
             << "MB | core " << num_core_msgs << "#, " << to_mb(num_core_bytes) << "MB ");
}

template <typename T>
void vector_repeat_resize(vector<T>& ret, int n)
{
    if (ret.size() > n) {
        ret.resize(n);
        return;
    }
    if (ret.size() == n) {
        return;
    }

    if (ret.size() == 0) {
        throw std::runtime_error("cannot repeat; nothing in there");
    }

    ret.reserve(n);
    int i = 0;
    while (ret.size() < n) {
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
vector<vector<int>> const _init_tokens{
    {1, 306, 4658, 278, 6593, 310, 2834, 338},
    {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871},
    {1, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 13},
    {1,    4103, 9632, 4223, 304,  5176, 29901, 13,  268, 7205, 4932, 357,
     1149, 301,  449,  276,  316,  2778, 13,    268, 715, 1878, 330,  3055,
     1725, 1149, 330,  3055, 1725, 4639, 28754, 13,  268, 923,  968,  1149}};

token_maker_t make_token_maker_with_shape(int nbatch, int nseq)
{
    vector<vector<int>> tokens = _init_tokens;
    for (auto& ts : tokens) {
        vector_repeat_resize(ts, nseq);
    }
    vector_repeat_resize(tokens, nbatch);

    return token_maker_t(tokens);
}

token_maker_t make_default_token_maker()
{
    return token_maker_t(_init_tokens);
}

// TODO: this is mostly just a copy of main_rank_zero_experiments
void main_rank_zero(gpu_mg_server_t& server, tensor_reader_t& reader, args_t& args)
{
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
    // DOUT("nbatch:                          " << nbatch);

    token_maker_t token_maker = make_token_maker_with_shape(nbatch, nseq);

    vtensor_t<int> init_tokens = token_maker.get_tokens();
    // DOUT(init_tokens.get());

    uint64_t bsz = init_tokens.get_shape()[0];
    uint64_t seqlen = init_tokens.get_shape()[1];

    {
        // Note: Assuming all is this being set?
        int seed = 99; // runif(10000);
        DOUT("Seed: " << seed);
        set_seed(seed);
    }

    string register_cmd = server.get_registered_cmd();

    model_args_t margs = model_args_t::llama(reader.num_files(), bsz);

    if (nseq > margs.max_seq_len) {
        throw std::runtime_error("The sequence length is too long for the model parameters.");
    }

    args.set_default<int>("max_n_layers", -1);
    {
        int n_layers = args.get<int>("max_n_layers");
        if (n_layers >= 0) {
            margs.n_layers = std::min(margs.n_layers, n_layers);
        }
    }

    builder_t builder = builder_t::make_first_token(margs, seqlen);

    auto start_reader = std::chrono::high_resolution_clock::now();

    dbuffer_t embedding_matrix;
    {
        map<int, relation_t>           relations;
        int                            current_tid = 0;
        map<int, tuple<int, buffer_t>> local_data;

        {
            vector<uint64_t> shape{margs.vocab_size, margs.dim};
            relation_t rel = reader(register_cmd, "tok_embeddings.weight", shape, current_tid);
            current_tid += rel.placement.num_parts();
            embedding_matrix = server.get_tensor(rel);
        }

        auto insert_reader_rel = [&relations, &current_tid](int gid, relation_t const& rel) {
            relations.insert({gid, rel});
            current_tid += rel.placement.num_parts();
        };
        auto insert_local_buffer = [&](int gid, buffer_t data) {
            local_data.insert({current_tid, {this_rank, data}});

            relation_t rel = relation_t::make_singleton(
                builder.input_dtype(gid), builder.input_shape(gid), current_tid);
            relations.insert({gid, rel});
            current_tid += 1;
        };

        for (auto const& [name, gid] : builder.weights) {
            auto       shape = builder.input_shape(gid);
            relation_t rel = reader(register_cmd, name, shape, current_tid);
            insert_reader_rel(gid, rel);
        }

        {
            int const& gid = builder.freqs_cis;
            buffer_t   freqs_cis = transformer_t::form_full_freqs_cis(margs).data;
            insert_local_buffer(gid, freqs_cis);
        }

        {
            int const& gid = builder.embeddings;
            dbuffer_t  embeddings =
                lookup_embeddings(margs.vocab_size, margs.dim, embedding_matrix, init_tokens);
            insert_local_buffer(gid, embeddings.data);
        }

        server.local_insert_tensors(local_data);

        // At this point we've called local_insert_tensors on the server directly
        // or via the reader, so tell the server how it maps to gids
        for (auto const& [gid, rel] : relations) {
            server.insert_gid_without_data(gid, rel);
        }
    }

    reader.shutdown(register_cmd);
    // time it
    auto end_reader = std::chrono::high_resolution_clock::now();
    DOUT("Reader shutdown. Time: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end_reader - start_reader).count()
         << "ms");

    {
        autoplace_config_t config =
            autoplace_config_t::make_default01(num_gpus, num_computes_per_loc);
        vector<placement_t> pls = autoplace01(builder.graph, config);
        server.execute_graph(builder.graph, pls);
    }

    {
        dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

        uint64_t       top_n = 1;
        vtensor_t<int> top_choices = get_top_choices(scores, bsz, margs.vocab_size, top_n);

        token_maker.add_next_tokens(top_choices.subset({{0, bsz}, {0, 1}}));
    }

    // args.set_default("niter", int(100));
    // int niter = args.get<int>("niter");
    // for(int i = 0; i != niter; ++i) {
    //   builder = builder_t::make_next_token(builder);
    //   server.remap_gids(builder.remap.value());

    //   vector<placement_t> pls = autoplacer(builder.graph);

    //   {
    //     dbuffer_t embeddings = lookup_embeddings(
    //       margs.vocab_size,
    //       margs.dim,
    //       embedding_matrix,
    //       token_maker.last_column());

    //     server.insert_tensor(builder.embeddings, pls[builder.embeddings], embeddings);
    //   }

    //   server.execute_graph(builder.graph, pls);

    //   {
    //     dbuffer_t scores = server.get_tensor_from_gid(builder.scores);

    //     uint64_t top_n = 1;
    //     vtensor_t<int> top_choices = get_top_choices(
    //       scores, bsz, margs.vocab_size, top_n);

    //     token_maker.add_next_tokens(top_choices.subset({ {0, bsz}, {0, 1} }));
    //   }
    // }

    // vtensor_t<int> const& tokens = token_maker.get_tokens();
    // int nrow = tokens.get_shape()[0];
    // for(int row = 0; row != nrow; ++row) {
    //   DOUT(tokens.index_subtensor(row).get());
    // }
}

// ./gpu_llama 7B 1 max_n_layers n
int main(int argc, char** argv)
{
    set_default_dtype(dtype_t::f16);

    if (argc < 3) {
        DOUT("argc " << argc);
        throw std::runtime_error(
            "required args: "
            "(1)base_data_file       (2)num_data_files");
    }

    string addr_zero = "0.0.0.0";
    bool   is_rank_zero = true;
    int    world_size = 1;

    string base_data_file(argv[1]);
    // add "../ " to the base_data_file
    base_data_file = "/home/zhimin/mytmpfs/" + base_data_file;
    int num_data_files = parse_with_ss<int>(argv[2]);

    if (is_rank_zero) {
        DOUT("world size:                      " << world_size);
        DOUT("base data file                   " << base_data_file);
        DOUT("num data files                   " << num_data_files);
    }

    communicator_t communicator(addr_zero, is_rank_zero, world_size);

    int this_rank = communicator.get_this_rank();

    args_t args(argc - 2, argv + 2);

    vector<uint64_t> buffer_sizes;
    // NOTE: 4 is hardcoded here since each anton has 4 gpus
    // 900GB storage: 14.5GB GPU buffer size
    for (int i = 0; i < 4; ++i) {
        buffer_sizes.push_back(120lu * 100lu * 1000lu * 1000lu);
    }

    gpu_mg_server_t server(communicator, buffer_sizes);

    auto reader_process = [&](map<int, buffer_t> const& data_) {
        map<int, tuple<int, buffer_t>> data;
        for (auto const& [tid, buffer] : data_) {
            data.insert({tid, {this_rank, buffer}});
        }
        server.local_insert_tensors(data);
    };

    tensor_reader_t reader(
        communicator, reader_process, this_rank, world_size, base_data_file, num_data_files);

    args.set_default("parallel_partialize", false);
    server.set_parallel_partialize(args.get<bool>("parallel_partialize"));

    args.set_default("use_storage", true);
    server.set_use_storage(args.get<bool>("use_storage"));

    args.set_default("split_off_inputs", true);
    server.set_split_off_inputs(args.get<bool>("split_off_inputs"));

    // DOUT("parallel_partialize:             " << server.parallel_partialize_);
    // DOUT("use_storage:                     " << server.use_storage_);
    // DOUT("split_off_inputs:                " << server.split_off_inputs_);

    if (is_rank_zero) {
        main_rank_zero(server, reader, args);

        server.shutdown();
    } else {
        server.register_listen(reader.read_cmd(), [&] { reader.listen_read(); });
        server.register_listen(reader.shutdown_cmd(), [&] { reader.listen_shutdown(); });

        server.listen();
    }
}
