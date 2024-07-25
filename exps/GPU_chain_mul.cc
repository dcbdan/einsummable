#include "../src/base/args.h"
#include "../src/server/gpu/server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"
#include <cstdint>

struct random_inserter_t;

void main_rank_zero(gpu_mg_server_t*   server,
                    random_inserter_t& random_inserter,
                    args_t&            pargs,
                    autoplace_config_t config);

struct random_inserter_t {
    random_inserter_t(gpu_mg_server_t* server, communicator_t& comm)
        : server(server), comm(comm), cmd("random_inserter")
    {
    }

    void client()
    {
        auto             rel = relation_t::from_wire(comm.recv_string(0));
        vector<scalar_t> vs = comm.recv_vector<scalar_t>(0);
        _local(rel, vs.at(0), vs.at(1));
    }

    void operator()(int gid, placement_t const& pl, scalar_t lower, scalar_t upper)
    {
        relation_t rel{
            .dtype = lower.dtype, .placement = pl, .tids = vtensor_t<int>(pl.block_shape())};
        {
            vector<int>& tids = rel.tids.get();
            int          next_tid = 1 + server->get_max_tid();
            std::iota(tids.begin(), tids.end(), next_tid);
        }

        int    world_size = comm.get_world_size();
        string registered_cmd = server->get_registered_cmd();
        for (int dst = 1; dst != world_size; ++dst) {
            comm.send_string(dst, registered_cmd);
            comm.send_string(dst, cmd);
            comm.send_string(dst, rel.to_wire());
            comm.send_vector(dst, vector<scalar_t>{lower, upper});
        }

        _local(rel, lower, upper);
        // DOUT("gid: " << gid);
        server->insert_gid_without_data(gid, rel);
    }

    void _local(relation_t const& rel, scalar_t lower, scalar_t upper)
    {
        // DLINE;
        int                this_rank = comm.get_this_rank();
        int                nbid = rel.placement.num_parts();
        vector<int> const& locs = rel.placement.locations.get();
        vector<int> const& tids = rel.tids.get();
        vector<uint64_t>   block_sizes = rel.placement.partition.all_block_sizes().get();

        map<int, tuple<int, buffer_t>> data;
        for (int bid = 0; bid != nbid; ++bid) {
            int const& loc = locs.at(bid);
            if (server->is_local_gpu(loc)) {
                dbuffer_t  d = make_dbuffer(rel.dtype, block_sizes.at(bid));
                int const& tid = tids.at(bid);
                data.insert({tid, {loc, d.data}});
            }
        }

        server->local_insert_tensors(data);
    }

    gpu_mg_server_t* server;
    communicator_t&  comm;
    string           cmd;
};

int main(int argc, char** argv)
{
    set_default_dtype(dtype_t::f32);

    int  world_size = 1;
    bool is_rank_zero = true;

    communicator_t communicator("0.0.0.0", is_rank_zero, world_size);
    int            this_rank = communicator.get_this_rank();

    uint64_t         mem_size = 16lu * 1000lu * 1000lu * 1000lu;
    vector<uint64_t> buffer_sizes;
    for (int i = 0; i < 4; ++i) {
        buffer_sizes.push_back(mem_size);
    }

    auto gpu_ptr = new gpu_mg_server_t(communicator, buffer_sizes);
    gpu_ptr->set_use_storage(false);
    gpu_ptr->set_split_off_inputs(false);

    std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(gpu_ptr);

    random_inserter_t random_inserter(gpu_ptr, communicator);

    if (!is_rank_zero) {
        server->register_listen(random_inserter.cmd, [&] { random_inserter.client(); });

        server->listen();

        return 0;
    }

    args_t args(argc, argv);
    args.set_default("config_threads", 1);
    int n_locs = 4;

    int num_config = args.get<int>("config_threads");
    DOUT("num compute per gpu " << num_config);
    autoplace_config_t config = autoplace_config_t::make_default01(n_locs, num_config);

    main_rank_zero(gpu_ptr, random_inserter, args, config);

    server->shutdown();

    return 0;
}

void main_rank_zero(gpu_mg_server_t*   server,
                    random_inserter_t& random_inserter,
                    args_t&            args,
                    autoplace_config_t config)
{
    DOUT("main rank zero enter");
    uint64_t S = 1000 * args.get<float16_t>("ds");
    bool     all_same = args.get<bool>("all_same");
    DOUT("S: " << S);
    DOUT("size of S matrix in bytes: " << S * S * sizeof(uint64_t));
    DOUT("size of S matrix in GB: " << S * S * sizeof(uint64_t) / 1e9);
    DOUT("all_same: " << std::boolalpha << all_same);
    vector<uint64_t> sA, sB, sC, sD, sE;
    if (all_same) {
        sA = vector<uint64_t>{S, S};
        sB = vector<uint64_t>{S, S};
        sC = vector<uint64_t>{S, S};
        sD = vector<uint64_t>{S, S};
        sE = vector<uint64_t>{S, S};

    } else {
        sA = vector<uint64_t>{S, S / 10};
        sB = vector<uint64_t>{S / 10, S};
        sC = vector<uint64_t>{S, S / 10};
        sD = vector<uint64_t>{S / 10, 10 * S};
        sE = vector<uint64_t>{10 * S, S};
    }

    using tensor_t = graph_writer_t::tensor_t;

    // DLINE;
    graph_writer_t writer;

    tensor_t A = writer.input(sA);
    tensor_t B = writer.input(sB);
    tensor_t C = writer.input(sC);
    tensor_t D = writer.input(sD);
    tensor_t E = writer.input(sE);

    tensor_t Y;
    {
        tensor_t S = writer.matmul(C, writer.matmul(D, E));
        tensor_t T = writer.matmul(A, B);
        Y = writer.add(S, T);
    }
    Y.save_inplace();
    // DLINE;

    auto const& graph = writer.get_graph();

    /////////////////////////

    vector<placement_t> pls;
    if (args.get<bool>("auto")) {
        // gremlin_t gremlin("building pls with auto..");
        DOUT("auto true");
        pls = autoplace01(graph, config);
    } else {
        // gremlin_t gremlin("building pls with heuristic..");
        DOUT("auto false");
        vector<partition_t> parts;
        int                 n = int(std::sqrt(double(config.n_compute())));
        DOUT("n compute: " << config.n_compute());
        DOUT("n is       " << n);
        if (config.n_compute() != n * n) {
            throw std::runtime_error("...");
        }
        DOUT("n is " << n);
        for (auto const& node : graph.nodes) {
            auto              shape = node.op.shape();
            vector<partdim_t> pds;
            for (uint64_t d : node.op.shape()) {
                pds.push_back(partdim_t::split(d, n));
            }
            parts.emplace_back(pds);
        }
        pls = alocate01(graph, parts, config.n_locs(), config.flops_per_byte_moved());
        // DLINE;
    }

    // for(auto const pl: pls) {
    //   DOUT(pl.partition);
    // }

    // DLINE;
    scalar_t lower(default_dtype(), "-0.0000001");
    scalar_t upper(default_dtype(), "0.0000001");
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_input()) {
            random_inserter(gid, pls.at(gid), lower, upper);
        }
    }
    // DLINE;
    server->execute_graph(graph, pls);
}