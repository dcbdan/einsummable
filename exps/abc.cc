#include "../src/base/args.h"
#include "../src/server/cpu/tg_server.h"
#include "../src/autoplace/autoplace.h"
#include "../src/einsummable/gwriter.h"

struct random_inserter_t;

void main_rank_zero(server_base_t*     server,
                    random_inserter_t& random_inserter,
                    args_t&            pargs,
                    autoplace_config_t config);

struct random_inserter_t {
    random_inserter_t(server_base_t* server, communicator_t& comm)
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

        server->insert_gid_without_data(gid, rel);
    }

    void _local(relation_t const& rel, scalar_t lower, scalar_t upper)
    {
        int                this_rank = comm.get_this_rank();
        int                nbid = rel.placement.num_parts();
        vector<int> const& locs = rel.placement.locations.get();
        vector<int> const& tids = rel.tids.get();
        vector<uint64_t>   block_sizes = rel.placement.partition.all_block_sizes().get();

        map<int, tuple<int, buffer_t>> data;
        for (int bid = 0; bid != nbid; ++bid) {
            if (locs.at(bid) == this_rank) {
                dbuffer_t  d = make_dbuffer(rel.dtype, block_sizes.at(bid));
                int const& tid = tids.at(bid);
                data.insert({tid, {this_rank, d.data}});
            }
        }

        server->local_insert_tensors(data);
    }

    server_base_t*  server;
    communicator_t& comm;
    string          cmd;
};

int main(int argc, char** argv)
{
    set_default_dtype(dtype_t::f32);

    int expected_argc = 9;
    if (argc < expected_argc) {
        return 1;
    }

    string addr_zero = parse_with_ss<string>(argv[1]);
    bool   is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
    int    world_size = parse_with_ss<int>(argv[3]);

    DLINEOUT(addr_zero);
    DLINEOUT(is_rank_zero);
    DLINEOUT(world_size);
    communicator_t communicator(addr_zero, is_rank_zero, world_size);

    uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
    uint64_t GB = 1000000000;
    mem_size *= GB;

    int num_threads = parse_with_ss<int>(argv[5]);
    int num_contraction_threads = parse_with_ss<int>(argv[6]);

    int num_channels = parse_with_ss<int>(argv[7]);
    int num_channels_per_move = parse_with_ss<int>(argv[8]);

    int this_rank = communicator.get_this_rank();

    std::unique_ptr<server_base_t> server = std::unique_ptr<server_base_t>(
        new cpu_tg_server_t(communicator, mem_size, num_threads, num_contraction_threads));

    random_inserter_t random_inserter(server.get(), communicator);

    if (!is_rank_zero) {
        server->register_listen(random_inserter.cmd, [&] { random_inserter.client(); });

        server->listen();

        return 0;
    }

    args_t args(argc - (expected_argc - 1), argv + (expected_argc - 1));
    args.set_default("config_threads", 64);

    int num_config_threads_per_machine = args.get<int>("config_threads");
    DOUT("num config threads per machine " << num_config_threads_per_machine);
    autoplace_config_t config =
        autoplace_config_t::make_default01(world_size, num_config_threads_per_machine);

    main_rank_zero(server.get(), random_inserter, args, config);

    server->shutdown();

    return 0;
}

void main_rank_zero(server_base_t*     server,
                    random_inserter_t& random_inserter,
                    args_t&            args,
                    autoplace_config_t config)
{
    uint64_t ra = 1000 * args.get<uint64_t>("row_a");
    uint64_t ca = 1000 * args.get<uint64_t>("col_a");
    uint64_t rb = 1000 * args.get<uint64_t>("row_b");
    uint64_t cb = 1000 * args.get<uint64_t>("col_b");
    uint64_t rc = 1000 * args.get<uint64_t>("row_c");
    uint64_t cc = 1000 * args.get<uint64_t>("col_c");

    using tensor_t = graph_writer_t::tensor_t;

    graph_writer_t writer;

    tensor_t A = writer.input({ra, ca});
    tensor_t B = writer.input({rb, cb});
    tensor_t C = writer.input({rc, cc});

    string   which = args.get<string>("op");
    tensor_t X;
    if (which == "mul") {
        X = writer.matmul(A, B);
    } else if (which == "add") {
        X = writer.add(A, B);
    } else {
        throw std::runtime_error("should not reach");
    }

    tensor_t Y = writer.matmul(X, C);
    Y.save_inplace();

    auto const& graph = writer.get_graph();

    /////////////////////////

    vector<placement_t> pls = autoplace01(graph, config);

    scalar_t lower(default_dtype(), "-0.0000001");
    scalar_t upper(default_dtype(), "0.0000001");
    for (int const& gid : {A.get_id(), B.get_id(), C.get_id()}) {
        random_inserter(gid, pls.at(gid), lower, upper);
    }

    server->execute_graph(graph, pls);
}
