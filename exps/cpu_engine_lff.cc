#include "../src/base/setup.h"
#include "../src/base/args.h"

#include "../src/server/cpu/server.h"

#include "../src/autoplace/autopart.h"

using tensor_t = graph_writer_t::tensor_t;
using full_dim_t = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

tuple<graph_t, vector<placement_t>> make_graph_ff(uint64_t batch,
                                                  int      nparts_batch,
                                                  uint64_t hidden,
                                                  int      nparts_hidden,
                                                  uint64_t dim,
                                                  int      nparts_dims);

double execute_direct_ff(uint64_t batch, uint64_t hidden, uint64_t dim);

uint64_t compute_hidden(uint64_t dim, uint64_t multiple_of);

partition_t make_p(vector<uint64_t> const& ds, vector<int> const& xs);

void rank_zero_main(cpu_mg_server_t& server, int argc, char** argv)
{
    args_t pargs(argc, argv);

    pargs.set_default<uint64_t>("batch", 1);
    pargs.set_default<uint64_t>("seqlen", 2048);
    pargs.set_default<uint64_t>("dim", 4096);
    pargs.set_default<uint64_t>("multiple_of", 256);
    pargs.set_default<int>("nrep", 10);
    pargs.set_default<bool>("direct", false);

    uint64_t batch = pargs.get<uint64_t>("batch");
    uint64_t seqlen = pargs.get<uint64_t>("seqlen");
    uint64_t dim = pargs.get<uint64_t>("dim");
    uint64_t multiple_of = pargs.get<uint64_t>("multiple_of");

    uint64_t hidden = compute_hidden(dim, multiple_of);

    int  nrep = pargs.get<int>("nrep");
    bool direct = pargs.get<bool>("direct");
    if (direct) {
        vector<double> ts;
        for (int i = 0; i != nrep; ++i) {
            DOUT(i + 1 << " / " << nrep);
            ts.push_back(execute_direct_ff(batch * seqlen, hidden, dim));
        }
        DOUT(ts);
    }

    pargs.set_default<int>("npart_batch", server.get_num_threads());
    pargs.set_default<int>("npart_hidden", 1);
    pargs.set_default<int>("npart_dims", 1);

    int npart_batch = pargs.get<int>("npart_batch");
    int npart_hidden = pargs.get<int>("npart_hidden");
    int npart_dims = pargs.get<int>("npart_dims");
    auto [graph, placements] =
        make_graph_ff(batch * seqlen, npart_batch, hidden, npart_hidden, dim, npart_dims);

    map<int, dbuffer_t> input_data;
    for (int gid = 0; gid != graph.nodes.size(); ++gid) {
        auto const& node = graph.nodes[gid];
        if (node.op.is_input()) {
            auto const& input = node.op.get_input();
            dbuffer_t   tensor = make_dbuffer(input.dtype, product(input.shape));
            tensor.random("-0.01", "0.01");
            input_data.insert({gid, tensor});
        }
    }

    for (int i = 0; i != nrep; ++i) {
        for (auto const& [gid, tensor] : input_data) {
            server.insert_tensor(gid, placements[gid], tensor);
        }
        server.execute_graph(graph, placements);
    }
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        throw std::runtime_error("provide addr_zero is_client world_size memsize");
    }

    string   addr_zero = parse_with_ss<string>(argv[1]);
    bool     is_rank_zero = parse_with_ss<int>(argv[2]) == 0;
    int      world_size = parse_with_ss<int>(argv[3]);
    uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);
    uint64_t GB = 1000000000;
    mem_size *= GB;

    int num_threads = std::max(1, int(std::thread::hardware_concurrency()));
    DOUT("number of threads in threadpool: " << num_threads)

    communicator_t communicator(addr_zero, is_rank_zero, world_size);

    cpu_mg_server_t server(communicator, mem_size, num_threads);

    if (is_rank_zero) {
        rank_zero_main(server, argc - 4, argv + 4);
    } else {
        server.listen();
    }
}

tuple<graph_t, vector<placement_t>> make_graph_ff(uint64_t batch,
                                                  int      nparts_batch,
                                                  uint64_t hidden,
                                                  int      nparts_hidden,
                                                  uint64_t dim,
                                                  int      nparts_dim)
{
    graph_writer_t writer;

    tensor_t x = writer.input({batch, dim});

    tensor_t w1 = writer.input({hidden, dim});
    tensor_t w2 = writer.input({dim, hidden});
    tensor_t w3 = writer.input({hidden, dim});

    tensor_t w1t = w1.transpose(0, 1);
    tensor_t w2t = w2.transpose(0, 1);
    tensor_t w3t = w3.transpose(0, 1);

    scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

    tensor_t a = writer.matmul(x, w1t);
    // tensor_t a = writer.ew(silu, writer.matmul(x, w1t));

    tensor_t b = writer.matmul(x, w3t);

    tensor_t c = writer.mul(a, b);

    tensor_t out = writer.matmul(c, w2t);
    out.save_inplace();

    auto      graph = writer.get_graph();
    partdim_t pd_batch = partdim_t::split(batch, nparts_batch);
    partdim_t pd_hidden = partdim_t::split(hidden, nparts_hidden);
    partdim_t pd_dim = partdim_t::split(dim, nparts_dim);
    auto      maybe_parts = autopart_from_inputs(
        graph,
        map<int, partition_t>{{x.get_id(), partition_t({pd_batch, pd_dim})},
                                   {w1.get_id(), partition_t({pd_hidden, pd_dim})},
                                   {w2.get_id(), partition_t({pd_dim, pd_hidden})},
                                   {w3.get_id(), partition_t({pd_hidden, pd_dim})}});
    if (!maybe_parts) {
        throw std::runtime_error("could not autopart graph from inptus");
    }
    vector<placement_t> pls;
    pls.reserve(graph.nodes.size());
    for (auto const& part : maybe_parts.value()) {
        pls.emplace_back(part);
    }

    return {graph, pls};
}

buffer_t make_out(vector<uint64_t> const& shape)
{
    dbuffer_t dbuffer = make_dbuffer(dtype_t::f32, product(shape));
    return dbuffer.data;
}

buffer_t make_data(vector<uint64_t> const& shape)
{
    buffer_t ret = make_out(shape);
    dbuffer_t(dtype_t::f32, ret).random("-0.00001", "0.00001");
    return ret;
}

double execute_direct_ff(uint64_t batch, uint64_t hidden, uint64_t dim)
{
    cpu_kernel_executor_t km;

    buffer_t x = make_data({batch, dim});
    buffer_t w1 = make_data({hidden, dim});
    buffer_t w2 = make_data({dim, hidden});
    buffer_t w3 = make_data({hidden, dim});

    einsummable_t e1 = einsummable_t::from_matmul_st(batch, dim, hidden);

    einsummable_t ee({batch, hidden}, {{0, 1}}, 2, scalarop_t::make_silu());

    einsummable_t e3 = einsummable_t::from_matmul_st(batch, dim, hidden);
    einsummable_t ea({batch, hidden}, {{0, 1}, {0, 1}}, 2, scalarop_t::make_add());

    einsummable_t e2 = einsummable_t::from_matmul_st(batch, hidden, dim);

    km.build(e1);
    km.build(ee);
    km.build(e3);
    km.build(ea);
    km.build(e2);

    auto start = clock_now();

    buffer_t z0 = make_out({batch, hidden});
    km(e1, z0->raw(), {x->raw(), w1->raw()});
    // km(ee, z0->raw(), {z0->raw()});

    buffer_t z1 = make_out({batch, hidden});
    km(e3, z1->raw(), {x->raw(), w3->raw()});

    km(ea, z1->raw(), {z0->raw(), z1->raw()});

    buffer_t z2 = make_out({batch, dim});
    km(e2, z2->raw(), {z1->raw(), w2->raw()});

    auto end = clock_now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

uint64_t compute_hidden(uint64_t dim, uint64_t multiple_of)
{
    uint64_t ret = 4 * dim;
    ret = uint64_t((2.0 * ret) / 3.0);
    ret = multiple_of * ((ret + multiple_of - 1) / multiple_of);
    return ret;
}

partition_t make_p(vector<uint64_t> const& ds, vector<int> const& xs)
{
    if (ds.size() != xs.size()) {
        throw std::runtime_error("make_p: incorect input sizes");
    }
    vector<partdim_t> pds;
    for (int i = 0; i != ds.size(); ++i) {
        uint64_t const& d = ds[i];
        int const&      x = xs[i];
        pds.push_back(partdim_t::split(d, x));
    }
    return partition_t(pds);
}
