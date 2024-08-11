#include "../src/autoplace/relationwise.h"
#include "../src/base/copyregion.h"

struct random_placement_t {
    placement_t operator()(vector<uint64_t> const& total_shape)
    {
        vector<partdim_t> partdims;
        for (uint64_t const& n : total_shape) {
            auto const& [beg_, end_] = part_size_rng;
            int p;
            if (end_ > n) {
                p = 1;
            } else {
                p = runif(beg_, end_);
            }
            partdims.push_back(partdim_t::split(n, p));
        }
        partition_t part(partdims);

        return placement_t::random(part, nloc);
    }

    tuple<int, int> part_size_rng;
    int             nloc;
};

void test01()
{
    int  nlocs = 3;
    int  n_threads_per = 1;
    auto kernel_coster = kernel_coster_t::for_cpu_cluster(nlocs);

    uint64_t ni = 10001;
    uint64_t nj = 10002;
    uint64_t nk = 10003;

    graph_constructor_t g;
    int lhs = g.insert_input(partition_t({partdim_t::split(ni, 3), partdim_t::split(nj, 4)}));
    int rhs = g.insert_input(partition_t({partdim_t::split(nj, 2), partdim_t::split(nk, 5)}));
    int join = g.insert_einsummable(
        partition_t({partdim_t::split(ni, 2), partdim_t::split(nk, 4), partdim_t::split(nj, 7)}),
        einsummable_t::from_matmul(ni, nj, nk),
        {lhs, rhs});
    int out =
        g.insert_formation(partition_t({partdim_t::split(ni, 3), partdim_t::split(nk, 2)}), join);

    graph_t const&      graph = g.graph;
    vector<placement_t> init_placements = g.get_placements();

    relationwise_t gwise(nlocs, n_threads_per, graph, kernel_coster, init_placements);

    double total = gwise.total_cost();

    int niter = 100000;

    // change all locs and back
    {
        double delta = 0.0;
        for (int iter = 0; iter != niter; ++iter) {
            int gid = runif(graph.nodes.size());
            int bid = runif(gwise.ginfos[gid].joins.size());
            int loc = runif(nlocs);
            delta += gwise(jid_t{gid, bid}, loc);
        }
        {
            gwise.reset_cost();
            double after_total = gwise.total_cost();
            DOUT(total + delta);
            DOUT(after_total);
        }
        for (int gid = 0; gid != graph.nodes.size(); ++gid) {
            for (int bid = 0; bid != gwise.ginfos[gid].joins.size(); ++bid) {
                delta += gwise(jid_t{gid, bid}, 0);
            }
        }
        {
            DOUT("and back");
            gwise.reset_cost();
            double after_total = gwise.total_cost();
            DOUT(total + delta);
            DOUT(after_total);
        }
        // Note that the delta is not going to be exactly correct because
        // all compute costs are approximate via threads_costs_t objects.
    }
}

void test02()
{
    int  nlocs = 8;
    int  n_threads_per = 1;
    auto kernel_coster = kernel_coster_t::for_cpu_cluster(nlocs);

    uint64_t ni = 10001;
    uint64_t nj = 10002;
    uint64_t nk = 10003;

    graph_constructor_t g;
    int lhs = g.insert_input(partition_t({partdim_t::split(ni, 3), partdim_t::split(nj, 4)}));
    int rhs = g.insert_input(partition_t({partdim_t::split(nj, 2), partdim_t::split(nk, 5)}));
    int join = g.insert_einsummable(
        partition_t({partdim_t::split(ni, 2), partdim_t::split(nk, 4), partdim_t::split(nj, 7)}),
        einsummable_t::from_matmul(ni, nj, nk),
        {lhs, rhs});
    int out =
        g.insert_formation(partition_t({partdim_t::split(ni, 3), partdim_t::split(nk, 2)}), join);

    graph_t const&      graph = g.graph;
    vector<placement_t> init_placements = g.get_placements();

    random_placement_t random_placement{.part_size_rng = {2, 8}, .nloc = nlocs};
    auto               make_random_placement = [&](int gid) {
        return random_placement(init_placements[gid].total_shape());
    };

    relationwise_t gwise(nlocs, n_threads_per, graph, kernel_coster, init_placements);

    auto total = gwise.total_cost();

    int niter = 10;

    // change all locs and back
    {
        double delta = 0.0;
        for (int iter = 0; iter != niter; ++iter) {
            int gid = runif(graph.nodes.size());
            delta += gwise(gid, make_random_placement(gid));
        }
        {
            gwise.reset_cost();
            double after_total = gwise.total_cost();
            DOUT(total + delta);
            DOUT(after_total);
        }
        for (int gid = 0; gid != graph.nodes.size(); ++gid) {
            delta += gwise(gid, init_placements[gid]);
        }
        {
            DOUT("and back");
            gwise.reset_cost();
            double after_total = gwise.total_cost();
            DOUT(total + delta);
            DOUT(after_total);
        }
    }
}

void test03()
{
    graph_writer_t writer;
    using id_t = graph_writer_t::tensor_t;

    uint64_t ni = 10000;
    uint64_t nj = 12000;
    uint64_t nk = 14000;

    id_t lhs = writer.input({ni, nj}, dtype_t::c64);
    id_t rhs = writer.input({nj, nk}, dtype_t::c64);
    id_t out = writer.matmul(lhs, rhs);

    out = out.to_real();
    out = out.to_complex();
    out = out.to_real();
    out = writer.add(out, out);
    out = out.to_complex();
    out = out.to_real();
    out = out.to_complex();
    out = writer.add(out, out);

    out = out.save();

    dtype_t dtype = dtype_t::f64;

    id_t a = writer.input({4000, 3000}, dtype);
    id_t b = writer.input({4000, 5000}, dtype);
    id_t c = writer.input({5000, 3000}, dtype);

    id_t x = writer.concat(1, {a, b});
    id_t y = writer.concat(0, {a, c});

    x = x.save();
    y = y.save();

    graph_t const& graph = writer.get_graph();

    int  niter = 10000;
    int  nlocs = 12;
    int  n_threads_per = 1;
    auto kernel_coster = kernel_coster_t::for_cpu_cluster(nlocs);

    random_placement_t random_placement{.part_size_rng = {2, 4}, .nloc = nlocs};
    auto               make_random_placement = [&](int gid) {
        return random_placement(graph.nodes[gid].op.shape());
    };

    relationwise_t gwise(
        nlocs, n_threads_per, graph, kernel_coster, graph.make_singleton_placement());
    double total = gwise.total_cost();
    double delta = 0.0;
    for (int iter = 0; iter != niter; ++iter) {
        int gid = runif(graph.nodes.size());
        delta += gwise(gid, make_random_placement(gid));
    }
    {
        gwise.reset_cost();
        double after_total = gwise.total_cost();
        DOUT(total + delta);
        DOUT(after_total);
    }
}

void test_copyregion_full()
{
    partition_t aa({partdim_t::split(100, 2), partdim_t::split(100, 3)});

    partition_t bb({partdim_t::split(100, 3), partdim_t::split(100, 3)});

    copyregion_full_t cr(aa, bb);

    do {
        DOUT(cr.idx_aa << " " << cr.idx_bb);
        DOUT(cr.index_aa << " " << cr.index_bb);
        DOUT(cr.offset_aa << " " << cr.offset_bb);
        DOUT("");
    } while (cr.increment());
}

void test_copyregion_join_inn()
{
    partition_t pjoin({partdim_t::split(100, 3), partdim_t::split(100, 1)});

    partition_t pinn({
        partdim_t::split(100, 5),
    });

    vector<int> inn = {1};

    copyregion_join_inn_t cr(pjoin, pinn, inn);
    do {
        DOUT(cr.idx_join() << " " << cr.idx_inn());
    } while (cr.increment());
}

int main()
{
    test01();
    DOUT("---");
    test02();
    DOUT("---");
    test03();
    // test_copyregion_full();
    // test_copyregion_join_inn();
}
