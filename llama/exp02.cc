#include "modules.h"
#include "builder.h"

#include "../src/base/args.h"

#include "../src/autoplace/apart.h"
#include "../src/autoplace/alocate.h"

#include "../src/einsummable/memgraph.h"
#include "../src/base/timetracker.h"

parts_space_t parse_parts_space(string const& space_)
{
    if (space_ == "contraction") {
        return parts_space_t::contraction;
    } else if (space_ == "all") {
        return parts_space_t::all;
    } else if (space_ == "all_range") {
        return parts_space_t::all_range;
    } else {
        throw std::runtime_error("invalid space_");
    }
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

    vector<placement_t> operator()(graph_t const& graph) const
    {
        int multiplier = double_workers ? 2 : 1;

        gremlin_t* gremlin_parts = new gremlin_t("parts");
        auto       parts =
            apart01(graph, multiplier * world_size * num_threads_per, max_branching, space);
        delete gremlin_parts;

        gremlin_t gremlin_locate("locate");
        uint64_t  flops_per_byte_moved = 100;
        auto      ret = alocate01(graph, parts, world_size, flops_per_byte_moved);
        return ret;
    }

    int           world_size;
    int           num_threads_per;
    parts_space_t space;
    bool          double_workers;
    int           max_branching;
    uint64_t      flops_per_byte_moved;
};

int main(int argc, char** argv)
{
    args_t args(argc, argv);

    args.set_default("model", "7B");
    int nfiles;
    if (args.get<string>("model") == "7B") {
        nfiles = 1;
    } else if (args.get<string>("model") == "13B") {
        nfiles = 2;
    } else if (args.get<string>("model") == "30B") {
        nfiles = 4;
    } else if (args.get<string>("model") == "65B") {
        nfiles = 8;
    }

    args.set_default("batch_size", uint64_t(1));
    args.set_default("seq_len", uint64_t(512));

    uint64_t bsz = args.get<uint64_t>("batch_size");
    uint64_t seqlen = args.get<uint64_t>("seq_len");

    model_args_t margs = model_args_t::llama(nfiles, bsz);
    margs.max_seq_len = seqlen + 2;

    args.set_default<int>("max_n_layers", -1);
    {
        int n_layers = args.get<int>("max_n_layers");
        if (n_layers >= 0) {
            margs.n_layers = std::min(margs.n_layers, n_layers);
        }
    }

    builder_t builder = builder_t::make_first_token(margs, seqlen);
    args.set_default("is_next", false);
    if (args.get<bool>("is_next")) {
        builder = builder_t::make_next_token(builder);
    }

    ///

    args.set_default("world_size", int(2));
    args.set_default("num_threads_per", int(32));

    int world_size = args.get<int>("world_size");
    int num_threads_per = args.get<int>("num_threads_per");

    llama_autoplacer_t autoplacer(world_size, num_threads_per, args);

    ///

    vector<placement_t> pls = autoplacer(builder.graph);

    taskgraph_t taskgraph;
    {
        gremlin_t gremlin("tg make");
        taskgraph = std::get<2>(taskgraph_t::make(builder.graph, pls));
    }
    DOUT("taskgraph num nodes: " << taskgraph.nodes.size());

    {
        args.set_default("mem_size", uint64_t(100));
        uint64_t mem_size = args.get<uint64_t>("mem_size");
        uint64_t GB = 1000000000;
        mem_size *= GB;

        vector<int>      which_storage = vector_iota<int>(world_size);
        vector<uint64_t> mem_sizes(world_size, mem_size);

        gremlin_t gremlin("mg make...");
        auto [_0, _1, maybe_i_mg, c_mg] =
            memgraph_t::make_(taskgraph,
                              which_storage,
                              mem_sizes,
                              {},
                              allocator_settings_t::default_settings(),
                              false,
                              true);
        DOUT("init num nodes: " << maybe_i_mg.value().nodes.size());
        DOUT("core num nodes: " << c_mg.nodes.size());

        uint64_t n_edges = 0;
        for (auto const& node : c_mg.nodes) {
            n_edges += node.inns.size();
        }
        DOUT("core num edges: " << n_edges);
    }

    auto& timetracker = get_timetracker();
    timetracker.print_totals(std::cout);
}
