#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/autoplace/apart.h"

#include <fstream>

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

int main()
{
    uint64_t     batch_size = 8;
    model_args_t margs = model_args_t::llama(1, batch_size);
    margs.n_layers = 3;

    graph_writer_t writer;
    transformer_t  model(&writer, margs, 0);

    tensor_t embeddings = writer.input(full_shape_t({full_dim_t::singleton(margs.batch_size),
                                                     full_dim_t::singleton(margs.max_seq_len),
                                                     margs.full_dim()}));

    tensor_t predictions = model.forward(embeddings);
    predictions.save_inplace();

    auto const& graph = writer.get_graph();

    {
        std::ofstream f("g.gv");
        graph.print_graphviz(f);
        DOUT("printed g.gv");
    }
    vector<layer_ids_t> layer_ids;
    for (auto const& layer : model.layers) {
        auto const& ff = layer.feedforward;
        auto const& aa = layer.attention;
        layer_ids.push_back(layer_ids_t{.w1 = ff.w1.get_id(),
                                        .w2 = ff.w2.get_id(),
                                        .w3 = ff.w3.get_id(),
                                        .wq = aa.wq.get_id(),
                                        .wk = aa.wk.get_id(),
                                        .wv = aa.wv.get_id(),
                                        .wo = aa.wo.get_id(),
                                        .fn = layer.attention_norm.weight.get_id(),
                                        .an = layer.feedforward_norm.weight.get_id()});
    }

    // for(auto const& [name,tensor]: model.weight_map()) {
    //   DOUT(name << ": " << tensor.get_id());
    // }

    //////////

    // "data parallel"
    // vector<partition_t> parts;
    //{
    //  map<tuple<int, int>, partdim_t> pds;
    //  int id = embeddings.get_id();
    //  pds.insert({ {id,0}, partdim_t::split(batch_size, 8) });
    //  parts = apart03(graph, pds);
    //}

    // w1: hidden_dim, args.full_dim()
    // w2: args.full_dim(), hidden_dim
    // w3: hidden_dim, args.full_dim()
    //
    // wq: args.full_dim(), args.full_dim()
    // wk: args.full_dim(), args.full_dim()
    // wv: args.full_dim(), args.full_dim()
    // wo: args.full_dim(), args.full_dim()
    //
    // fn, an: args.full_dim()

    vector<partition_t> parts;
    {
        map<tuple<int, int>, partdim_t> pds;

        // partdim_t pd = partdim_t::split(margs.n_heads, 8);

        // pds.insert({ {embeddings.get_id(), 2}, pd });
        // pds.insert({ {model.norm.weight.get_id(), 0}, pd });
        // pds.insert({ {model.w_vocab.get_id(), 1}, pd });

        // for(auto const& [w1,w2,w3,wq,wk,wv,wo,fn,an]: layer_ids) {
        //   pds.insert({ {w1,1}, pd });
        //   pds.insert({ {w2,0}, pd });
        //   pds.insert({ {w3,1}, pd });

        //  pds.insert({ {wq,0}, pd });
        //  pds.insert({ {wk,0}, pd });
        //  pds.insert({ {wv,0}, pd });
        //  pds.insert({ {wo,0}, pd });

        //  pds.insert({ {wq,2}, pd });
        //  pds.insert({ {wk,2}, pd });
        //  pds.insert({ {wv,2}, pd });
        //  pds.insert({ {wo,2}, pd });

        //  pds.insert({ {fn,0}, pd });
        //  pds.insert({ {an,0}, pd });
        //}

        partdim_t pd = partdim_t::split(margs.max_seq_len, 8);
        pds.insert({{embeddings.get_id(), 1}, pd});
        pds.insert({{model.full_freqs_cis.get_id(), 0}, pd});
        pds.insert({{model.mask.value().get_id(), 0}, pd});
        pds.insert({{model.mask.value().get_id(), 1}, pd});
        DLINEOUT("model mask is " << model.mask.value().get_id());

        parts = apart03(graph, pds);
    }

    {
        std::ofstream f("g.gv");
        graph.print_graphviz(f, parts);
        DOUT("printed g.gv");
    }
}
