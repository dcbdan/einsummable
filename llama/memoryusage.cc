#include "../src/base/args.h"
#include "../src/einsummable/gwriter.h"

#include "modules.h"

#include <fstream>

int main(int argc, char** argv) {
  args_t pargs(argc, argv);

  dtype_t dtype = default_dtype();

  model_args_t margs = model_args_t::llama(1, 1);

  pargs.set_default<int>("max_n_layers", -1);
  {
    int n_layers = pargs.get<int>("max_n_layers");
    DLINEOUT("n_layers " << n_layers);
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  pargs.set_default<uint64_t>("sequence_length", 4096);
  margs.max_seq_len = pargs.get<uint64_t>("sequence_length");

  pargs.set_default<uint64_t>("batch_size", 1);
  margs.batch_size = pargs.get<uint64_t>("batch_size");

  graph_writer_t writer;
  transformer_t model(&writer, margs, 0, std::nullopt);

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(margs.batch_size),
    full_dim_t::singleton(margs.max_seq_len),
    margs.full_dim()
  }));

  // predictions: batch size, vocab size
  tensor_t predictions = model.forward(embeddings);
  tensor_t labels = writer.input(
    vector<uint64_t>{margs.batch_size, margs.vocab_size},
    dtype);

  // Compute the loss
  //   l{n} = log [ exp(v{n,y{n}})) / sum_c exp(v{n,c}) ]
  //   Loss = sum_n (l{n}) / N
  // Note, shift by c for numerical stability;
  //   where c{n} = max_c v{n,c}
  tensor_t loss;
  {
    tensor_t v = predictions;
    tensor_t c = writer.reduction("bv->b", castable_t::max, v);
    // v = v - c
    v = writer.ew("bv,b->bv", scalarop_t::make_sub(dtype), v, c);
    // ev = exp(v)
    tensor_t ev = writer.ew(scalarop_t::make_exp(dtype), v);
    // evsubset{b} = sum_v ev{b,v}*labels{b,v}
    tensor_t evsubset = writer.contraction("bv,bv->b", ev, labels);
    tensor_t evsum    = writer.reduction("bv->b", castable_t::add, ev);

    tensor_t lll = writer.ew(
      "b,b->b",
      scalarop_t::make_div(dtype),
      evsubset, evsum);

    lll = writer.ew(scalarop_t::make_log(dtype), lll);

    // (would like to use unsqueeze here but it is not implemented)

    double one_over_bsz = 1.0 / double(margs.batch_size);
    loss = lll.scale(scalar_t(dtype, write_with_ss(one_over_bsz)));
  }

  auto weight_map = model.weight_map();

  vector<tensor_t> weights;
  for(auto const& [name, tensor]: weight_map) {
    weights.push_back(tensor);
  }

  scalarop_t update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(dtype),
      scalarop_t::make_scale("learning_rate", dtype)
    }
  );

  auto grads = writer.backprop(loss, weights);
  vector<tensor_t> updated_weights;
  for(int i = 0; i != grads.size(); ++i) {
    tensor_t const& g = grads[i];
    tensor_t const& w = weights[i];
    tensor_t u = writer.straight_bew(update, w, g);
    u.save_inplace();
  }

  loss.save_inplace();

  graph_t graph = writer.get_graph();

  int num_keep = 0;
  map<int, string> colors;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    bool is_keep = true;
    if(node.op.is_einsummable() && !node.op.is_save()) {
      auto const& e = node.op.get_einsummable();
      if(!e.has_aggregation()) {
        is_keep = false;
      }
    }

    if(is_keep) {
      colors.insert({gid, "green"});
      num_keep++;
    } else {
      colors.insert({gid, "red"});
    }
  }
  DOUT("num keep " << num_keep << " / " << graph.nodes.size());
  {
    string filename = "g.gv";
    std::ofstream f(filename);
    graph.print_graphviz(f, colors);
    DOUT("printed " << filename);
  }

  graph = std::get<2>(graph_t::fuse(graph));
  {
    string filename = "ng.gv";
    std::ofstream f(filename);
    graph.print_graphviz(f);
    DOUT("printed " << filename);
  }

//  for(auto const& node: graph.nodes) {
//    uint64_t size = product(node.op.out_shape()) * dtype_size(node.op.out_dtype());
//    std::cout << size << " ";
//    std::cout << std::boolalpha << node.op.is_save() << " ";
//    for(auto const& inn: node.get_inns_set()) {
//      std::cout << inn << " ";
//    }
//
//    std::cout << std::endl;
//  }
}
