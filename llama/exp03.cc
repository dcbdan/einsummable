#include "modules.h"
#include "builder.h"

#include "../src/base/args.h"

#include <fstream>

int main(int argc, char** argv) {
  args_t args(argc, argv);

  args.set_default("model", "7B");
  int nfiles;
  if(args.get<string>("model") == "7B") {
    nfiles = 1;
  } else if(args.get<string>("model") == "13B") {
    nfiles = 2;
  } else if(args.get<string>("model") == "30B") {
    nfiles = 4;
  } else if(args.get<string>("model") == "65B") {
    nfiles = 8;
  }

  args.set_default("batch_size", uint64_t(1));
  args.set_default("seq_len", uint64_t(512));

  uint64_t bsz    = args.get<uint64_t>("batch_size");
  uint64_t seqlen = args.get<uint64_t>("seq_len");

  model_args_t margs = model_args_t::llama(nfiles, bsz);
  margs.max_seq_len = seqlen + 2;

  args.set_default<int>("max_n_layers", -1);
  {
    int n_layers = args.get<int>("max_n_layers");
    if(n_layers >= 0) {
      margs.n_layers = std::min(margs.n_layers, n_layers);
    }
  }

  builder_t builder = builder_t::make_first_token(margs, seqlen);

  std::unordered_set<einsummable_t> einsummables;
  for(auto const& node: builder.graph.nodes) {
    if(node.op.is_einsummable()) {
      auto e = node.op.get_einsummable();
      for(uint64_t& d: e.join_shape) {
        d = 1;
      }
      einsummables.insert(e);
    }
  }
  int num = einsummables.size();
  int num_unary = 0;
  int num_binary = 0;
  for(auto const& e: einsummables) {
    if(e.inns.size() == 1) {
      num_unary++;
    }
    if(e.inns.size() == 2) {
      num_binary++;
    }
  }
  DOUT("total, unary, binary: " << num << " " << num_unary << " " << num_binary);

  vector<int> weight_ids;
  for(auto const& [name, weight_id]: builder.weights) {
    weight_ids.push_back(weight_id);
  }
  int const& scores = builder.scores;

  graph_t& graph = builder.graph;

  int num_nodes_before = graph.nodes.size();

  {
    string filename = "llama_exp03_before.gv";
    std::ofstream f(filename);
    graph.print_graphviz(f);
    DOUT("printed " << filename);
  }

  graph.backprop(scores, weight_ids);
  int num_nodes_after = graph.nodes.size();

  DOUT("num nodes: from " << num_nodes_before << " to " << num_nodes_after);

  map<int, string> colors;
  for(int i = num_nodes_before; i != num_nodes_after; ++i) {
    auto const& node = graph.nodes[i];
    if(node.op.is_einsummable() && node.op.get_einsummable().is_contraction()) {
      colors.insert({i, "darkorchid1"});
    } else {
      colors.insert({i, "azure"});
    }
  }

  {
    string filename = "llama_exp03_after.gv";
    std::ofstream f(filename);
    graph.print_graphviz(f, colors);
    DOUT("printed " << filename);
  }
}

