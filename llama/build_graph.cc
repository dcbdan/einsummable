#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"

using tensor_t = graph_writer_t::tensor_t;



struct RMSNorm_t {

  RMSNorm_t(graph_writer_t& w, 
  string name, //should be "ffn_norm." or "attention_norm"
  int dim,
  float eps = 1e-6)
  : writer(w), eps(eps){
    vector<uint64_t> weights_shape = {dim};
    weights = writer.input(weights_shape);
    input_names.insert(weights.get_id(), "weight");
  }
    
  map<int, string> input_names;
  graph_writer_t& writer;
  float eps;

  tensor_t weight;
  
}



struct attention_t {
  attention_t(graph_writer_t& w, 
  string name, //should be "attention."
  ModelArgs_t args, 
  int world_size)
  : writer(w), model_args(args){
    n_local_heads = args.n_head / world_size;
    head_dim = args.dim / args.n_heads;

    //input tensors
    vector<uint64_t> kqv_initshape = {n_local_heads, head_dim, model_args.dim};
    vector<uint64_t> kqv_reshape = {model_args.dim, model_args.dim};

    wq = writer.input(kqv_initshape);
    input_names.insert(wq.get_id(), "wq.weight");
    wq = wq.view(kqv_reshape);

    wk = writer.input(kqv_initshape);
    input_names.insert(wk.get_id(), "wk.weight");
    wk = wk.view(kqv_reshape);

    wv = writer.input(kqv_initshape);
    input_names.insert(wv.get_id(), "wv.weight");
    wv = wv.view(kqv_reshape);
    //TODO: not sure about the size of wo.weight. wo starts at dim so n_head*headdim so no need view
    wo = writer.input(kqv_initshape);
    input_names.insert(wo.get_id(), "wo.weight");

  }

  /* Get and return the mapping?*/
  map<int, string> input_map();


  map<int, string> input_names;
  graph_writer_t& writer;
  ModelArgs_t model_args; 
  int n_local_heads;
  int head_dim;

  /*Weights tensors*/
  tensor_t wq;
  tensor_t wk;
  tensor_t wv;
  tensor_t wo;
};

struct feedforward_t {
  feedforward_t(
    graph_writer_t& w,
    string name, //should pass in "feed_forward"
    vector<uint64_t> dim,
    uint64_t hidden_dim,
    uint64_t multiple_of)
    : writer(w)
  { 
    // add to the writer

    // silu = ...

    hidden_dim = 2 * hidden_dim / 3;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of)

    vector<uint64_t> w1w3shape = {hidden_dim, dim};
    w1 = writer.input(...)
    input_names.insert(w1.get_id(), ...);
    w1 = w1.view(...)

    w2 = writer.input(...)
    input_names.insert(w2.get_id(), ...);
    w2 = w2.view(...)

    w3 = writer.input(...)
    input_names.insert(w3.get_id(), ...);
    w3 = w3.view(...)
  }

  tensor_t forward(tensor_t x) {
    // return self.w2(F.silu(self.w1(x)) * self.w3(x))
  }

  map<int, string> input_map();

  graph_writer_t& writer;
  map<int, string> input_names;

  tensor_t w1;
  tensor_t w2;
  tensor_t w3;

  scalarop_t silu;
};


struct transformer_block_t {
  transformer_block_t(graph_writer_t& w, int layer_id, ModelArgs_t args, int world_size)
    : writer(w), layer_id(layer_id)
  {
    n_heads = args.n_heads;
    dim = args.dim;
    head_dim = args.dim / args.n_head;
    attention = attention_t(w, "attention.", args, world_size);
    feed_forward = feedforward_t(w, "feed_forward.", args.dim, 4*args.dim, args.multiple_of);



  }

  int forward(tensor_t x, uint64_t start_pos, tensor_t freqs_cis, tensor_t mask) {

  }

  map<int, string> input_map();

  int n_heads;
  int dim;
  int head_dim;
  attention_t attention;
  feedforward_t feed_forward;
  int layer_id;
  RMSNorm_t attention_norm;
  RMSNorm_t ffn_norm;

  graph_writer_t& writer;
  map<int, string> input_names;
};


// Helpful structure for llama model
struct ModelArgs_t {
  ModelArgs_t(int dim=512, 
  int n_layers = 8, 
  int n_head = 8, 
  int vocab_size = -1, 
  int multiple_of = 256, 
  float norms_eps = 1e-5,
  int max_batch_size = 32,
  int max_seq_len = 2048): 
  dim(dim), n_layers(n_layers), n_head(n_head), 
  vocab_size(vocab_size), multiple_of(multiple_of), norms_eps(norms_eps),
  max_batch_size(max_batch_size), max_seq_len(max_seq_len);

  int dim;
  int n_layers;
  int n_head;
  int vocab_size;
  int multiple_of;
  float norms_eps;
  int max_batch_size;
  int max_seq_len;

}



struct transformer_t {
  attention_t(graph_writer_t& w, std::string name, ModelArgs_t params, int world_size): 
    writer(w), params(params){
    vocab_size = params.vocab_size;
    n_layers = params.n_layers;


  }

  /* Get and return the mapping?*/
  map<int, string> input_map();


  graph_writer_t& writer;
  ModelArgs_t params; 
  int vocab_size;
};
