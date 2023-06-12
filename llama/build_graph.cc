#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"

using tensor_t = graph_writer_t::tensor_t;

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


struct RMSNorm_t {

  RMSNorm_t(graph_writer_t& w, 
  string name, //should be "ffn_norm." or "attention_norm"
  int dim,
  float eps = 1e-6)
  : writer(w), eps(eps){
    vector<uint64_t> weights_shape = {dim};
    weights = writer.input(weights_shape);
    input_names.insert(weights.get_id(), name + "weight");
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
    vector<uint64_t> kqv_initshape = {model_args.dim, n_local_heads, head_dim}; //outshape comes first because F.linear is xA^t, so shape is (out_features, in_features)
    vector<uint64_t> kqv_reshape = {model_args.dim, model_args.dim};

    wq = writer.input(kqv_initshape);
    input_names.insert(wq.get_id(), name + "wq.weight");
    wq = wq.view(kqv_reshape);

    wk = writer.input(kqv_initshape);
    input_names.insert(wk.get_id(), name + "wk.weight");
    wk = wk.view(kqv_reshape);

    wv = writer.input(kqv_initshape);
    input_names.insert(wv.get_id(), name + "wv.weight");
    wv = wv.view(kqv_reshape);
    //TODO: not sure about the size of wo.weight. wo starts at dim so n_head*headdim so no need view
    wo = writer.input(kqv_initshape);
    input_names.insert(wo.get_id(), name + "wo.weight");

  }

  /**
   * @brief Forward function of the attention block
   * 
   * @param x the input to attention block
   * @param start_pos 
   * @param freqs_cis 
   * @param mask 
   * @return tensor_t 
   */
  tensor_t forward(tensor_t x, int start_pos, tensor_t freqs_cis, optional<tensor_t> mask) {
    vector<uint64_t> input_shape = x.get_shape();
    uint64_t bsz = input_shape[0];
    uint64_t seqlen = input_shape[1];
    tensor_t xq = xq.transpose(0,1); //->transpose->x * xq^t
    tensor_t xk = xk.transpose(0,1);
    tensor_t xv = xv.transpose(0,1);
    xq = writer.matmul(x, xq);
    xq = writer.matmul(x, xk);
    xq = writer.matmul(x, xv);

    vector<uint64_t> initial_view_shape = {bsz, seqlen, n_local_heads, head_dim};
    xq = xq.view(initial_view_shape);
    xk = xk.view(initial_view_shape);
    xv = xv.view(initial_view_shape);


  }

  /* Get and return the mapping?*/
  map<int, string> input_map(){
    return input_names;
  };


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
    string name, //should pass in "feed_forward."
    vector<uint64_t> dim,
    uint64_t hidden_dim,
    uint64_t multiple_of)
    : writer(w)
  { 
    // silu = ...

    hidden_dim = 2 * hidden_dim / 3;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

    vector<uint64_t> w1w3shape = {hidden_dim, dim};
    vector<uint64_t> w2shape = {dim, hidden_dim};
    //TODO: still have to view? 
    w1 = writer.input(w1w3shape);
    input_names.insert(w1.get_id(), name + "w1.weight");

    w2 = writer.input(w2shape)
    input_names.insert(w2.get_id(), name + "w2.weight");

    w3 = writer.input(w1w3shape)
    input_names.insert(w3.get_id(), name + "w3.weight");
  }

  tensor_t forward(tensor_t x) {
    // return self.w2(F.silu(self.w1(x)) * self.w3(x))
  }

  map<int, string> input_map(){
    return input_names;
  };

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
    //TODO: loop the input_names mappping and insert to our own map for both att and ffn
    attention = attention_t(w, "attention.", args, world_size);
    feed_forward = feedforward_t(w, "feed_forward.", args.dim, 4*args.dim, args.multiple_of);
    //Attention_t names mapping
    map<int, string> att_inner_names = attention.input_map();
    for (auto x = att_inner_names.begin(); x != att_inner_names.end(); ++x) {
      int tensor_id = x->first;
      std::string name = x->second;
      input_names.insert(tensor_id, "layers." + std::to_string(layer_id) + "." + name;)
    }
    //feed_forward names mapping
    map<int, string> ff_inner_names = feed_forward.input_map();
    for (auto x = ff_inner_names.begin(); x != ff_inner_names.end(); ++x) {
      int tensor_id = x->first;
      std::string name = x->second;
      input_names.insert(tensor_id, "layers." + std::to_string(layer_id) + "." + name;)
    }

    attention_norm = RMSNorm_t(writer, "attention_norm.", args.dim, args.norms_eps);
    ffn_norm = RMSNorm_t(writer, "ffn_norm.", args.dim, args.norms_eps);
  }

  int forward(tensor_t x, uint64_t start_pos, tensor_t freqs_cis, tensor_t mask) {

  }

  map<int, string> input_map(){
    return input_names;
  };

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


struct transformer_t {
  attention_t(graph_writer_t& w, 
  std::string name, 
  ModelArgs_t params, 
  int world_size,
  tensor_t embedded_token): //size should be [tokensize, dim] 
    writer(w), params(params){
    n_layers = params.n_layers;

    for (int layer_id = 0; layer_id < n_layers; layer_id ++){
      //loop over input_map and insert into our own map.
      transformer_block_t block = transformer_block_t(writer, layer_id, params, world_size);
      layers.push_back(block);
      map<int, string> inner_names = block.input_map();
      for (auto x = inner_names.begin(); x != inner_names.end(); ++x) {
        int tensor_id = x->first;
        std::string name = x->second;
        input_names.insert(tensor_id, name;)
      } 
    }

    norm = RMSNorm_t(writer, "norm.weight", params.dim, params.norm_eps);
    
    //TODO: for the output weight from dim->vocab_size, we also do it in frontend?
    vector<uint64_t> output_shape = {vocab_size, dim};
    output_weight = writer.input(output_shape);
    input_names.insert(output_weight.get_id(), "output.weight");

    //TODO:freqciqs?

    /*Note for us: we still need to have code to convert the token embeddings
    back to vocab size on c++ side, but we don't have the */
  }

  /* Get and return the mapping?*/
  map<int, string> input_map(){
    return input_names;
  };


  graph_writer_t& writer;
  map<int,string> input_names;
  ModelArgs_t params; 
  int vocab_size;
  vector<transformer_block_t> layers;
  RMSNorm_t norm;
  tensor_t output_weight;
};
