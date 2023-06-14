#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"

using tensor_t = graph_writer_t::tensor_t;

// TODO: put this in a main
set_default_dtype(dtype::f16);

// Helpful structure for llama model
struct model_args_t {
  static model_args_t make_default();

  int dim;
  int n_layers;
  int n_head;
  int vocab_size;
  int multiple_of;
  float norms_eps;
  int max_batch_size;
  int max_seq_len;
};

struct rms_norm_t {
  rms_norm_t(
    graph_writer_t& w,
    string name, //should be "ffn_norm." or "attention_norm"
    int dim,
    float eps = 1e-6);

  map<int, string> input_names;
  graph_writer_t& writer;
  float eps;

  tensor_t weight;
};

struct attention_t {
  attention_t(
    graph_writer_t& w,
    string name, //should be "attention."
    model_args_t args,
    int world_size);

  /**
   * @brief Forward function of the attention block
   *
   * @param x the input to attention block
   * @param start_pos
   * @param freqs_cis
   * @param mask
   * @return tensor_t
   */
  tensor_t forward(
    tensor_t x,
    int start_pos,
    tensor_t freqs_cis,
    optional<tensor_t> mask);

  /* Get and return the mapping?*/
  map<int, string> input_map(){
    return input_names;
  };

  map<int, string> input_names;
  graph_writer_t& writer;
  model_args_t model_args;
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
    uint64_t multiple_of);

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
  transformer_block_t(
    graph_writer_t& w,
    int layer_id,
    model_args_t args,
    int world_size);

  int forward(tensor_t x, uint64_t start_pos, tensor_t freqs_cis, tensor_t mask);

  map<int, string> input_map(){
    return input_names;
  };

  int n_heads;
  int dim;
  int head_dim;
  attention_t attention;
  feedforward_t feed_forward;
  int layer_id;
  rms_norm_t attention_norm;
  rms_norm_t ffn_norm;

  graph_writer_t& writer;
  map<int, string> input_names;
};

struct transformer_t {
  transformer_t(
    graph_writer_t& w,
    std::string name,
    model_args_t params,
    int world_size);

  tensor_t forward(
    tensor_t x, //size should be [tokensize, dim]
    int start_pos);

  /* Get and return the mapping?*/
  map<int, string> input_map(){
    return input_names;
  };

  graph_writer_t& writer;
  map<int,string> input_names;
  /*freq_cis_map has key as the tensor id for freqcis, and vector<dim, end> as the value*/
  map<int, vector<uint64_t>> freq_cis_map;
  model_args_t params;
  int vocab_size;
  vector<transformer_block_t> layers;
  rms_norm_t norm;
  tensor_t output_weight;
  tensor_t tok_embed_weight;
};
