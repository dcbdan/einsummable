#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"

using tensor_t = graph_writer_t::tensor_t;

uint64_t uint_div(uint64_t top, uint64_t bot, string err_msg = "");

struct full_dim_t {
  vector<uint64_t> dim_parts;

  uint64_t dim() const { return product(dim_parts); }
};

struct full_shape_t {
  vector<full_dim_t> shape;
};

// Helpful structure for llama model
struct model_args_t {
  static model_args_t make_default();

  uint64_t dim;
  int n_layers;
  uint64_t n_heads;
  uint64_t multiple_of;
  double norm_eps;
  uint64_t max_batch_size;
  uint64_t max_seq_len;
};

struct rms_norm_t {
  rms_norm_t() {}

  rms_norm_t(
    graph_writer_t* w,
    string name, //should be "ffn_norm." or "attention_norm"
    full_dim_t dim,
    float eps);

  map<int, string> input_map() const;

  tensor_t norm(tensor_t x);

  tensor_t forward(tensor_t x);

  graph_writer_t* writer;
  float eps;

  string name;
  tensor_t weight;
};

struct attention_t {
  attention_t() {}

  attention_t(
    graph_writer_t* w,
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
  map<int, string> input_map() const;

  graph_writer_t* writer;
  model_args_t model_args;
  string name;

  int n_local_heads;
  int head_dim;

  /*Weights tensors*/
  tensor_t wq;
  tensor_t wk;
  tensor_t wv;
  tensor_t wo;
};

struct feedforward_t {
  feedforward_t() {}

  feedforward_t(
    graph_writer_t* w,
    string name, //should pass in "feed_forward."
    uint64_t dim,
    uint64_t hidden_dim,
    uint64_t multiple_of);

  map<int, string> input_map() const;

  tensor_t forward(tensor_t x);

  graph_writer_t* writer;

  string name;

  tensor_t w1;
  tensor_t w2;
  tensor_t w3;

  scalarop_t silu;
};

struct transformer_block_t {
  transformer_block_t(
    graph_writer_t* w,
    int layer_id,
    model_args_t args,
    int world_size);

  int forward(
    tensor_t x,
    uint64_t start_pos,
    tensor_t freqs_cis,
    tensor_t mask);

  map<int, string> input_map() const;

  graph_writer_t* writer;

  attention_t attention;
  feedforward_t feed_forward;

  int n_heads;
  int dim;
  int head_dim;

  int layer_id;

  rms_norm_t attention_norm;
  rms_norm_t ffn_norm;
};

struct transformer_t {
  transformer_t(
    graph_writer_t* w,
    std::string name,
    model_args_t params,
    uint64_t vocab_size,
    int world_size);

  tensor_t forward(
    tensor_t x, //size should be [tokensize, dim]
    int start_pos);

  map<int, string> input_map() const;

  graph_writer_t* writer;
  model_args_t params;
  int vocab_size;

  tensor_t freq_cis;
  vector<uint64_t> freq_cis_shape;

  vector<transformer_block_t> layers;
  rms_norm_t norm;
  tensor_t output_weight;
};

