#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/dbuffer.h"

using tensor_t     = graph_writer_t::tensor_t;
using full_dim_t   = graph_writer_t::full_dim_t;
using full_shape_t = graph_writer_t::full_shape_t;

uint64_t uint64_div(uint64_t top, uint64_t bot, string err_msg = "");

// Helpful structure for llama model
struct model_args_t {
  static model_args_t llama(int n, uint64_t batch_size = 1);
  static model_args_t llama_7B(    uint64_t batch_size = 1);
  static model_args_t llama_13B(   uint64_t batch_size = 1);
  static model_args_t llama_30B(   uint64_t batch_size = 1);
  static model_args_t llama_65B(   uint64_t batch_size = 1);

  uint64_t dim;
  int n_layers;
  uint64_t n_heads;
  uint64_t multiple_of;
  double norm_eps;
  uint64_t batch_size;
  uint64_t max_seq_len;
  uint64_t vocab_size;

  uint64_t head_dim() const {
    return uint64_div(dim, n_heads, "head_dim");
  }
  full_dim_t full_dim() const {
    return full_dim_t({ n_heads, head_dim() });
  }
};

struct rms_norm_t {
  rms_norm_t() {}

  rms_norm_t(
    graph_writer_t* w,
    string name, //should be "ffn_norm." or "attention_norm."
    full_dim_t dim,
    float eps,
    dtype_t dtype = default_dtype());

  map<string, tensor_t> weight_map() const;

  tensor_t norm(tensor_t x);

  tensor_t forward(tensor_t x);

  graph_writer_t* writer;
  float eps;
  string name;
  dtype_t dtype;
  tensor_t weight;
};

tensor_t apply_rotary_embedding(
  graph_writer_t& writer, tensor_t x, tensor_t freqs_cis);

struct attention_t {
  attention_t() {}

  attention_t(
    graph_writer_t* w,
    string name, //should be "attention."
    model_args_t args,
    uint64_t start_pos);

  tensor_t apply_rotary_embedding(tensor_t x, tensor_t freqs_cis);

  static tensor_t _apply_rotary_embedding(
    graph_writer_t& writer, tensor_t x, tensor_t freqs_cis);

  tensor_t forward(
    tensor_t x,
    tensor_t freqs_cis,
    optional<tensor_t> mask);

  void set_next_keys_and_values(tensor_t k, tensor_t v);

  map<string, tensor_t> weight_map() const;

  tuple<tensor_t, tensor_t>
  get_prev_kv() const { return prev_kv.value(); }

  tuple<tensor_t, tensor_t>
  get_new_kv() const { return next_kv.value(); }

  graph_writer_t* writer;
  model_args_t args;
  string name;

  uint64_t batch_size;
  uint64_t n_heads;
  uint64_t head_dim;

  tensor_t wq;
  tensor_t wk;
  tensor_t wv;
  tensor_t wo;

  optional<tuple<tensor_t, tensor_t>> prev_kv;

  // This gets set after in the forward pass
  optional<tuple<tensor_t, tensor_t>> next_kv;
};

struct feedforward_t {
  feedforward_t() {}

  feedforward_t(
    graph_writer_t* w,
    string name, //should pass in "feed_forward."
    full_dim_t dim,
    uint64_t hidden_dim);

  map<string, tensor_t> weight_map() const;

  tensor_t forward(tensor_t x);

  graph_writer_t* writer;

  string name;

  tensor_t w1;
  tensor_t w2;
  tensor_t w3;
};

struct transformer_block_t {
  transformer_block_t(
    graph_writer_t* w,
    int layer_id,
    model_args_t args,
    uint64_t start_pos);

  tensor_t forward(
    tensor_t x,
    tensor_t freqs_cis,
    optional<tensor_t> mask);

  map<string, tensor_t> weight_map() const;

  tuple<tensor_t, tensor_t>
  get_prev_kv() const { return attention.get_prev_kv(); }

  tuple<tensor_t, tensor_t>
  get_new_kv() const { return attention.get_new_kv(); }

  graph_writer_t* writer;
  model_args_t args;
  int layer_id;

  attention_t attention;
  feedforward_t feedforward;
  rms_norm_t attention_norm;
  rms_norm_t feedforward_norm;
};

struct transformer_t {
  transformer_t(
    graph_writer_t* w,
    model_args_t args,
    uint64_t start_pos);

  tensor_t forward(tensor_t x);

  map<string, tensor_t> weight_map() const;

  static dbuffer_t form_full_freqs_cis(model_args_t const& args);
  static dbuffer_t form_full_freqs_cis(
    uint64_t args_dim, uint64_t args_n_heads, uint64_t args_max_seq_len);
  static dbuffer_t form_freqs_cis(uint64_t dim, uint64_t end);
  static dbuffer_t form_start_mask(uint64_t seqlen, dtype_t dtype = default_dtype());

  // grab full_freqs_cis from [start_pos: start_pos+seqlen]
  tensor_t get_freqs_cis(uint64_t seqlen);

  vector<tuple<tensor_t, tensor_t>> get_prev_kvs() const;
  vector<tuple<tensor_t, tensor_t>> get_new_kvs() const;

  graph_writer_t* writer;
  model_args_t args;
  uint64_t expected_batch_size;
  uint64_t start_pos;

  tensor_t full_freqs_cis;
  optional<tensor_t> mask;

  vector<transformer_block_t> layers;
  rms_norm_t norm;
  tensor_t w_vocab;
};


