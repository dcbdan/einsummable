#pragma once
#include "../src/base/setup.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/scalarop.h"

using tensor_t = graph_writer_t::tensor_t;

uint64_t uint64_div(uint64_t top, uint64_t bot, string err_msg = "");

struct full_dim_t {
  vector<uint64_t> dim_parts;

  uint64_t dim() const { return product(dim_parts); }

  static full_dim_t singleton(uint64_t d) {
    return full_dim_t { .dim_parts = { d } };
  }
};

struct full_shape_t {
  vector<full_dim_t> shape_parts;

  vector<uint64_t> full_shape() {
    return vector_flatten(
      vector_from_each_member(shape_parts, vector<uint64_t>, dim_parts)
    );
  }
  vector<uint64_t> shape() {
    return vector_from_each_method(shape_parts, uint64_t, dim);
  }
};

// Helpful structure for llama model
struct model_args_t {
  static model_args_t make_default();
  static model_args_t llama_7B();

  uint64_t dim;
  int n_layers;
  uint64_t n_heads;
  uint64_t multiple_of;
  double norm_eps;
  uint64_t max_batch_size;
  uint64_t max_seq_len;
  uint64_t vocab_size;
  int world_size;

  // TODO: remove local and world distinction
  uint64_t n_local_heads() const {
    return uint64_div(n_heads, world_size, "n_local_heads");
  }
  uint64_t head_dim() const {
    return uint64_div(dim, n_heads, "head_dim");
  }
  full_dim_t full_dim() const {
    return full_dim_t {
      .dim_parts = { n_local_heads(), head_dim() }
    };
  }
};

struct rms_norm_t {
  rms_norm_t() {}

  rms_norm_t(
    graph_writer_t* w,
    string name, //should be "ffn_norm." or "attention_norm"
    full_dim_t dim,
    float eps,
    dtype_t dtype = default_dtype());

  map<int, string> input_map() const;

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
    model_args_t args);

  tensor_t apply_rotary_embedding(tensor_t x, tensor_t freqs_cis);

  static tensor_t _apply_rotary_embedding(
    graph_writer_t& writer, tensor_t x, tensor_t freqs_cis);

  tensor_t forward(
    tensor_t x,
    tensor_t freqs_cis,
    optional<tensor_t> mask);

  tuple<tensor_t, tensor_t>
  get_keys_and_values(tensor_t k, tensor_t v);

  uint64_t start_pos() const {
    return prev_k ? prev_k.value().get_shape()[1] : 0;
  }

  map<int, string> input_map() const;

  graph_writer_t* writer;
  model_args_t args;
  string name;

  uint64_t n_local_heads;
  uint64_t head_dim;

  tensor_t wq;
  tensor_t wk;
  tensor_t wv;
  tensor_t wo;

  optional<tensor_t> prev_k;
  optional<tensor_t> prev_v;
};

struct feedforward_t {
  feedforward_t() {}

  feedforward_t(
    graph_writer_t* w,
    string name, //should pass in "feed_forward."
    full_dim_t dim,
    uint64_t hidden_dim);

  map<int, string> input_map() const;

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
    model_args_t args);

  tensor_t forward(
    tensor_t x,
    tensor_t freqs_cis,
    optional<tensor_t> mask);

  map<int, string> input_map() const;

  uint64_t start_pos() const {
    return attention.start_pos();
  }

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
    std::string name,
    model_args_t args);

  tensor_t forward(tensor_t x);

  map<int, string> input_map() const;

  uint64_t start_pos() const {
    return layers[0].start_pos();
  }

  // grab full_freqs_cis from [:, start_pos(): start_pos()+n]
  tensor_t next_freqs_cis(uint64_t n);
  tensor_t next_mask(uint64_t seqlen);

  graph_writer_t* writer;
  model_args_t args;

  // TODO: list freq_cis info
  // TODO: list mask info

  tensor_t full_freqs_cis;

  vector<transformer_block_t> layers;
  rms_norm_t norm;
  tensor_t w_vocab;

  struct mask_info_t {
    tensor_t mask;
    uint64_t start_pos;
    uint64_t seqlen;
  };
  vector<mask_info_t> mask_infos;
};

