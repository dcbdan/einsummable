#include "modules.h"

model_args_t model_args_t::make_default() {
  return model_args_t {
    .dim             = 512,
    .n_layers        = 8,
    .n_heads         = 8,
    .multiple_of     = 256,
    .norm_eps       = 1e-5,
    .max_batch_size  = 32,
    .max_seq_len     = 2048
  };
}

rms_norm_t::rms_norm_t(
  graph_writer_t* w,
  string name,
  uint64_t dim,
  float eps)
    : writer(w), eps(eps), name(name)
{
  weight = writer->input({dim}, dtype_t::f16); // TODO: dtype as an arg?
}

map<int, string> rms_norm_t::input_map() const
{
  map<int, string> ret;
  ret.insert({weight.get_id(), name + "weight"});
  return ret;
}

#ifdef ASDASDADASSDASDASDASD

attention_t::attention_t(
  graph_writer_t& w,
  string name,
  model_args_t args,
  int world_size)
  : writer(w), model_args(args)
{
  if(args.n_head % world_size != 0) {
    throw std::runtime_error("args.n_head % worlside != 0");
  }
  if(args.dim % args.n_heads != 0) {
    throw std::runtime_error("args.dim % args.n_head != 0");
  }

  n_local_heads = args.n_head / world_size;
  head_dim = args.dim / args.n_heads;

  //input tensors

  //  outshape comes first because F.linear is xA^t,
  //  so shape is (out_features, in_features)
  vector<uint64_t> kqv_initshape = {model_args.dim, n_local_heads, head_dim};
  vector<uint64_t> kqv_reshape = {model_args.dim, model_args.dim};

  // TODO: put dtype elsewhere?

  wq = writer.input(kqv_initshape, dtype_t::f16);
  input_names.insert(wq.get_id(), name + "wq.weight");
  wq = wq.view(kqv_reshape);

  wk = writer.input(kqv_initshape,  dtype_t::f16);
  input_names.insert(wk.get_id(), name + "wk.weight");
  wk = wk.view(kqv_reshape);

  wv = writer.input(kqv_initshape, dtype_t::f16);
  input_names.insert(wv.get_id(), name + "wv.weight");
  wv = wv.view(kqv_reshape);

  //TODO: not sure about the size of wo.weight. wo starts at
  //      dim so n_head*headdim so no need view
  wo = writer.input(kqv_initshape, dtype_t::f16);
  input_names.insert(wo.get_id(), name + "wo.weight");
}

tensor_t attention_t::forward(
  tensor_t x,
  int start_pos,
  tensor_t freqs_cis,
  optional<tensor_t> mask)
{
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

feedforward_t::feedforward_t(
  graph_writer_t& w,
  string name,
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

  // TODO: feedforward weights have to be split up, then viewed
  //       unsplit.. how to do?

  //TODO: still have to view?
  w1 = writer.input(w1w3shape, dtype_t::f16);
  input_names.insert(w1.get_id(), name + "w1.weight");

  w2 = writer.input(w2shape, dtype_t::f16)
  input_names.insert(w2.get_id(), name + "w2.weight");

  w3 = writer.input(w1w3shape, dtype_t::f16)
  input_names.insert(w3.get_id(), name + "w3.weight");
}

tensor_t feedforward_t::forward(tensor_t x) {
  // return self.w2(F.silu(self.w1(x)) * self.w3(x))
}

transformer_block_t::transformer_block_t(
  graph_writer_t& w,
  int layer_id,
  model_args_t args,
  int world_size)
  : writer(w), layer_id(layer_id)
{
  n_heads = args.n_heads;
  dim = args.dim;
  head_dim = args.dim / args.n_head;
  //TODO: loop the input_names mappping and insert to our
  //      own map for both att and ffn
  attention = attention_t(w, "attention.", args, world_size);
  feed_forward = feedforward_t(
    w, "feed_forward.", args.dim, 4*args.dim, args.multiple_of);

  //Attention_t names mapping
  map<int, string> att_inner_names = attention.input_map();
  for (auto x = att_inner_names.begin(); x != att_inner_names.end(); ++x) {
    int tensor_id = x->first;
    std::string name = x->second;
    input_names.insert(
      tensor_id,
      "layers." + std::to_string(layer_id) + "." + name;)
  }
  //feed_forward names mapping
  map<int, string> ff_inner_names = feed_forward.input_map();
  for (auto x = ff_inner_names.begin(); x != ff_inner_names.end(); ++x) {
    int tensor_id = x->first;
    std::string name = x->second;
    input_names.insert(
      tensor_id,
      "layers." + std::to_string(layer_id) + "." + name;)
  }

  attention_norm = rms_norm_t(writer, "attention_norm.", args.dim, args.norms_eps);
  ffn_norm = rms_norm_t(writer, "ffn_norm.", args.dim, args.norms_eps);
}

int transformer_block_t::forward(
  tensor_t x,
  uint64_t start_pos,
  tensor_t freqs_cis,
  tensor_t mask)
{
  // TODO
}

#endif

transformer_t::transformer_t(
  graph_writer_t* w,
  std::string name,
  model_args_t params,
  uint64_t vocab_size,
  int world_size)
  : writer(w), params(params), vocab_size(vocab_size)
{
  int const& n_layers = params.n_layers;

  //Insert token_embedding weight into mapping
  //(add it to graph because we want to give it a tensor_id)
  //  vector<uint64_t> tok_embed_shape = {params.vocab_size, params.dim};
  //  tensor_t tok_embed_weight = writer.input(tok_embed_shape, dtype_t::f16);
  //  input_names.insert(tok_embed_weight.get_id(), "tok_embeddings.weight");

  //Insert freq_cis into freq_cis_map so we know size of freq_cis
  //when we create it during execution
  freq_cis_shape =
    {params.dim / params.n_heads, params.max_seq_len * 2};
  freq_cis = writer->input(freq_cis_shape, dtype_t::f16);

  for (int layer_id = 0; layer_id < n_layers; layer_id ++) {
    //loop over input_map and insert into our own map.
    layers.emplace_back(
      writer, layer_id, params, world_size);
  }

  norm = rms_norm_t(writer, "norm.weight", params.dim, params.norm_eps);

  vector<uint64_t> output_shape = {vocab_size, params.dim};
  output_weight = writer->input(output_shape);
}

tensor_t transformer_t::forward(
  tensor_t x, //size should be [tokensize, dim]
  int start_pos)
{
  // TODO
}

map<int, string> transformer_t::input_map() const {
  map<int, string> ret;
  for(transformer_block_t const& block: layers) {
    for(auto [tensor_id, name]: block.input_map()) {
      ret.insert({tensor_id, name});
    }
  }

  ret.insert({output_weight.get_id(), "output.weight"});

  return ret;
}

