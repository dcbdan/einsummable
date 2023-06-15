#include "modules.h"

// TODO: wherever division occurs, make sure no modulo;
//       add a helper method to throw runtime error

uint64_t uint64_div(uint64_t top, uint64_t bot, string err_msg)
{
  if(top % bot != 0) {
    err_msg = "uint64_div: has remainder. " + err_msg;
    throw std::runtime_error(err_msg);
  } else {
    return top / bot;
  }
}

model_args_t model_args_t::make_default() {
  return model_args_t {
    .dim             = 512,
    .n_layers        = 8,
    .n_heads         = 8,
    .multiple_of     = 256,
    .norm_eps        = 1e-5,
    .max_batch_size  = 32,
    .max_seq_len     = 2048,
    .world_size      = 1
  };
}

rms_norm_t::rms_norm_t(
  graph_writer_t* w,
  string name,
  full_dim_t dim,
  float eps,
  dtype_t dtype)
    : writer(w), eps(eps), name(name), dtype(dtype)
{
  weight = writer->input(dim.dim_parts, dtype);
  weight = weight.view({ dim.dim() });
}

map<int, string> rms_norm_t::input_map() const
{
  map<int, string> ret;
  ret.insert({weight.get_id(), name + "weight"});
  return ret;
}

tensor_t rms_norm_t::norm(tensor_t x) {
  if(x.get_dtype() != dtype_t::f32) {
    throw std::runtime_error("invalid dtype in rms norm :: norm");
  }

  auto x_shape = x.get_shape();
  int out_rank = x_shape.size();
  if(out_rank <= 1) {
    throw std::runtime_error("rms_norm: not a big enough output rank");
  }

  scalarop_t inverse_sqrt = scalarop_t::make_inverse_sqrt(dtype_t::f32);
  scalarop_t square       = scalarop_t::make_square(dtype_t::f32);
  scalarop_t mul          = scalarop_t::make_mul(dtype_t::f32);

  scalar_t _e(eps);
  scalar_t _a(1/float(1.0*x_shape.back()));
  scalarop_t scale_then_add_eps = scalarop_t::combine(
    scalarop_t::make_add(dtype_t::f32),
    {
      scalarop_t::make_scale(_a),
      scalarop_t::make_constant(_e)
    });

  string ijk(out_rank, ' ');
  std::iota(ijk.begin(), ijk.end(), 'a');

  string ij(out_rank-1, ' ');
  std::iota(ij.begin(), ij.end(), 'a');

  string ijk_to_ij     = ijk + "->" + ij;
  string ijk_ij_to_ijk = ijk + ","  + ij + "->" + ijk;

  // z = x * np.power(np.mean(np.square(x), axis=-1, keepdims=True) + eps, -0.5);

  // y = np.mean(np.square(x), axis=-1) + eps
  tensor_t y;
  y = writer->ew(square,                            x);
  y = writer->reduction(ijk_to_ij, castable_t::add, y);
  y = writer->ew(scale_then_add_eps,                y);
  y = writer->ew(inverse_sqrt,                      y);

  // x * y
  return writer->ew(ijk_ij_to_ijk, mul, x, y);
}

tensor_t rms_norm_t::forward(tensor_t x) {
  if(dtype != x.get_dtype()) {
    throw std::runtime_error("invalid input dtype rms norm t forward");
  }
  tensor_t output = norm(x.to_dtype(dtype_t::f32)).to_dtype(dtype);

  int out_rank = x.get_shape().size();

  string ijk(out_rank, ' ');
  std::iota(ijk.begin(), ijk.end(), 'a');
  string k(1, char('a' + (out_rank-1)));
  string str = ijk + "," + k + "->" + ijk;

  scalarop_t mul = scalarop_t::make_mul(dtype);

  return writer->ew(str, mul, output, weight);
}

attention_t::attention_t(
  graph_writer_t* w,
  string name,
  model_args_t args)
  : writer(w), model_args(args), name(name),
    n_local_heads(args.n_local_heads()),
    head_dim(args.head_dim())
{
  //input tensors

  //  outshape comes first because F.linear is xA^t,
  //  so shape is (out_features, in_features)

  vector<uint64_t> kqv_initshape = { n_local_heads, head_dim, n_local_heads, head_dim };
  vector<uint64_t> kqv_reshape = {model_args.dim, model_args.dim};

  wq = writer->input(kqv_initshape);
  wq = wq.view(kqv_reshape);

  wk = writer->input(kqv_initshape);
  wk = wk.view(kqv_reshape);

  wv = writer->input(kqv_initshape);
  wv = wv.view(kqv_reshape);

  wo = writer->input(kqv_initshape);
  wo.view(kqv_reshape);
}

tensor_t attention_t::apply_rotary_embedding(
  tensor_t x, tensor_t freqs_cis)
{
  return _apply_rotary_embedding(*writer, x, freqs_cis);
}

tensor_t attention_t::_apply_rotary_embedding(
  graph_writer_t& writer, tensor_t x, tensor_t freqs_cis)
{
  if(x.get_dtype() != dtype_t::f32) {
    throw std::runtime_error("rot emb needs f32 x");
  }
  if(freqs_cis.get_dtype() != dtype_t::c64) {
    throw std::runtime_error("rot emb needs c64 freqs_cis");
  }
  x = x.to_complex();
  x = writer.ew(
    "abcd,bd->abcd",
    scalarop_t::make_mul(dtype_t::c64),
    x, freqs_cis);
  return x.to_real();
}

map<int, string> attention_t::input_map() const {
  map<int, string> input_names;

  input_names.insert({wq.get_id(), name + "wq.weight"});
  input_names.insert({wk.get_id(), name + "wk.weight"});
  input_names.insert({wv.get_id(), name + "wv.weight"});
  input_names.insert({wo.get_id(), name + "wo.weight"});

  return input_names;
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

  if(input_shape.size() != 3 || input_shape[2] != model_args.dim) {
    throw std::runtime_error("invalid shape x to attention forward");
  }

  tensor_t xq = writer->matmul(x, wq.transpose(0,1));
  tensor_t xk = writer->matmul(x, wk.transpose(0,1));
  tensor_t xv = writer->matmul(x, wv.transpose(0,1));

  vector<uint64_t> full_xshape = {
    bsz, seqlen, n_local_heads, head_dim
  };

  xq = xq.view(full_xshape);
  xk = xk.view(full_xshape);
  xv = xv.view(full_xshape);

  // TODO: apply rotary embedding to xq,xk

  // TODO: concat something cache
}

feedforward_t::feedforward_t(
  graph_writer_t* w,
  string name,
  full_dim_t dim,
  uint64_t hidden_dim)
  : writer(w), name(name)
{
  full_shape_t to_hidden {
    .shape_parts = { full_dim_t::singleton(hidden_dim), dim }
  };
  full_shape_t to_dim {
    .shape_parts = { dim, full_dim_t::singleton(hidden_dim) }
  };

  w1 = writer->input(to_hidden.full_shape()).view(to_hidden.shape());
  w2 = writer->input(to_dim.full_shape()   ).view(to_dim.shape()   );
  w3 = writer->input(to_hidden.full_shape()).view(to_hidden.shape());
}

map<int, string> feedforward_t::input_map() const {
  map<int, string> input_names;

  input_names.insert({w1.get_id(), name + "w1.weight"});
  input_names.insert({w2.get_id(), name + "w2.weight"});
  input_names.insert({w3.get_id(), name + "w3.weight"});

  return input_names;
}

tensor_t feedforward_t::forward(tensor_t x) {
  tensor_t w1t = w1.transpose(0,1);
  tensor_t w2t = w2.transpose(0,1);
  tensor_t w3t = w3.transpose(0,1);

  scalarop_t silu = scalarop_t::make_silu(x.get_dtype());

  // return self.w2(F.silu(self.w1(x)) * self.w3(x))
  //                ------------------   ----------
  //                a                    b
  //                -------------------------------
  //                 c

  tensor_t a = writer->ew(silu, writer->matmul(x, w1t));
  tensor_t b = writer->matmul(x, w3t) ;

  tensor_t c = writer->mul(a, b);

  return writer->matmul(c, w2t);
}

transformer_block_t::transformer_block_t(
  graph_writer_t* w,
  int layer_id,
  model_args_t args)
  : writer(w), layer_id(layer_id), args(args)
{
  attention = attention_t(writer, "attention.", args);

  uint64_t hidden_dim = 4 * args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    args.multiple_of * ( (hidden_dim + args.multiple_of - 1) / args.multiple_of );
  feedforward = feedforward_t(writer, "feed_forward.", args.full_dim(), hidden_dim);

  attention_norm   = rms_norm_t(writer, "attention_norm.", args.full_dim(), args.norm_eps);
  feedforward_norm = rms_norm_t(writer, "ffn_norm.",       args.full_dim(), args.norm_eps);
}

map<int, string> transformer_block_t::input_map() const {
  //TODO: loop the input_names mappping and insert to our
  //      own map for both att and ffn

  map<int, string> input_names;

  //Attention_t names mapping
  map<int, string> att_inner_names = attention.input_map();
  for(auto const& [tensor_id, name]: attention.input_map()) {
    input_names.insert({
      tensor_id,
      "layers." + std::to_string(layer_id) + "." + name
    });
  }

  //feedforward names mapping
  for(auto const& [tensor_id, name]: feedforward.input_map()) {
    input_names.insert({
      tensor_id,
      "layers." + std::to_string(layer_id) + "." + name
    });
  }

  // TODO: attention_norm and ffn_norm mappings
}

tensor_t transformer_block_t::forward(
  tensor_t x,
  uint64_t start_pos,
  tensor_t freqs_cis,
  tensor_t mask)
{
  tensor_t h = writer->add(
    x,
    attention.forward(attention_norm.forward(x), start_pos, freqs_cis, mask));
  tensor_t out = writer->add(
    h,
    feedforward.forward(feedforward_norm.forward(h)));
  return out;
}

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
      writer, layer_id, params);
  }

  // TODO
  //norm = rms_norm_t(writer, "norm.weight", params.dim, params.norm_eps);

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

