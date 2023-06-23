#include "modules.h"

#include "../src/base/hrect.h"

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
    .vocab_size      = 0,
    .world_size      = 1
  };
}

model_args_t model_args_t::llama_7B() {
  return model_args_t {
    .dim             = 4096,
    .n_layers        = 32,
    .n_heads         = 32,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .max_batch_size  = 32,
    .max_seq_len     = 256,
    .vocab_size      = 32000,
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
  weight = writer->input(full_shape_t({ dim }), dtype);
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

  auto x_shape = x.get_shape()();
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

  int out_rank = x.rank();

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
  : writer(w), args(args), name(name),
    n_local_heads(args.n_local_heads()),
    head_dim(args.head_dim())
{
  //input tensors

  //  outshape comes first because F.linear is xA^t,
  //  so shape is (out_features, in_features)

  full_shape_t kqv_shape({ args.full_dim(), args.full_dim() });

  wq = writer->input(kqv_shape);
  wk = writer->input(kqv_shape);
  wv = writer->input(kqv_shape);
  wo = writer->input(kqv_shape);
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
  tensor_t freqs_cis,
  optional<tensor_t> mask)
{
  dtype_t dtype = x.get_dtype();
  vector<uint64_t> input_shape = x.get_shape()();
  uint64_t bsz = input_shape[0];
  uint64_t seqlen = input_shape[1];

  if(input_shape.size() != 3 || input_shape[2] != args.dim) {
    throw std::runtime_error("invalid shape x to attention forward");
  }

  tensor_t xq = writer->matmul(x, wq.transpose(0,1));
  tensor_t xk = writer->matmul(x, wk.transpose(0,1));
  tensor_t xv = writer->matmul(x, wv.transpose(0,1));

  vector<uint64_t> full_xshape = {
    bsz, seqlen, n_local_heads, head_dim
  };

  xq = xq.view_full(full_xshape);
  xk = xk.view_full(full_xshape);
  xv = xv.view_full(full_xshape);

  xq = apply_rotary_embedding(xq.to_f32(), freqs_cis).to_dtype(dtype);
  xk = apply_rotary_embedding(xk.to_f32(), freqs_cis).to_dtype(dtype);

  auto [keys, values] = get_keys_and_values(xk, xv);

  xq = xq.transpose(1, 2);
  keys = keys.transpose(1, 2);
  values = values.transpose(1, 2);

  scalarop_t scale = scalarop_t::make_scale(
    scalar_t(dtype, write_with_ss(
      1.0 / (std::sqrt(double(1.0) * args.head_dim()))
    ))
  );

  tensor_t scores;
  scores = writer->matmul(xq, keys.transpose(2, 3));
  scores = writer->ew(scale, scores);

  if(mask) {
    scores = writer->ew(
      "abcd,cd->abcd",
      scalarop_t::make_add(scores.get_dtype()),
      scores,
      mask.value());
  }

  scores = writer->softmax(scores.to_f32()).to_dtype(dtype);

  tensor_t output;
  output = writer->matmul(scores, values);
  output = output.transpose(1, 2);

  full_shape_t output_shape({
    full_dim_t::singleton(bsz),
    full_dim_t::singleton(seqlen),
    full_dim_t({n_local_heads, head_dim})
  });
  output = output.view(output_shape);

  return writer->matmul(output, wo.transpose(0,1));
}

tuple<tensor_t, tensor_t>
attention_t::get_keys_and_values(tensor_t k, tensor_t v)
{
  if(k.get_shape() != v.get_shape()) {
    throw std::runtime_error("k and v must have the same shape");
  }

  if(prev_k) {
    prev_k = writer->concat(1, {prev_k.value(), k});
    prev_v = writer->concat(1, {prev_v.value(), v});
  } else {
    prev_k = k;
    prev_v = v;
  }

  return {prev_k.value(), prev_v.value()};
}

feedforward_t::feedforward_t(
  graph_writer_t* w,
  string name,
  full_dim_t dim,
  uint64_t hidden_dim)
  : writer(w), name(name)
{
  full_shape_t to_hidden(
    { full_dim_t::singleton(hidden_dim), dim }
  );
  full_shape_t to_dim(
    { dim, full_dim_t::singleton(hidden_dim) }
  );

  w1 = writer->input(to_hidden);
  w2 = writer->input(to_dim);
  w3 = writer->input(to_hidden);
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
  map<int, string> input_names;

  string header = "layers." + std::to_string(layer_id) + ".";

  //Attention_t names mapping
  map<int, string> att_inner_names = attention.input_map();
  for(auto const& [tensor_id, name]: attention.input_map()) {
    input_names.insert({tensor_id, header + name});
  }

  //feedforward names mapping
  for(auto const& [tensor_id, name]: feedforward.input_map()) {
    input_names.insert({tensor_id, header + name});
  }

  for(auto const& [tensor_id, name]: attention_norm.input_map()) {
    input_names.insert({tensor_id, header + name});
  }
  for(auto const& [tensor_id, name]: feedforward_norm.input_map()) {
    input_names.insert({tensor_id, header + name});
  }

  return input_names;
}

tensor_t transformer_block_t::forward(
  tensor_t x,
  tensor_t freqs_cis,
  optional<tensor_t> mask)
{
  tensor_t h = writer->add(
    x,
    attention.forward(attention_norm.forward(x), freqs_cis, mask));
  tensor_t out = writer->add(
    h,
    feedforward.forward(feedforward_norm.forward(h)));
  return out;
}

transformer_t::transformer_t(
  graph_writer_t* w,
  std::string name,
  model_args_t args)
  : writer(w), args(args)
{
  for (int layer_id = 0; layer_id != args.n_layers; ++layer_id) {
    //loop over input_map and insert into our own map.
    layers.emplace_back(
      writer, layer_id, args);
  }

  full_freqs_cis = writer->input(
    { args.max_seq_len, uint64_div(args.head_dim(), 2) },
    dtype_t::c64);

  norm = rms_norm_t(writer, "norm.", args.full_dim(), args.norm_eps);

  full_shape_t to_vocab({
    full_dim_t::singleton(args.vocab_size),
    args.full_dim()
  });
  w_vocab = writer->input(to_vocab);
}

tensor_t transformer_t::forward(tensor_t x)
{
  auto x_shape = x.get_shape()();
  uint64_t const& bsz    = x_shape[0];
  uint64_t const& seqlen = x_shape[1];

  optional<tensor_t> mask;
  if(seqlen > 1) {
    if(start_pos() != 0) {
      throw std::runtime_error(
        "Masks are only supported on the first iteration");
      // TODO: how do masks work and should they be supported
      //       whenever seqlen > 1?
    }
    mask = next_mask(seqlen);
  }

  tensor_t freqs_cis = next_freqs_cis(seqlen);

  for(auto& layer: layers) {
    x = layer.forward(x, freqs_cis, mask);
  }
  x = norm.forward(x);
  // x: bsz, seqlen, dim

  using _all = graph_writer_t::idx_t::all;
  using _idx = graph_writer_t::idx_t::idx;
  x = x.subset({ _all{}, _idx{-1}, _all{} });
  // x: bsz, dim

  return writer->matmul(x, w_vocab.transpose(0,1));
}

map<int, string> transformer_t::input_map() const {
  map<int, string> ret;
  for(transformer_block_t const& block: layers) {
    for(auto const& [tensor_id, name]: block.input_map()) {
      ret.insert({tensor_id, name});
    }
  }

  for(auto const& [tensor_id, name]: norm.input_map()) {
    ret.insert({tensor_id, name});
  }

  ret.insert({w_vocab.get_id(), "output.weight"});

  return ret;
}

tensor_t transformer_t::next_freqs_cis(uint64_t n) {
  using _all = graph_writer_t::idx_t::all;
  using _rng = graph_writer_t::idx_t::rng;
  using idx_t = graph_writer_t::idx_t;

  vector<uint64_t> shape = full_freqs_cis.get_shape()();

  vector<idx_t> subset(shape.size(), _all{});

  int64_t b0 = int64_t(start_pos());
  int64_t e0 = int64_t(start_pos() + n);
  subset[0] = _rng{ b0, e0 };

  return full_freqs_cis.subset(subset);
}

tensor_t transformer_t::next_mask(uint64_t seqlen) {
  mask_infos.push_back(mask_info_t {
    .mask = writer->input({ seqlen, seqlen }),
    .start_pos = start_pos(),
    .seqlen = seqlen
  });
  return mask_infos.back().mask;
}
