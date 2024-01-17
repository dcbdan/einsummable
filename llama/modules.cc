#include "modules.h"

#include "../src/base/hrect.h"

#define SAVE_TENSOR(t, x) \
  [&] { \
    auto ret = t.save(); \
    DLINEOUT(x << " | tensor id " << ret.get_id()); \
    return ret; \
  }()
#define NO_SAVE_TENSOR(t, x) \
  [&] { \
    return t; \
  }()

model_args_t model_args_t::llama(int n, uint64_t batch_size) {
  if(n == 1) { return llama_7B( batch_size); }
  if(n == 2) { return llama_13B(batch_size); }
  if(n == 4) { return llama_30B(batch_size); }
  if(n == 8) { return llama_65B(batch_size); }
  throw std::runtime_error("must have 1,2,4,8 number of llama \"n\"");
}

model_args_t model_args_t::llama_7B(uint64_t batch_size) {
  return model_args_t {
    .dim             = 4096,
    .n_layers        = 32,
    .n_heads         = 32,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .batch_size      = batch_size,
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
  };
}


model_args_t model_args_t::llama_13B(uint64_t batch_size) {
  return model_args_t {
    .dim             = 5120,
    .n_layers        = 40,
    .n_heads         = 40,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .batch_size      = batch_size,
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
  };
}

model_args_t model_args_t::llama_30B(uint64_t batch_size) {
  return model_args_t {
    .dim             = 6656,
    .n_layers        = 60,
    .n_heads         = 52,
    .multiple_of     = 256,
    .norm_eps        = 1e-6,
    .batch_size      = batch_size,
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
  };
}

model_args_t model_args_t::llama_65B(uint64_t batch_size) {
  return model_args_t {
    .dim             = 8192,
    .n_layers        = 80,
    .n_heads         = 64,
    .multiple_of     = 256,
    .norm_eps        = 1e-5,
    .batch_size      = batch_size,
    .max_seq_len     = 2048,
    .vocab_size      = 32000,
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

map<string, tensor_t> rms_norm_t::weight_map() const
{
  map<string, tensor_t> ret;
  ret.insert({name + "weight", weight});
  return ret;
}

tensor_t rms_norm_t::norm(tensor_t x) {
  dtype_t d = x.get_dtype();

  if(d == dtype_t::f16) {
    throw std::runtime_error("rms_norm_t::norm needs >16 precision");
  }

  auto x_shape = x.get_shape()();
  int out_rank = x_shape.size();
  if(out_rank <= 1) {
    throw std::runtime_error("rms_norm: not a big enough output rank");
  }

  scalarop_t inverse_sqrt = scalarop_t::make_inverse_sqrt(d);
  scalarop_t square       = scalarop_t::make_square(d);
  scalarop_t mul          = scalarop_t::make_mul(d);

  scalar_t _e(d, write_with_ss(eps));
  scalar_t _a(d, write_with_ss(1.0/double(double(1.0)*x_shape.back())));
  scalarop_t scale_then_add_eps = scalarop_t::combine(
    scalarop_t::make_add(d),
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

  // compute output with a minimum precision of 32
  tensor_t output;
  if(dtype == dtype_t::f16) {
    output = norm(x.to_dtype(dtype_t::f32)).to_dtype(dtype);
  } else {
    output = norm(x);
  }

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
  model_args_t args,
  uint64_t start_pos)
  : writer(w), args(args), name(name),
    batch_size(args.batch_size),
    n_heads(args.n_heads),
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

  if(start_pos != 0) {
    vector<uint64_t> prev_shape({
      batch_size, start_pos, n_heads, head_dim });
    prev_kv = tuple<tensor_t, tensor_t> {
      writer->input(prev_shape),
      writer->input(prev_shape)
    };
  }
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

map<string, tensor_t> attention_t::weight_map() const {
  return map<string, tensor_t> {
    {name + "wq.weight", wq },
    {name + "wk.weight", wk },
    {name + "wv.weight", wv },
    {name + "wo.weight", wo }
  };
}

tensor_t attention_t::forward(
  tensor_t x,
  tensor_t freqs_cis,
  optional<tensor_t> mask)
{
  dtype_t dtype = x.get_dtype();
  vector<uint64_t> input_shape = x.get_shape()();
  uint64_t seqlen = input_shape[1];

  if(input_shape.size() != 3 || input_shape[2] != args.dim) {
    throw std::runtime_error("invalid shape x to attention forward");
  }

  tensor_t xq = writer->matmul(x, wq.transpose(0,1));
  tensor_t xk = writer->matmul(x, wk.transpose(0,1));
  tensor_t xv = writer->matmul(x, wv.transpose(0,1));

  vector<uint64_t> full_xshape = {
    batch_size, seqlen, n_heads, head_dim
  };

  xq = xq.view_full(full_xshape);
  xk = xk.view_full(full_xshape);
  xv = xv.view_full(full_xshape);

  xq = apply_rotary_embedding(xq.to_f32(), freqs_cis).to_dtype(dtype);
  xk = apply_rotary_embedding(xk.to_f32(), freqs_cis).to_dtype(dtype);

  // set next_kv for use later
  set_next_keys_and_values(xk, xv);
  // copy the tensor objects so that we can transpose them here
  auto [keys, values] = next_kv.value();
  // batch_size, start_pos + seqlen, n_heads, head_dim

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

  // compute softmax with a minimum of 32 bits precision
  if(dtype == dtype_t::f16) {
    scores = writer->softmax(scores.to_f32()).to_dtype(dtype);
  } else {
    scores = writer->softmax(scores);
  }

  tensor_t output;
  output = writer->matmul(scores, values);
  output = output.transpose(1, 2);

  full_shape_t output_shape({
    full_dim_t::singleton(batch_size),
    full_dim_t::singleton(seqlen),
    full_dim_t({n_heads, head_dim})
  });
  output = output.view(output_shape);

  return writer->matmul(output, wo.transpose(0,1));
}

void attention_t::set_next_keys_and_values(tensor_t k, tensor_t v)
{
  if(k.get_shape() != v.get_shape()) {
    throw std::runtime_error("k and v must have the same shape");
  }

  if(next_kv) {
    throw std::runtime_error("can't have multiple next kvs");
  }

  next_kv = tuple<tensor_t, tensor_t>();
  auto& [next_k, next_v] = next_kv.value();

  if(prev_kv) {
    auto const& [prev_k, prev_v] = prev_kv.value();
    next_k = writer->concat(1, {prev_k, k});
    next_v = writer->concat(1, {prev_v, v});
  } else {
    next_k = k;
    next_v = v;
  }
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

map<string, tensor_t> feedforward_t::weight_map() const {
  return map<string, tensor_t> {
    { name + "w1.weight", w1, },
    { name + "w2.weight", w2, },
    { name + "w3.weight", w3, }
  };
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
  model_args_t args,
  uint64_t start_pos)
  : writer(w), layer_id(layer_id), args(args)
{
  attention = attention_t(writer, "attention.", args, start_pos);

  uint64_t hidden_dim = 4 * args.dim;
  hidden_dim = uint64_t( (2.0 * hidden_dim) / 3.0 );
  hidden_dim =
    args.multiple_of * ( (hidden_dim + args.multiple_of - 1) / args.multiple_of );
  feedforward = feedforward_t(writer, "feed_forward.", args.full_dim(), hidden_dim);

  attention_norm   = rms_norm_t(writer, "attention_norm.", args.full_dim(), args.norm_eps);
  feedforward_norm = rms_norm_t(writer, "ffn_norm.",       args.full_dim(), args.norm_eps);
}

map<string, tensor_t> transformer_block_t::weight_map() const {
  map<string, tensor_t> ret;

  string header = "layers." + std::to_string(layer_id) + ".";

  //Attention_t names mapping
  for(auto const& [name, tensor_id]: attention.weight_map()) {
    ret.insert({header + name, tensor_id});
  }

  //feedforward names mapping
  for(auto const& [name, tensor_id]: feedforward.weight_map()) {
    ret.insert({header + name, tensor_id});
  }

  for(auto const& [name, tensor_id]: attention_norm.weight_map()) {
    ret.insert({header + name, tensor_id});
  }
  for(auto const& [name, tensor_id]: feedforward_norm.weight_map()) {
    ret.insert({header + name, tensor_id});
  }

  return ret;
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
  model_args_t args,
  uint64_t start_pos)
  : writer(w), args(args),
    expected_batch_size(args.batch_size), start_pos(start_pos)
{
  for (int layer_id = 0; layer_id != args.n_layers; ++layer_id) {
    layers.emplace_back(
      writer, layer_id, args, start_pos);
  }

  full_freqs_cis = writer->input(
    { 2*args.max_seq_len, uint64_div(args.head_dim(), 2) },
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

  if(bsz != expected_batch_size) {
    throw std::runtime_error("The batch size cannot be variable!");
  }

  if(seqlen > 1) {
    if(start_pos != 0) {
      throw std::runtime_error(
        "Masks are only supported on the first iteration");
      // TODO: how do masks work and should they be supported
      //       whenever seqlen > 1?
    }
    mask = writer->input({ seqlen, seqlen });
  }

  tensor_t freqs_cis = get_freqs_cis(seqlen);

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

map<string, tensor_t> transformer_t::weight_map() const {
  map<string, tensor_t> ret;

  for(transformer_block_t const& block: layers) {
    for(auto const& [name, tensor_id]: block.weight_map()) {
      ret.insert({name, tensor_id});
    }
  }

  for(auto const& [name, tensor_id]: norm.weight_map()) {
    ret.insert({name, tensor_id});
  }

  ret.insert({"output.weight", w_vocab});

  return ret;
}

dbuffer_t transformer_t::form_full_freqs_cis(model_args_t const& args) {
  return form_full_freqs_cis(args.dim, args.n_heads, args.max_seq_len);
}

dbuffer_t transformer_t::form_full_freqs_cis(
  uint64_t args_dim, uint64_t args_n_heads, uint64_t args_max_seq_len)
{
  uint64_t dim  = uint64_div(args_dim, args_n_heads);
  uint64_t end = 2*args_max_seq_len;

  return form_freqs_cis(dim, end);
}

dbuffer_t transformer_t::form_freqs_cis(
  uint64_t dim, uint64_t end)
{
  float theta = 10000.0;
  uint64_t hdim = uint64_div(dim, 2);

  dbuffer_t xs = make_dbuffer(dtype_t::f32, hdim);
  for(int i = 0; i != hdim; ++i) {
    xs.f32()[i] = 1.0 / (std::pow(theta, (i*2.0) / dim));
  }

  dbuffer_t t = make_dbuffer(dtype_t::f32, end);
  for(int i = 0; i != end; ++i) {
    t.f32()[i] = 1.0*i;
  }

  dbuffer_t freqs = make_dbuffer(dtype_t::f32, end*hdim);
  for(int i = 0; i != end;  ++i) {
  for(int j = 0; j != hdim; ++j) {
    freqs.f32()[i*hdim+j] = t.f32()[i] * xs.f32()[j];
  }}

  dbuffer_t freqs_cis = make_dbuffer(dtype_t::c64, end*hdim);
  for(int i = 0; i != end*hdim; ++i) {
    freqs_cis.c64()[i] = std::polar(float(1.0), freqs.f32()[i]);
  }

  return freqs_cis;
}

template <typename T>
void _form_start_mask_set_zero(uint64_t const& seqlen, T* data)
{
  for(int i = 0; i != seqlen; ++i) {
  for(int j = 0; j != i+1;    ++j) {
    data[i*seqlen + j] = 0.0;
  }}
}

dbuffer_t transformer_t::form_start_mask(uint64_t seqlen, dtype_t dtype) {
  dbuffer_t mask = make_dbuffer(dtype, seqlen*seqlen);

  mask.fill(scalar_t::negative_inf(dtype));

  if(dtype == dtype_t::f16) {
    _form_start_mask_set_zero(seqlen, mask.f16());
  } else if(dtype == dtype_t::f32) {
    _form_start_mask_set_zero(seqlen, mask.f32());
  } else if(dtype == dtype_t::f64) {
    _form_start_mask_set_zero(seqlen, mask.f64());
  } else {
    throw std::runtime_error("should not happen: invalid dtype form start mask");
  }

  return mask;
}

tensor_t transformer_t::get_freqs_cis(uint64_t n) {
  using _all = graph_writer_t::idx_t::all;
  using _rng = graph_writer_t::idx_t::rng;
  using idx_t = graph_writer_t::idx_t;

  vector<uint64_t> shape = full_freqs_cis.get_shape()();

  vector<idx_t> subset(shape.size(), _all{});

  int64_t b0 = int64_t(start_pos);
  int64_t e0 = int64_t(start_pos + n);
  subset[0] = _rng{ b0, e0 };

  return full_freqs_cis.subset(subset);
}

vector<tuple<tensor_t, tensor_t>>
transformer_t::get_prev_kvs() const {
  vector<tuple<tensor_t, tensor_t>> ret;
  for(auto const& layer: layers) {
    ret.push_back(layer.get_prev_kv());
  }
  return ret;
}

vector<tuple<tensor_t, tensor_t>>
transformer_t::get_new_kvs() const {
  vector<tuple<tensor_t, tensor_t>> ret;
  for(auto const& layer: layers) {
    ret.push_back(layer.get_new_kv());
  }
  return ret;
}
