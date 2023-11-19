#include "builder.h"

builder_t
builder_t::_make(
  model_args_t const& args,
  uint64_t start_pos,
  uint64_t seqlen,
  bool is_last)
{
  bool is_first = start_pos == 0;

  graph_writer_t writer;

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(args.batch_size),
    full_dim_t::singleton(seqlen),
    args.full_dim()
  }));

  transformer_t model(&writer, args, start_pos);
  tensor_t scores = model.forward(embeddings);

  tensor_t freqs_cis = model.full_freqs_cis;

  auto model_weight_map = model.weight_map();

  // before making the taskgraph, save all the required tensors
  scores.save_inplace();
  if(!is_last) {
    freqs_cis.save_inplace();
    for(auto& [name, weight]: model_weight_map) {
      weight.save_inplace();
    }
    for(auto [k,v]: model.get_new_kvs()) {
      k.save_inplace();
      v.save_inplace();
    }
  }

  map<string, int> weights;
  for(auto const& [name,tensor]: model_weight_map) {
    weights.insert({name, tensor.get_id()});
  }

  optional<vector<tuple<int, int>>> prev_kv;
  if(!is_first) {
    prev_kv = vector<tuple<int,int>>();
    auto& _prev_kv = prev_kv.value();
    for(auto const& [k,v]: model.get_prev_kvs()) {
      _prev_kv.emplace_back(k.get_id(), v.get_id());
    }
  }

  optional<vector<tuple<int, int>>> next_kv;
  if(!is_last) {
    next_kv = vector<tuple<int, int>>();
    auto& _next_kv = next_kv.value();
    for(auto const& [k,v]: model.get_new_kvs()) {
      _next_kv.emplace_back(k.get_id(), v.get_id());
    }
  }

  optional<int> mask;
  if(model.mask) {
    mask = model.mask.value().get_id();
  }

  return builder_t {
    .args                   = args,
    .graph                  = std::move(writer.get_graph()),
    .start_pos              = start_pos,
    .weights                = std::move(weights),
    .freqs_cis              = freqs_cis.get_id(),
    .embeddings             = embeddings.get_id(),
    .prev_kv                = std::move(prev_kv),
    .next_kv                = std::move(next_kv),
    .mask                   = std::move(mask),
    .scores                 = scores.get_id(),
    .remap                  = std::nullopt
  };
}

builder_t
builder_t::make_first_token(
  model_args_t const& args,
  uint64_t seqlen)
{
  return _make(args, 0, seqlen, false);
}

builder_t
builder_t::make_next_token(
  builder_t const& prev,
  bool make_last)
{
  if(prev.is_last()) {
    throw std::runtime_error("can't build next token from a last builder");
  }

  auto const& args = prev.args;

  uint64_t start_pos;
  if(prev.next_kv) {
    auto next_kv = prev.next_kv.value();
    auto const& [key_gid, _] = next_kv.at(0);
    auto key_shape = prev.graph.nodes[key_gid].op.out_shape();
    start_pos = key_shape.at(1);
  } else {
    throw std::runtime_error("the key values need to be returned");
  }

  uint64_t seqlen = 1;

  builder_t ret = _make(args, start_pos, seqlen, make_last);

  ret.remap = vector<tuple<int,int>>();
  auto& remap = ret.remap.value();

  // Copy the tids from prev into the new items:
  //   weights
  //   freqs_cis
  //   next_kv -> prev_kv
  for(auto const& [name, gid_new]: ret.weights) {
    auto const& gid_prev = prev.weights.at(name);
    remap.emplace_back(gid_prev, gid_new);
  }

  remap.emplace_back(prev.freqs_cis, ret.freqs_cis);

  {
    auto const& prev_kv = prev.next_kv.value();
    auto const& next_kv = ret.prev_kv.value();
    if(prev_kv.size() != next_kv.size() || prev_kv.size() != args.n_layers) {
      throw std::runtime_error("prev and next kv should be n_layers length");
    }
    for(int i = 0; i != args.n_layers; ++i) {
      auto const& [prev_k, prev_v] = prev_kv[i];
      auto const& [new_k,  new_v]  = next_kv[i];
      remap.emplace_back(prev_k, new_k);
      remap.emplace_back(prev_v, new_v);
    }
  }

  return ret;
}

void builder_t::print_info() const {
  std::cout << "start_pos " << start_pos << std::endl;
  std::cout << "weight ids: " << std::endl;
  for(auto const& [name, gid]: weights) {
    std::cout << "  " << name << " " << gid << std::endl;
  }
  std::cout << "freqs_cis " << freqs_cis << std::endl;
  std::cout << "embeddings " << embeddings << std::endl;
  if(prev_kv) {
    std::cout << "prev_kv: " << std::endl;
    for(auto const& [k,v]: prev_kv.value()) {
      std::cout << "  " << "kv: " << k << ", " << v << std::endl;
    }
  }
  if(next_kv) {
    std::cout << "next_kv: " << std::endl;
    for(auto const& [k,v]: next_kv.value()) {
      std::cout << "  " << "kv: " << k << ", " << v << std::endl;
    }
  }
}

dtype_t builder_t::input_dtype(int gid) const {
  auto const& node = graph.nodes[gid];
  if(!node.op.is_input()) {
    throw std::runtime_error("invalid: builder expects input");
  }
  return node.op.out_dtype();
}

vector<uint64_t> builder_t::input_shape(int gid) const {
  auto const& node = graph.nodes[gid];
  if(!node.op.is_input()) {
    throw std::runtime_error("invalid: builder expects input");
  }
  return node.op.out_shape();
}

