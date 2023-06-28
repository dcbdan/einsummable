#include "builder.h"

#include "../src/execution/cpu/repartition.h"

struct make_inn_save_transfer_t {
  using tinfo_t = builder_t::tinfo_t;

  tinfo_t make_inn(tensor_t t) const {
    int gid = t.get_id();
    dtype_t dtype = t.get_dtype();
    return tinfo_t {
      .dtype = dtype,
      .placement = pls.at(gid),
      .tids = inns_g_to_t.at(gid)
    };
  }

  tinfo_t make_save(tensor_t t) const {
    int gid = t.get_id();
    dtype_t dtype = t.get_dtype();
    return tinfo_t {
      .dtype = dtype,
      .placement = pls.at(gid),
      .tids = saves_g_to_t.at(gid)
    };
  }

  tinfo_t make_innsave(tensor_t t) const {
    auto const& infoi = make_inn(t);
    auto const& infos = make_save(t);
    if(!vector_equal(infoi.tids.get(), infos.tids.get())) {
      throw std::runtime_error("make_innsave");
    }
    return infoi;
  }

  tinfo_t make_transfer(tensor_t t) const {
    if(is_last) {
      return make_inn(t);
    } else {
      return make_innsave(t);
    }
  }

  bool is_last;
  vector<placement_t> const& pls;
  map<int, vtensor_t<int>> const& inns_g_to_t;
  map<int, vtensor_t<int>> const& saves_g_to_t;
};

builder_t
builder_t::_make(
  model_args_t const& args,
  uint64_t start_pos,
  uint64_t seqlen)
{
  bool is_first = start_pos == 0;
  bool is_last = start_pos + seqlen >= args.max_seq_len;

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

  // TODO: will need to choose a proper placement
  vector<placement_t> pls = writer.get_graph().make_singleton_placement();

  auto const& [inns_g_to_t, saves_g_to_t, taskgraph] = taskgraph_t::make(
    writer.get_graph(), pls);

  make_inn_save_transfer_t m { is_last, pls, inns_g_to_t, saves_g_to_t };

  map<string, tinfo_t> weights;
  for(auto const& [name,tensor]: model_weight_map) {
    weights.insert({name, m.make_transfer(tensor)});
  }

  optional<vector<tuple<tinfo_t, tinfo_t>>> prev_kv;
  if(!is_first) {
    prev_kv = vector<tuple<tinfo_t, tinfo_t>>();
    auto& _prev_kv = prev_kv.value();
    for(auto const& [k,v]: model.get_prev_kvs()) {
      _prev_kv.emplace_back(m.make_inn(k), m.make_inn(v));
    }
  }

  optional<vector<tuple<tinfo_t, tinfo_t>>> next_kv;
  if(!is_last) {
    next_kv = vector<tuple<tinfo_t, tinfo_t>>();
    auto& _next_kv = next_kv.value();
    for(auto const& [k,v]: model.get_new_kvs()) {
      _next_kv.emplace_back(m.make_save(k), m.make_save(v));
    }
  }

  optional<tinfo_t> mask;
  if(model.mask) {
    mask = m.make_inn(model.mask.value());
  }

  return builder_t {
    .args                   = args,
    .graph                  = std::move(writer.get_graph()),
    .taskgraph              = std::move(taskgraph),
    .start_pos              = start_pos,
    .weights                = std::move(weights),
    .freqs_cis              = m.make_transfer(freqs_cis),
    .embeddings             = m.make_inn(embeddings),
    .prev_kv                = std::move(prev_kv),
    .next_kv                = std::move(next_kv),
    .mask                   = std::move(mask),
    .scores                 = m.make_save(scores),
    .prev_tid_to_input_tids = std::nullopt
  };
}

builder_t
builder_t::make_first_token(
  model_args_t const& args,
  uint64_t seqlen)
{
  return _make(args, 0, seqlen);
}

builder_t
builder_t::make_next_token(builder_t const& prev)
{
  if(prev.is_last()) {
    throw std::runtime_error("can't build next token from a last builder");
  }
  auto const& args = prev.args;

  uint64_t start_pos;
  if(prev.next_kv) {
    auto next_kv = prev.next_kv.value();
    auto const& [key_info, _] = next_kv.at(0);
    auto key_shape = key_info.placement.total_shape();
    start_pos = key_shape.at(1);
  } else {
    throw std::runtime_error("the key values need to be returned");
  }

  uint64_t seqlen = 1;

  builder_t ret = _make(args, start_pos, seqlen);

  ret.prev_tid_to_input_tids = map<int,int>();
  map<int,int>& conversion = ret.prev_tid_to_input_tids.value();

  // Copy the tids from prev into the new items:
  //   weights
  //   freqs_cis
  //   next_kv -> prev_kv
  for(auto const& [name, t_new]: ret.weights) {
    auto const& t_prev = prev.weights.at(name);
    same_placement_convert(conversion, t_prev, t_new);
  }

  same_placement_convert(conversion, prev.freqs_cis, ret.freqs_cis);

  {
    auto const& prev_kv = prev.next_kv.value();
    auto const& next_kv = ret.prev_kv.value();
    if(prev_kv.size() != next_kv.size() || prev_kv.size() != args.n_layers) {
      throw std::runtime_error("prev and next kv should be n_layers length");
    }
    for(int i = 0; i != args.n_layers; ++i) {
      auto const& [prev_k, prev_v] = prev_kv[i];
      auto const& [new_k,  new_v]  = next_kv[i];
      same_placement_convert(conversion, prev_k, new_k);
      same_placement_convert(conversion, prev_v, new_v);
    }
  }

  return ret;
}

void builder_t::transform_from_prev(map<int, buffer_t>& data) const
{
  map<int, buffer_t> ret;

  auto const& to_new_tid = prev_tid_to_input_tids.value();
  for(auto const& [p,n]: to_new_tid) {
    ret.insert({n, data.at(p)});
  }

  data = ret;
}

void builder_t::print_info() const {
  std::cout << "start_pos " << start_pos << std::endl;
  std::cout << "weight ids: " << std::endl;
  for(auto const& [name, tinfo]: weights) {
    std::cout << "  " << name << " " << tinfo.tids.get() << std::endl;
  }
  std::cout << "freqs_cis " << freqs_cis.tids.get() << std::endl;
  std::cout << "embeddings " << embeddings.tids.get() << std::endl;
  if(prev_kv) {
    std::cout << "prev_kv: " << std::endl;
    for(auto const& [k,v]: prev_kv.value()) {
      std::cout << "  " << "kv: " << k.tids.get() << ", " << v.tids.get() << std::endl;
    }
  }
  if(next_kv) {
    std::cout << "next_kv: " << std::endl;
    for(auto const& [k,v]: next_kv.value()) {
      std::cout << "  " << "kv: " << k.tids.get() << ", " << v.tids.get() << std::endl;
    }
  }
}

void builder_t::same_placement_convert(
  map<int, int>& prev_to_new,
  tinfo_t const& prev_info,
  tinfo_t const& new_info)
{
  if(prev_info.dtype != new_info.dtype) {
    throw std::runtime_error("same_placement_convert: must have same dtype");
  }
  if(prev_info.placement.partition != new_info.placement.partition) {
    throw std::runtime_error("same_placement_convert: must have same partition");
  }
  if(!vector_equal(
    prev_info.placement.locations.get(),
    new_info.placement.locations.get()))
  {
    throw std::runtime_error("same_placement_convert: must have same placement");
  }

  auto const& ps = prev_info.tids.get();
  auto const& ns = new_info.tids.get();

  int n = ps.size();
  if(n != ns.size()) {
    throw std::runtime_error("very bad: mismatch in tid sizing even though same placements");
  }
  for(int i = 0; i != n; ++i) {
    prev_to_new.insert({ps[i], ns[i]});
  }
}

void repartition_into_map_single_loc(
  map<int, buffer_t>& tid_to_buffer,
  builder_t::tinfo_t const& tinfo,
  buffer_t inn_relation)
{
  vtensor_t<buffer_t> bs = repartition(
    tinfo.dtype, tinfo.placement.partition, inn_relation);

  int nitems = tinfo.tids.get().size();
  for(int i = 0; i != nitems; ++i) {
    int const& tid = tinfo.tids.get()[i];
    buffer_t b = bs.get()[i];
    tid_to_buffer.insert({tid, b});
  }
}

dbuffer_t unpartitioned_from_map_single_loc(
  map<int, buffer_t>& tid_to_buffer,
  builder_t::tinfo_t const& tinfo)
{
  vtensor_t<buffer_t> data(tinfo.placement.partition.block_shape());
  int nitems = tinfo.tids.get().size();
  for(int i = 0; i != nitems; ++i) {
    int const& tid = tinfo.tids.get()[i];
    data.get()[i] = tid_to_buffer.at(tid);
  }

  auto ret = repartition(
    tinfo.dtype,
    partition_t::singleton(tinfo.placement.total_shape()),
    data,
    tinfo.placement.partition);

  return dbuffer_t(tinfo.dtype, ret.get()[0]);
}
