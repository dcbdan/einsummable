#include "builder.h"

#include "../src/execution/cpu/repartition.h"

builder_t
builder_t::make_first_token(
  model_args_t const& args,
  uint64_t seqlen)
{
  bool is_last = seqlen >= args.max_seq_len;

  graph_writer_t writer;

  tensor_t embeddings = writer.input(full_shape_t({
    full_dim_t::singleton(args.batch_size),
    full_dim_t::singleton(seqlen),
    args.full_dim()
  }));

  transformer_t model(&writer, args, 0);
  tensor_t scores = model.forward(embeddings);

  tensor_t freq_cis = model.full_freqs_cis;

  auto model_weight_map = model.weight_map();

  // before making the taskgraph, save all the required tensors
  scores.save_inplace();
  if(!is_last) {
    freq_cis.save_inplace();
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

  auto make_inn = [&](tensor_t t) {
    int gid = t.get_id();
    return tinfo_t {
      .dtype = writer.get_graph().out_dtype(gid),
      .placement = pls.at(gid),
      .tids = inns_g_to_t.at(gid)
    };
  };
  auto make_save = [&](tensor_t t) {
    int gid = t.get_id();
    return tinfo_t {
      .dtype = writer.get_graph().out_dtype(gid),
      .placement = pls.at(gid),
      .tids = saves_g_to_t.at(gid)
    };
  };
  auto make_innsave = [&](tensor_t t) {
    auto const& infoi = make_inn(t);
    auto const& infos = make_save(t);
    if(!vector_equal(infoi.tids.get(), infos.tids.get())) {
      throw std::runtime_error("make_innsave");
    }
    return infoi;
  };
  auto make_transfer = [&](tensor_t t) {
    if(is_last) {
      return make_inn(t);
    } else {
      return make_innsave(t);
    }
  };

  map<string, tinfo_t> weights;
  for(auto const& [name,tensor]: model_weight_map) {
    weights.insert({name, make_transfer(tensor)});
  }

  optional<vector<tuple<tinfo_t, tinfo_t>>> next_kv;
  next_kv = vector<tuple<tinfo_t, tinfo_t>>();
  auto& _next_kv = next_kv.value();
  if(!is_last) {
    for(auto const& [k,v]: model.get_new_kvs()) {
      _next_kv.emplace_back(make_save(k), make_save(v));
    }
  }

  optional<tinfo_t> mask;
  if(model.mask) {
    mask = make_inn(model.mask.value());
  }

  return builder_t {
    .args                   = args,
    .graph                  = std::move(writer.get_graph()),
    .taskgraph              = std::move(taskgraph),
    .start_pos              = 0,
    .weights                = std::move(weights),
    .freqs_cis              = make_transfer(model.full_freqs_cis),
    .embeddings             = make_inn(embeddings),
    .prev_kv                = std::nullopt,
    .next_kv                = std::move(next_kv),
    .mask                   = std::move(mask),
    .scores                 = make_save(scores),
    .prev_tid_to_input_tids = std::nullopt
  };
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
