#include "reference.h"
#include "../base/hrect.h"

void _assert_correct_dtype(string err_msg, dtype_t dtype, dbuffer_t const& data) {
  if(dtype != data.dtype) {
    throw std::runtime_error("incorrect dtype: " + err_msg);
  }
}
void _assert_correct_size(string err_msg, uint64_t size, dbuffer_t const& data) {
  if(size != data.size()) {
    throw std::runtime_error("incorrect size: " + err_msg);
  }
}

map<int, dbuffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, dbuffer_t> const& inputs)
{
  map<int, dbuffer_t> outs;
  map<int, dbuffer_t> tensors;

  for(int id = 0; id != graph.nodes.size(); ++id)
  {
    auto const& node = graph.nodes[id];

    if(node.op.is_formation()) {
      auto const& f = node.op.get_formation();
      auto const& expected_dtype = f.dtype;
      auto expected_size = dtype_size(f.dtype) * product(f.shape);
      auto const& t = tensors[node.inns[0]];
      _assert_correct_dtype("formation", expected_dtype, t);
      _assert_correct_size("formation", expected_size, t);
      tensors[id] = t;
    } else if(node.op.is_input()) {
      auto const& t = inputs.at(id);
      auto const& ii = node.op.get_input();
      auto const& expected_dtype = ii.dtype;
      auto expected_size = dtype_size(ii.dtype) * product(ii.shape);
      _assert_correct_dtype("input", expected_dtype, t);
      _assert_correct_size("input", expected_size, t);
      tensors[id] = t;
    } else if(node.op.is_einsummable()) {
      vector<dbuffer_t> inns;
      inns.reserve(node.inns.size());
      for(auto const& id_inn: node.inns) {
        inns.push_back(tensors[id_inn]);
      }
      tensors[id] = reference_einsummable(node.op.get_einsummable(), inns);
    } else if(node.op.is_concat()) {
      vector<dbuffer_t> inns;
      inns.reserve(node.inns.size());
      for(auto const& id_inn: node.inns) {
        inns.push_back(tensors[id_inn]);
      }
      tensors[id] = reference_concat(node.op.get_concat(), inns);
    } else {
      throw std::runtime_error("should not reach: reference compute graph");
    }

    if(node.op.is_save()) {
      outs[id] = tensors[id];
    }
  }

  return outs;
}

map<int, dbuffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, dbuffer_t> const& inputs)
{
  map<int, dbuffer_t> outs;

  // id -> (location, buffer)
  map<int, tuple<int, dbuffer_t> > tensors;
  auto get_at = [&tensors](int id, int loc, optional<dtype_t> dtype) {
    auto const& [actual_loc, buffer] = tensors.at(id);
    if(loc != actual_loc) {
      throw std::runtime_error("incorrect locs in taskgraph");
    }
    if(dtype && dtype.value() != buffer.dtype) {
      throw std::runtime_error("incorrect dtype in reference compute taskgraph");
    }
    return buffer;
  };

  for(auto const& id: taskgraph.get_order())
  {
    auto const& node = taskgraph.nodes[id];

    if(node.op.is_input()) {
      auto const& ii = node.op.get_input();
      dtype_t const& expected_dtype = ii.dtype;
      uint64_t expected_size = dtype_size(ii.dtype) * ii.nelem;
      auto const& t = inputs.at(id);
      _assert_correct_dtype("tg input", expected_dtype, t);
      _assert_correct_size("tg input", expected_size, t);
      tensors[id] = {node.op.out_loc(), t};
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();

      vector<dbuffer_t> inputs;
      inputs.reserve(apply.inns.size());
      for(auto const& inn: apply.inns) {
        // reference_einsummable will check the input dtype
        inputs.push_back(get_at(inn, apply.loc, std::nullopt));
      }

      tensors[id] = {
        apply.loc,
        reference_einsummable(apply.einsummable,inputs)
      };
    } else if(node.op.is_move()) {
      auto const& move = node.op.get_move();
      dbuffer_t buffer_src = get_at(move.inn, move.src, move.dtype);
      dbuffer_t buffer_dst = buffer_src.copy();
      tensors[id] = std::make_tuple(move.dst, buffer_dst);
    } else if(node.op.is_partialize()) {
      int loc = node.op.out_loc();
      dtype_t dtype = node.op.out_dtype();

      dbuffer_t write = make_dbuffer(dtype, node.op.out_nelem());
      tensors[id] = {loc, write};

      // Note: ignoring consummables
      for(vector<tuple<int, touch_t>> const& ts: node.op.get_touches()) {
        if(ts.size() == 0) {
          continue;
        }

        // The first touch is an initialize, not an update,
        // so set the castable to none
        auto const& [inn0, t0] = ts[0];
        reference_touch(
          touch_t {
            .selection = t0.selection,
            .castable = std::optional<castable_t>(),
            .dtype = dtype
          },
          write,
          get_at(inn0, loc, std::nullopt));

        for(int i = 1; i < ts.size(); ++i) {
          auto const& [inn, t] = ts[i];
          reference_touch(t, write, get_at(inn, loc, std::nullopt));
        }
      }
    } else {
      throw std::runtime_error("should not reach");
    }

    if(node.is_save) {
      outs[id] = std::get<1>(tensors[id]);
    }
  }

  return outs;
}

void reference_compute_memgraph(
  memgraph_t const& memgraph,
  vector<buffer_t>& compute_location_buffers)
{
  auto make_buffer_at = [&](int loc, mem_t const& mem) {
    return make_buffer_reference(
      compute_location_buffers[loc]->data + mem.offset,
      mem.size);
  };
  auto make_dbuffer_at = [&](int loc, mem_t const& mem, dtype_t dtype) {
    return dbuffer_t(dtype, make_buffer_at(loc, mem));
  };

  vector<map<int, buffer_t>> caches(memgraph.num_cache_locs);
  set<int> groups_touched;

  for(int const& id: memgraph.get_order()) {
    auto const& op = memgraph.nodes[id].op;
    if(op.is_input()) {
      // nothing to do
    } else if(op.is_apply()) {
      auto const& apply = op.get_apply();
      auto const& [loc, mems, _, group] = apply;

      dbuffer_t out_buffer = make_dbuffer_at(loc, mems[0], apply.out_dtype());

      if(apply.is_einsummable()) {
        auto const& einsummable = apply.get_einsummable();

        vector<dbuffer_t> inn_buffers;
        for(int i = 1; i != mems.size(); ++i) {
          int which_arg = i-1;
          dtype_t inn_dtype = einsummable.inn_dtype(which_arg);

          inn_buffers.push_back(
            make_dbuffer_at(loc, mems[i], inn_dtype));
        }

        reference_einsummable_inplace(einsummable, out_buffer, inn_buffers);
      } else if(apply.is_touch()) {
        touch_t touch = apply.get_touch();

        vector<dbuffer_t> inn_buffers;
        for(int i = 1; i != mems.size(); ++i) {
          inn_buffers.push_back(
            make_dbuffer_at(loc, mems[i], touch.dtype));
        }

        if(group < 0) {
          // in this case, there is only one touch to this output region
          // so this operation is a copy. Mark this by reseting the
          // optional castable in touch.
          touch.castable.reset();
        } else {
          if(groups_touched.count(group) == 0) {
            // this is the first touch, so it is a copy
            touch.castable.reset();
            groups_touched.insert(group);
          }
        }
        if(inn_buffers.size() != 1) {
          throw std::runtime_error("touch at ref: invalid mem buffers size");
        }
        reference_touch(touch, out_buffer, inn_buffers[0]);
      }
    } else if(op.is_move()) {
      auto const& move = op.get_move();

      auto const& [src_loc, src_offset] = move.src;
      auto const& [dst_loc, dst_offset] = move.dst;
      auto const& size = move.size;

      buffer_t src_buffer = make_buffer_at(src_loc, mem_t { src_offset, size });
      buffer_t dst_buffer = make_buffer_at(dst_loc, mem_t { dst_offset, size });

      std::copy(src_buffer->data, src_buffer->data + size, dst_buffer->data);
    } else if(op.is_evict()) {
      auto const& evict = op.get_evict();
      auto const& [loc, cache_id, offset, size] = evict;

      buffer_t cache_buffer = make_buffer(size);
      buffer_t loc_buffer = make_buffer_at(loc, mem_t { offset, size });

      std::copy(loc_buffer->data, loc_buffer->data + size, cache_buffer->data);

      if(caches[loc].count(cache_id) > 0) {
        throw std::runtime_error("duplicate cache id");
      }
      caches[loc].insert({cache_id, cache_buffer});
    } else if(op.is_load()) {
      auto const& load = op.get_load();
      auto const& [cache_id, loc, offset, size] = load;

      buffer_t loc_buffer = make_buffer_at(loc, mem_t { offset, size });
      buffer_t cache_buffer = caches[loc].at(cache_id);

      std::copy(cache_buffer->data, cache_buffer->data + size, loc_buffer->data);

      caches[loc].erase(cache_id);
    } else if(op.is_partialize()) {
      // nothing to do
    } else if(op.is_del()) {
      // nothing to do
    } else {
      throw std::runtime_error("reference_compute_memgraph: should not happen");
    }
  }
}

tensor_t<dbuffer_t> partition_buffer(
  partition_t const& partition,
  dbuffer_t const& inn)
{
  vector<int> block_shape = partition.block_shape();
  vector<uint64_t> inn_shape = partition.total_shape();

  tensor_t<dbuffer_t> ret(block_shape);

  vector<int> block_index(block_shape.size(), 0);

  do {
    auto hrect = partition.get_hrect(block_index);
    auto offset = vector_mapfst(hrect);

    dbuffer_t& buffer = ret.at(block_index);
    vector<uint64_t> buffer_shape = hrect_shape(hrect);
    buffer = make_dbuffer(inn.dtype, product(buffer_shape));

    vector<uint64_t> inn_index = offset;
    do {
      vector<uint64_t> buffer_index = vector_sub(inn_index, offset);
      int out_idx = indexer_utils<uint64_t>::idxs_to_index(buffer_shape, buffer_index);
      int inn_idx = indexer_utils<uint64_t>::idxs_to_index(inn_shape, inn_index);
      buffer.set(out_idx, inn.get(inn_idx));
    } while(indexer_utils<uint64_t>::increment_idxs_region(hrect, inn_index));
  } while(increment_idxs(block_shape, block_index));

  return ret;
}

dbuffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<dbuffer_t> const& inn)
{
  dtype_t dtype;
  {
    auto const& ds = inn.get();
    if(ds.size() == 0) {
      throw std::runtime_error("should not be empty");
    }
    dtype = ds[0].dtype;
    for(int i = 1; i != ds.size(); ++i) {
      _assert_correct_dtype("unpartition_buffer", dtype, ds[i]);
    }
  }

  vector<int> block_shape = partition.block_shape();
  vector<uint64_t> out_shape = partition.total_shape();

  dbuffer_t out_buffer = make_dbuffer(dtype, product(out_shape));

  vector<int> block_index(block_shape.size(), 0);
  do {
    auto hrect = partition.get_hrect(block_index);
    auto offset = vector_mapfst(hrect);

    dbuffer_t const& inn_buffer = inn.at(block_index);
    vector<uint64_t> inn_shape = hrect_shape(hrect);

    vector<uint64_t> out_index = offset;
    do {
      vector<uint64_t> inn_index = vector_sub(out_index, offset);
      int inn_idx = indexer_utils<uint64_t>::idxs_to_index(inn_shape, inn_index);
      int out_idx = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);

      out_buffer.set(out_idx, inn_buffer.get(inn_idx));
    } while(indexer_utils<uint64_t>::increment_idxs_region(hrect, out_index));
  } while(increment_idxs(block_shape, block_index));

  return out_buffer;
}

dbuffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<dbuffer_t> const& inputs)
{
  dbuffer_t ret = make_dbuffer(einsummable.out_dtype(), einsummable.out_nelem());

  reference_einsummable_inplace(einsummable, ret, inputs);

  return ret;
}

void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  dbuffer_t ret,
  vector<dbuffer_t> const& inputs)
{
  auto const& join_shape = einsummable.join_shape;
  auto mid = join_shape.begin() + einsummable.out_rank;
  vector<uint64_t> out_shape(join_shape.begin(), mid);
  vector<uint64_t> agg_shape(mid, join_shape.end());

  vector<uint64_t> out_index(out_shape.size(), 0);
  vector<uint64_t> agg_index(agg_shape.size(), 0);

  auto inn_shapes = einsummable.inn_shapes();

  // check the inputs are as expected
  if(inn_shapes.size() != inputs.size()) {
    throw std::runtime_error("invalid input to reference einsummable");
  }
  for(int i = 0; i != inputs.size(); ++i) {
    auto const& dbuffer = inputs[i];
    auto const& shape  = inn_shapes[i];
    if(dbuffer.nelem() != product(shape)) {
      throw std::runtime_error("incorrect number of elem to reference einsummable");
    }
    dtype_t expected_dtype = einsummable.inn_dtype(i);
    _assert_correct_dtype("einsummable_inplace", expected_dtype, dbuffer);
  }

  auto inn_dtypes = einsummable.inn_dtypes();

  auto get_inputs = [&](vector<uint64_t> const& join_index) {
    vector<scalar_t> inns;
    inns.reserve(inn_shapes.size());
    for(int i = 0; i != inputs.size(); ++i) {
      dbuffer_t const& dbuffer = inputs[i];
      auto const& shape = inn_shapes[i];
      auto inn_index = einsummable.get_input_from_join(join_index, i);
      int which = indexer_utils<uint64_t>::idxs_to_index(shape, inn_index);
      inns.push_back(dbuffer.get(which));
    }
    return inns;
  };

  auto const& join_op = einsummable.join;

  // do the initialization
  do {
    auto join_index = vector_concatenate(out_index, agg_index);
    auto out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    vector<scalar_t> inn_scalars = get_inputs(join_index);
    ret.set(out_i, join_op.eval(inn_scalars));
  } while(indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));

  if(agg_shape.size() == 0) {
    // nothing else to do
    return;
  }

  castable_t const& castable = einsummable.castable.value();

  out_index = vector<uint64_t>(out_shape.size(), 0);
  do {
    int out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    // At this point, agg_index should be all zeros.
    // Increment it once since the zero case was done in the initialization.
    // Then proceed.
    while(indexer_utils<uint64_t>::increment_idxs(agg_shape, agg_index)) {
      auto join_index = vector_concatenate(out_index, agg_index);
      vector<scalar_t> inn_scalars = get_inputs(join_index);
      ret.agg_into(out_i, castable, join_op.eval(inn_scalars));
    }
    agg_index = vector<uint64_t>(agg_shape.size(), 0);
  } while (indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));
}

dbuffer_t reference_concat(
  concat_t const& concat,
  vector<dbuffer_t> const& inns)
{
  if(inns.size() != concat.inn_shapes.size()) {
    throw std::runtime_error("incorrect number of inputs");
  }
  for(auto const& inn: inns) {
    _assert_correct_dtype("reference_concat", concat.dtype, inn);
  }

  vector<uint64_t> out_shape = concat.shape();

  vector<uint64_t> offsets = concat.get_offsets();

  dbuffer_t ret = make_dbuffer(concat.dtype, product(out_shape));

  for(int i = 0; i != inns.size(); ++i) {
    dbuffer_t const& inn = inns[i];
    vector<uint64_t> const& inn_shape = concat.inn_shapes[i];

    if(inn.nelem() != product(inn_shape)) {
      throw std::runtime_error("incorrectly sized input");
    }

    uint64_t offset = offsets[i];

    vector<tuple<uint64_t, uint64_t>> hrect = concat.get_hrect(i);
    auto out_index = vector_mapfst(hrect);
    do {
      vector<uint64_t> inn_index = out_index;
      inn_index[concat.dim] -= offset;

      auto out_idx = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
      auto inn_idx = indexer_utils<uint64_t>::idxs_to_index(inn_shape, inn_index);

      ret.set(out_idx, inn.get(inn_idx));
    } while(indexer_utils<uint64_t>::increment_idxs_region(hrect, out_index));
  }

  return ret;
}

template <typename T>
std::function<void(T&, T const&)>
reference_touch_update(optional<castable_t> const& c)
{
  if(c) {
    auto const& op = c.value();

    if(op == castable_t::add) {
      return [](T& v, T a) {
        v += a;
      };
    } else if(op == castable_t::mul) {
      return [](T& v, T a) {
        v *= a;
      };
    } else if(op == castable_t::min) {
      return [](T& v, T a) {
        v = std::min(v,a);
      };
    } else if(op == castable_t::max) {
      return [](T& v, T a) {
        v = std::max(v,a);
      };
    } else {
      throw std::runtime_error("ref touch update not implemented");
    }
  } else {
    return [](T& out, T const& inn) {
      out = inn;
    };
  }
}

template <>
std::function<void(std::complex<float>&, std::complex<float> const&)>
reference_touch_update(optional<castable_t> const& c)
{
  using T = std::complex<float>;

  if(c) {
    auto const& op = c.value();

    if(op == castable_t::add) {
      return [](T& v, T a) {
        v += a;
      };
    } else if(op == castable_t::mul) {
      return [](T& v, T a) {
        v *= a;
      };
    } else {
      throw std::runtime_error("ref touch update complex not implemented");
    }
  } else {
    return [](T& out, T const& inn) {
      out = inn;
    };
  }
}

template <typename T>
void _reference_touch(
  touch_t const& touch,
  T* out,
  T const* inn)
{
  vector<uint64_t> shape = vector_from_each_member(
    touch.selection, uint64_t, size);
  vector<uint64_t> shape_inn = vector_from_each_member(
    touch.selection, uint64_t, d_inn);
  vector<uint64_t> shape_out = vector_from_each_member(
    touch.selection, uint64_t, d_out);
  vector<uint64_t> offset_inn = vector_from_each_member(
    touch.selection, uint64_t, offset_inn);
  vector<uint64_t> offset_out = vector_from_each_member(
    touch.selection, uint64_t, offset_out);

  vector<uint64_t> index(shape.size(), 0);

  vector<uint64_t> index_inn(shape.size());
  vector<uint64_t> index_out(shape.size());

  auto update = reference_touch_update<T>(touch.castable);

  do {
    index_inn = vector_add(index, offset_inn);
    int idx_inn = indexer_utils<uint64_t>::idxs_to_index(shape_inn, index_inn);

    index_out = vector_add(index, offset_out);
    int idx_out = indexer_utils<uint64_t>::idxs_to_index(shape_out, index_out);

    update(out[idx_out], inn[idx_inn]);
  } while(indexer_utils<uint64_t>::increment_idxs(shape, index));
}

void reference_touch(
  touch_t const& touch,
  dbuffer_t out,
  dbuffer_t inn)
{
  auto const& dtype = touch.dtype;

  if(dtype == dtype_t::f16) {
    _reference_touch(touch, out.f16(), inn.f16());
  } else if(dtype == dtype_t::f32) {
    _reference_touch(touch, out.f32(), inn.f32());
  } else if(dtype == dtype_t::f64) {
    _reference_touch(touch, out.f64(), inn.f64());
  } else if(dtype == dtype_t::c64) {
    _reference_touch(touch, out.c64(), inn.c64());
  } else {
    throw std::runtime_error("should not reach fill");
  }
}

tensor_t<dbuffer_t> get_partitioned_buffer(
  map<int, dbuffer_t> items,
  tensor_t<int> whiches)
{
  vector<dbuffer_t> vec;
  vec.reserve(product(whiches.get_shape()));
  for(auto const& which: whiches.get()) {
    vec.push_back(items.at(which));
  }

  return tensor_t<dbuffer_t>(whiches.get_shape(), vec);
}

map<int, dbuffer_t> init_buffer_map(
  tensor_t<int> keys,
  tensor_t<dbuffer_t> values)
{
  map<int, dbuffer_t> ret;
  fill_buffer_map(ret, keys, values);
  return ret;
}

void fill_buffer_map(
  map<int, dbuffer_t>& items,
  tensor_t<int> keys,
  tensor_t<dbuffer_t> values)
{
  if(!vector_equal(keys.get_shape(), values.get_shape())) {
    throw std::runtime_error("invalid fill_buffer_map");
  }
  auto const& ks = keys.get();
  auto const& vs = values.get();
  for(int i = 0; i != ks.size(); ++i) {
    auto const& k = ks[i];
    auto const& v = vs[i];
    items.insert({k,v});
  }
}

