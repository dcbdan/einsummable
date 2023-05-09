#include "reference.h"

buffer_t make_buffer(uint64_t size) {
  return std::make_shared<buffer_holder_t>(size);
}
buffer_t make_buffer_reference(float* data, uint64_t size) {
  return std::make_shared<buffer_holder_t>(data, size);
}

void buffer_holder_t::random(float lower, float upper) {
  std::uniform_real_distribution<float> dist(lower, upper);
  auto gen = [&dist]{ return dist(random_gen()); };
  std::generate(data, data+size, gen);
}

vector<float> buffer_holder_t::as_vector() const {
  vector<float> ret;
  ret.reserve(size);
  std::copy(data, data + size, std::back_inserter(ret));
  return ret;
}

bool operator==(buffer_t const& lhs, buffer_t const& rhs) {
  return *lhs == *rhs;
}
bool operator!=(buffer_t const& lhs, buffer_t const& rhs) {
  return !(lhs == rhs);
}
bool operator==(buffer_holder_t const& lhs, buffer_holder_t const& rhs) {
  if(lhs.size != rhs.size) {
    return false;
  }
  for(int i = 0; i != lhs.size; ++i) {
    if(lhs.data[i] != rhs.data[i]) {
      return false;
    }
  }
  return true;
}
bool operator!=(buffer_holder_t const& lhs, buffer_holder_t const& rhs) {
  return !(lhs == rhs);
}

bool is_close(buffer_t const& lhs, buffer_t const& rhs, float eps) {
  buffer_holder_t const& l = *lhs;
  buffer_holder_t const& r = *rhs;
  return is_close(l, r, eps);
}
bool is_close(buffer_holder_t const& lhs, buffer_holder_t const& rhs, float eps) {
  if(lhs.size != rhs.size) {
    return false;
  }
  for(int i = 0; i != lhs.size; ++i) {
    if(!is_close(lhs.data[i], rhs.data[i], eps)) {
      return false;
    }
  }
  return true;
}
bool is_close(float lhs, float rhs, float eps) {
  return (lhs <= rhs + eps) && (lhs >= rhs - eps);
}
bool is_close(
  buffer_t const& lhs, uint64_t offset_lhs,
  buffer_t const& rhs, uint64_t offset_rhs,
  uint64_t size,
  float eps)
{
  if(lhs->size < offset_lhs + size) {
    return false;
  }
  if(rhs->size < offset_rhs + size) {
    return false;
  }
  float* raw_lhs = lhs->data + offset_lhs;
  float* raw_rhs = rhs->data + offset_rhs;
  for(int i = 0; i != size; ++i) {
    if(!is_close(raw_lhs[i], raw_rhs[i], eps)) {
      return false;
    }
  }
  return true;
}

map<int, buffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, buffer_t> const& inputs)
{
  map<int, buffer_t> outs;
  map<int, buffer_t> tensors;

  for(int id = 0; id != graph.nodes.size(); ++id)
  {
    auto const& node = graph.nodes[id];

    if(node.op.is_formation()) {
      tensors[id] = tensors[node.inns[0]];
    } else if(node.op.is_input()) {
      tensors[id] = inputs.at(id);
    } else if(node.op.is_einsummable()) {
      vector<buffer_t> inns;
      inns.reserve(node.inns.size());
      for(auto const& id_inn: node.inns) {
        inns.push_back(tensors[id_inn]);
      }
      tensors[id] = reference_einsummable(node.op.get_einsummable(), inns);
    } else {
      throw std::runtime_error("should not reach: reference compute graph");
    }

    if(node.op.is_save()) {
      outs[id] = tensors[id];
    }
  }

  return outs;
}

map<int, buffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, buffer_t> const& inputs)
{
  map<int, buffer_t> outs;

  // id -> (location, buffer)
  map<int, tuple<int, buffer_t> > tensors;
  auto get_at = [&tensors](int id, int loc) {
    auto const& [actual_loc, buffer] = tensors.at(id);
    if(loc != actual_loc) {
      throw std::runtime_error("incorrect locs in taskgraph");
    }
    return buffer;
  };

  for(auto const& id: taskgraph.get_order())
  {
    auto const& node = taskgraph.nodes[id];

    if(node.op.is_input()) {
      tensors[id] = {node.op.output_loc(), inputs.at(id)};
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();

      vector<buffer_t> inputs;
      inputs.reserve(apply.inns.size());
      for(auto const& inn: apply.inns) {
        inputs.push_back(get_at(inn, apply.loc));
      }

      tensors[id] = {
        apply.loc,
        reference_einsummable(apply.einsummable,inputs)
      };
    } else if(node.op.is_move()) {
      auto const& move = node.op.get_move();
      buffer_t buffer_src = get_at(move.inn, move.src);

      buffer_t buffer_dst = make_buffer(buffer_src->size);
      std::copy(
        buffer_src->data,
        buffer_src->data + buffer_src->size,
        buffer_dst->data);

      tensors[id] = std::make_tuple(move.dst, buffer_dst);
    } else if(node.op.is_partialize()) {
      int loc = node.op.output_loc();

      buffer_t write = make_buffer(node.op.tensor_size());
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
            .castable = std::optional<castable_t>()
          },
          write,
          get_at(inn0, loc));

        for(int i = 1; i < ts.size(); ++i) {
          auto const& [inn, t] = ts[i];
          reference_touch(t, write, get_at(inn, loc));
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
  for(int const& id: memgraph.get_order()) {
    auto const& op = memgraph.nodes[id].op;
    if(op.is_input()) {
      // nothing to do
    } else if(op.is_apply()) {
      //auto const& [loc, mems, op, group] = op.get_apply();

    } else if(op.is_move()) {
      // TODO
    } else if(op.is_evict()) {
      // TODO
    } else if(op.is_load()) {
      // TODO
    } else if(op.is_del()) {
      // nothing to do
    } else {
      throw std::runtime_error("reference_compute_memgraph: should not happen");
    }
  }
}

tensor_t<buffer_t> partition_buffer(
  partition_t const& partition,
  buffer_t const& inn)
{
  vector<int> block_shape = partition.block_shape();
  vector<uint64_t> inn_shape = partition.total_shape();

  tensor_t<buffer_t> ret(block_shape);

  vector<int> block_index(block_shape.size(), 0);

  do {
    auto hrect = partition.get_hrect(block_index);
    auto offset = vector_mapfst(hrect);

    buffer_t& buffer = ret.at(block_index);
    vector<uint64_t> buffer_shape = shape_hrect(hrect);
    buffer = make_buffer(product(buffer_shape));

    vector<uint64_t> inn_index = offset;
    do {
      vector<uint64_t> buffer_index = vector_sub(inn_index, offset);
      int out_idx = indexer_utils<uint64_t>::idxs_to_index(buffer_shape, buffer_index);
      int inn_idx = indexer_utils<uint64_t>::idxs_to_index(inn_shape, inn_index);
      buffer->data[out_idx] = inn->data[inn_idx];
    } while(indexer_utils<uint64_t>::increment_idxs_region(hrect, inn_index));
  } while(increment_idxs(block_shape, block_index));

  return ret;
}

buffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<buffer_t> const& inn)
{
  vector<int> block_shape = partition.block_shape();
  vector<uint64_t> out_shape = partition.total_shape();

  buffer_t out_buffer = make_buffer(product(out_shape));

  vector<int> block_index(block_shape.size(), 0);
  do {
    auto hrect = partition.get_hrect(block_index);
    auto offset = vector_mapfst(hrect);

    buffer_t const& inn_buffer = inn.at(block_index);
    vector<uint64_t> inn_shape = shape_hrect(hrect);

    vector<uint64_t> out_index = offset;
    do {
      vector<uint64_t> inn_index = vector_sub(out_index, offset);
      int inn_idx = indexer_utils<uint64_t>::idxs_to_index(inn_shape, inn_index);
      int out_idx = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);

      out_buffer->data[out_idx] = inn_buffer->data[inn_idx];
    } while(indexer_utils<uint64_t>::increment_idxs_region(hrect, out_index));
  } while(increment_idxs(block_shape, block_index));

  return out_buffer;
}

std::function<void(float&, float)> einsummable_update(castable_t op)
{
  if(op == castable_t::add) {
    return [](float& v, float a) {
      v += a;
    };
  } else if(op == castable_t::mul) {
    return [](float& v, float a) {
      v *= a;
    };
  } else if(op == castable_t::min) {
    return [](float& v, float a) {
      v = std::min(v,a);
    };
  } else if(op == castable_t::max) {
    return [](float& v, float a) {
      v = std::max(v,a);
    };
  } else {
    throw std::runtime_error("einsummable_update not implemented");
  }
}

buffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<buffer_t> const& inputs)
{
  buffer_t ret = make_buffer(product(einsummable.out_shape()));

  reference_einsummable_inplace(einsummable, ret, inputs);

  return ret;
}

void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  buffer_t& ret,
  vector<buffer_t> const& inputs)
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
    auto const& buffer = inputs[i];
    auto const& shape  = inn_shapes[i];
    if(buffer->size != product(shape)) {
      throw std::runtime_error("incorrect input size to reference einsummable");
    }
  }

  auto get_inputs = [&](vector<uint64_t> const& join_index) {
    vector<float> inns;
    inns.reserve(inn_shapes.size());
    for(int i = 0; i != inn_shapes.size(); ++i) {
      auto const& buffer = inputs[i];
      auto const& shape = inn_shapes[i];
      auto inn_index = einsummable.get_input_from_join(join_index, i);
      int which = indexer_utils<uint64_t>::idxs_to_index(shape, inn_index);
      inns.push_back(buffer->data[which]);
    }
    return inns;
  };

  scalarop_t const& joinop = einsummable.join;
  auto eval = [&joinop](vector<float> const& xs){
    return joinop.eval(xs);
  };

  // do the initialization
  do {
    auto join_index = vector_concatenate(out_index, agg_index);
    auto out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    ret->data[out_i] = eval(get_inputs(join_index));
  } while(indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));

  if(agg_shape.size() == 0) {
    // nothing else to do
    return;
  }

  auto update = einsummable_update(einsummable.castable.value());

  out_index = vector<uint64_t>(out_shape.size(), 0);
  do {
    int out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    // At this point, agg_index should be all zeros.
    // Increment it once since the zero case was done in the initialization.
    // Then proceed.
    while(indexer_utils<uint64_t>::increment_idxs(agg_shape, agg_index)) {
      auto join_index = vector_concatenate(out_index, agg_index);
      update(ret->data[out_i], eval(get_inputs(join_index)));
    }
    agg_index = vector<uint64_t>(agg_shape.size(), 0);
  } while (indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));
}

std::function<void(float&, float const&)> reference_touch_update(
  std::optional<castable_t> castable)
{
  if(castable) {
    return einsummable_update(castable.value());
  } else {
    return [](float& val, float const& inn) {
      val = inn;
    };
  }
}

void reference_touch(
  touch_t const& touch,
  buffer_t& out,
  buffer_t const& inn)
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

  auto update = reference_touch_update(touch.castable);

  do {
    index_inn = vector_add(index, offset_inn);
    int idx_inn = indexer_utils<uint64_t>::idxs_to_index(shape_inn, index_inn);

    index_out = vector_add(index, offset_out);
    int idx_out = indexer_utils<uint64_t>::idxs_to_index(shape_out, index_out);

    update(out->data[idx_out], inn->data[idx_inn]);
  } while(indexer_utils<uint64_t>::increment_idxs(shape, index));
}

tensor_t<buffer_t> get_partitioned_buffer(
  map<int, buffer_t> items,
  tensor_t<int> whiches)
{
  vector<buffer_t> vec;
  vec.reserve(product(whiches.get_shape()));
  for(auto const& which: whiches.get()) {
    vec.push_back(items.at(which));
  }

  return tensor_t<buffer_t>(whiches.get_shape(), vec);
}

map<int, buffer_t> init_buffer_map(
  tensor_t<int> keys,
  tensor_t<buffer_t> values)
{
  map<int, buffer_t> ret;
  fill_buffer_map(ret, keys, values);
  return ret;
}

void fill_buffer_map(
  map<int, buffer_t>& items,
  tensor_t<int> keys,
  tensor_t<buffer_t> values)
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

std::ostream& operator<<(std::ostream& out, buffer_t const& buffer)
{
  out << "buffer[" << buffer->size << "]{";
  if(buffer->size > 0) {
    out << buffer->data[0];
    for(uint64_t i = 1; i != buffer->size; ++i) {
      out << "," << buffer->data[i];
    }
  }
  out << "}";
  return out;
}


