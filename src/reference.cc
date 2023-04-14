#include "reference.h"

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

  for(int id = 0; id != taskgraph.nodes.size(); ++id)
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
      tensors[id] = std::make_tuple(
        move.dst,
        get_at(move.inn, move.src));
    } else if(node.op.is_partialize()) {
      int loc = node.op.output_loc();

      buffer_t write = std::make_shared<buffer_holder_t>(node.op.tensor_size());
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
    buffer = std::make_shared<buffer_holder_t>(product(buffer_shape));

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

  buffer_t out_buffer = std::make_shared<buffer_holder_t>(product(out_shape));

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

std::function<float(vector<float> const&)> einsummable_eval(scalar_join_t op)
{
  if(op == scalar_join_t::add) {
    return [](vector<float> const& inns) {
      return inns[0] + inns[1];
    };
  } else if(op == scalar_join_t::sub)    {
    return [](vector<float> const& inns) {
      return inns[0] - inns[1];
    };
  } else if(op == scalar_join_t::mul)    {
    return [](vector<float> const& inns) {
      return inns[0] * inns[1];
    };
  } else if(op == scalar_join_t::relu)   {
    return [](vector<float> const& inns) {
      return inns[0] > 0 ? inns[0] : 0.0;
    };
  } else if(op == scalar_join_t::negate) {
    return [](vector<float> const& inns) {
      return -1*inns[0];
    };
  } else if(op == scalar_join_t::min)    {
    return [](vector<float> const& inns) {
      return std::min(inns[0], inns[1]);
    };
  } else if(op == scalar_join_t::max)    {
    return [](vector<float> const& inns) {
      return std::max(inns[0], inns[1]);
    };
  } else {
    throw std::runtime_error("einsummable_eval not implemented");
  }
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
  buffer_t ret = std::make_shared<buffer_holder_t>(product(einsummable.out_shape()));

  auto const& join_shape = einsummable.join_shape;
  auto mid = join_shape.begin() + einsummable.out_rank;
  vector<uint64_t> out_shape(join_shape.begin(), mid);
  vector<uint64_t> agg_shape(mid, join_shape.end());

  vector<uint64_t> out_index(out_shape.size(), 0);
  vector<uint64_t> agg_index(agg_shape.size(), 0);

  auto inn_shapes = einsummable.inn_shapes();

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

  auto eval = einsummable_eval(einsummable.join);
  auto update = einsummable_update(einsummable.castable);

  // do the initialization
  do {
    auto join_index = vector_concatenate(out_index, agg_index);
    auto out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    ret->data[out_i] = eval(get_inputs(join_index));
  } while(indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));

  out_index = vector<uint64_t>(out_shape.size(), 0);
  do {
    int out_i = indexer_utils<uint64_t>::idxs_to_index(out_shape, out_index);
    while(indexer_utils<uint64_t>::increment_idxs(agg_shape, agg_index)) {
      auto join_index = vector_concatenate(out_index, agg_index);
      update(ret->data[out_i], eval(get_inputs(join_index)));
    }
  } while (indexer_utils<uint64_t>::increment_idxs(out_shape, out_index));

  return ret;
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

