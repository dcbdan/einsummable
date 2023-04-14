#include "reference.h"

map<int, buffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, buffer_t> const& inputs)
{
  // TODO
  return {};
}

map<int, buffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, buffer_t> const& inputs)
{
  // TODO
  return {};
}

tensor_t<buffer_t> partition_buffer(
  partition_t const& partition,
  buffer_t const& buffer)
{
  // TODO
  return {};
}

buffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<buffer_t> const& buffer)
{
  // TODO
  return nullptr;
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

