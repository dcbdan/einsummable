#include "repartition.h"

#include "execute.h"

#include "../../einsummable/graph.h"
#include "../../einsummable/taskgraph.h"

vtensor_t<optional<buffer_t>>
repartition(
  mpi_t* mpi,
  dtype_t dtype,
  placement_t const& out_placement,
  buffer_t data)
{
  placement_t inn_placement(partition_t::singleton(out_placement.total_shape()));
  vtensor_t<optional<buffer_t>> inn_data(inn_placement.block_shape());
  inn_data.get()[0] = data;
  return repartition(mpi, dtype, out_placement, inn_data, inn_placement);
}

// TODO: Calling execute may be overkill in some cases...
//       Doing this for every weight matrix might be slow because of the
//       overhead of launching and closing the execution engine over and over
//       (Or maybe not)
vtensor_t<optional<buffer_t>>
repartition(
  mpi_t* mpi,
  dtype_t dtype,
  placement_t const& out_placement,
  vtensor_t<optional<buffer_t>> const& data,
  placement_t const& inn_placement)
{
  auto [out_tids, inn_tids, taskgraph] =
    make_repartition(dtype, out_placement, inn_placement);

  int this_loc = bool(mpi) ? mpi->this_rank : 0;

  map<int, buffer_t> tensors;
  int n_blocks_inn = inn_placement.num_parts();
  for(int i = 0; i != n_blocks_inn; ++i) {
    int const& tid = inn_tids.get()[i];
    if(taskgraph.out_loc(tid) == this_loc) {
      buffer_t d = data.get()[i].value();
      tensors.insert({tid, d});
    }
  }

  settings_t settings = settings_t::only_touch_settings();
  kernel_manager_t ks;
  execute(taskgraph, settings, ks, mpi, tensors);

  vtensor_t<optional<buffer_t>> ret(out_placement.block_shape());

  int n_blocks_out = out_placement.num_parts();
  for(int i = 0; i != n_blocks_out; ++i) {
    int const& tid = out_tids.get()[i];
    if(taskgraph.out_loc(tid) == this_loc) {
      ret.get()[i] = tensors.at(tid);
    }
  }

  return ret;
}

vtensor_t<buffer_t>
repartition(
  dtype_t dtype,
  partition_t const& out_partition,
  vtensor_t<buffer_t> const& inn_data,
  partition_t const& inn_partition)
{
  if(out_partition == inn_partition) {
    return inn_data;
  }

  vtensor_t<optional<buffer_t>> maybe_inn_data(inn_partition.block_shape());
  std::copy(
    inn_data.get().begin(), inn_data.get().end(),
    maybe_inn_data.get().begin());
  auto maybe_out_data = repartition(
    nullptr,
    dtype,
    placement_t(out_partition),
    maybe_inn_data,
    placement_t(inn_partition));

  vtensor_t<buffer_t> out_data(out_partition.block_shape());
  std::transform(
    maybe_out_data.get().begin(), maybe_out_data.get().end(),
    out_data.get().begin(),
    [](optional<buffer_t> const& maybe) { return maybe.value(); });

  return out_data;
}

vtensor_t<buffer_t>
repartition(
  dtype_t dtype,
  partition_t const& inn_partition,
  buffer_t data)
{
  partition_t const& out_partition =
    partition_t::singleton(inn_partition.total_shape());

  vtensor_t<buffer_t> out_data(out_partition.block_shape(), {data});

  return repartition(
    dtype,
    inn_partition,
    out_data,
    out_partition);
}

tuple<
  vtensor_t<int>, // out
  vtensor_t<int>, // inn
  taskgraph_t>
make_repartition(
  dtype_t dtype,
  placement_t const& out_placement,
  placement_t const& inn_placement)
{
  graph_constructor_t g;

  int inn = g.insert_input(inn_placement, dtype);
  int out = g.insert_formation(out_placement, inn, true);

  auto [inn_g_to_t, out_g_to_t, tg] = taskgraph_t::make(g.graph, g.get_placements());

  return {
    out_g_to_t.at(out),
    inn_g_to_t.at(inn),
    tg
  };
}
