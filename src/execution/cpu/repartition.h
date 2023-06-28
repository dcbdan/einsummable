#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"
#include "../../base/placement.h"
#include "../../base/vtensor.h"
#include "../../einsummable/scalarop.h"
#include "../../einsummable/taskgraph.h"

#include "mpi_class.h"

vtensor_t<optional<buffer_t>>
repartition(
  mpi_t* mpi,
  dtype_t dtype,
  placement_t const& placement,
  buffer_t data);

vtensor_t<optional<buffer_t>>
repartition(
  mpi_t* mpi,
  dtype_t dtype,
  placement_t const& out_placement,
  vtensor_t<optional<buffer_t>> const& data,
  placement_t const& inn_placement);

vtensor_t<buffer_t>
repartition(
  dtype_t dtype,
  partition_t const& out_partition,
  vtensor_t<buffer_t> const& data,
  partition_t const& inn_partition);

vtensor_t<buffer_t>
repartition(
  dtype_t dtype,
  partition_t const& partition,
  buffer_t data);

tuple<
  vtensor_t<int>, // out
  vtensor_t<int>, // inn
  taskgraph_t>
make_repartition(
  dtype_t dtype,
  placement_t const& out_placement,
  placement_t const& inn_placement);
