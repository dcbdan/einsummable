#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

#include "../../einsummable/relation.h"
#include "../../einsummable/memgraph.h"

#include "mpi_class.h"
#include "storage.h"

// This should by called by each rank with the same remap object.
// The data, across all ranks, should initially be distributed
// according to the remap input set. On completion, the data should
// contain the remap output set.
//
// To run the computation, a taskgraph is created and executed.
void repartition(
  mpi_t* mpi,
  remap_relations_t const& remap,
  map<int, buffer_t>& data);

// This is the same as the other function except instead the data
// is distributed across mem and storage objects. data_locs contains
// a mapping from ids to where on this node data currently is.
//
// To run the computation, a memgraph is created and executed.
//
// Assumption: for all i, storage loc i == compute loc i
void repartition(
  mpi_t* mpi,
  remap_relations_t const& remap,
  allocator_settings_t const& alloc_settings,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage);

