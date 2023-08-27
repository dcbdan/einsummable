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
// Note: remap must be the same across all nodes

// These server, client pair functions are also do a repartition, but
// by creating a taskgraph & updating the tid mapping in data_locs
// accordingly
void repartition_server(
  mpi_t* mpi,
  remap_relations_t const& remap,
  allocator_settings_t const& alloc_settings,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage);

void repartition_client(
  mpi_t* mpi,
  int server_rank,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage);

