#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

#include "../../einsummable/relation.h"

#include "mpi_class.h"

void repartition(
  mpi_t* mpi,
  remap_relations_t const& remap,
  map<int, buffer_t>& data);

