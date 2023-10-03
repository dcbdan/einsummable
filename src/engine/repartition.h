#pragma once
#include "../base/setup.h"

#include "../base/buffer.h"
#include "communicator.h"
#include "../einsummable/relation.h"

void repartition(
  communicator_t& comm,
  remap_relations_t const& remap,
  map<int, buffer_t>& data);
