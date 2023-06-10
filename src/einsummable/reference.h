#pragma once
#include "../base/setup.h"

#include "../base/tensor.h"

#include "dbuffer.h"
#include "graph.h"
#include "taskgraph.h"
#include "memgraph.h"

map<int, dbuffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, dbuffer_t> const& inputs);

map<int, dbuffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, dbuffer_t> const& inputs);

void reference_compute_memgraph(
  memgraph_t const& memgraph,
  vector<buffer_t>& compute_location_buffers);

tensor_t<dbuffer_t> partition_buffer(
  partition_t const& partition,
  dbuffer_t const& buffer);

dbuffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<dbuffer_t> const& buffer);

dbuffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<dbuffer_t> const& inputs);

void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  dbuffer_t out,
  vector<dbuffer_t> const& inputs);

dbuffer_t reference_concat(
  concat_t const& concat,
  vector<dbuffer_t> const& inputs);

void reference_touch(
  touch_t const& touch,
  dbuffer_t out,
  dbuffer_t const inn);

tensor_t<dbuffer_t> get_partitioned_buffer(
  map<int, dbuffer_t> items,
  tensor_t<int> whiches);

map<int, dbuffer_t> init_buffer_map(
  tensor_t<int> keys,
  tensor_t<dbuffer_t> values);

void fill_buffer_map(
  map<int, dbuffer_t>& items,
  tensor_t<int> keys,
  tensor_t<dbuffer_t> values);

