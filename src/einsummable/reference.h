#pragma once
#include "../base/setup.h"

#include "dbuffer.h"
#include "graph.h"
#include "taskgraph.h"
#include "memgraph.h"

map<int, dbuffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, dbuffer_t> const& inputs);

map<int, buffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, buffer_t> const& inputs);

void reference_compute_memgraph(
  memgraph_t const& memgraph,
  vector<buffer_t>& compute_location_buffers,
  vector<map<int, buffer_t>>& storages);

void reference_compute_memgraph(
  memgraph_t const& memgraph,
  vector<buffer_t>& compute_location_buffers);

vtensor_t<dbuffer_t> partition_buffer(
  partition_t const& partition,
  dbuffer_t const& buffer);

dbuffer_t unpartition_buffer(
  partition_t const& partition,
  vtensor_t<dbuffer_t> const& buffer);

dbuffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<dbuffer_t> const& inputs);

void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  dbuffer_t out,
  vector<dbuffer_t> const& inputs);

dbuffer_t reference_fill(fill_t const& fill);

dbuffer_t reference_select(
  select_t const& select,
  vector<dbuffer_t> const& inputs);

void reference_touch(
  touch_t const& touch,
  dbuffer_t out,
  dbuffer_t const inn);

vtensor_t<dbuffer_t> get_partitioned_buffer(
  map<int, dbuffer_t> items,
  vtensor_t<int> whiches);

map<int, dbuffer_t> init_buffer_map(
  vtensor_t<int> keys,
  vtensor_t<dbuffer_t> values);

void fill_buffer_map(
  map<int, dbuffer_t>& items,
  vtensor_t<int> keys,
  vtensor_t<dbuffer_t> values);

map<int, dbuffer_t> to_typed_buffer_map(
  map<int, buffer_t> const& bs,
  map<int, dtype_t> to_dtypes);

map<int, buffer_t> to_untyped_buffer_map(
  map<int, dbuffer_t> const& dbs);

map<int, dtype_t> typed_task_ids(
  graph_t const& graph,
  map<int, vtensor_t<int>> const& gid_to_tids);

map<int, dbuffer_t>
typed_reference_compute_taskgraph_from_graph_info(
  taskgraph_t const& taskgraph,
  map<int, dbuffer_t> const& inputs,
  graph_t const& graph,
  map<int, vtensor_t<int>> const& save_gid_to_tids);

