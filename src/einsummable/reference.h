#pragma once
#include "setup.h"

#include "tensor.h"
#include "graph.h"
#include "taskgraph.h"
#include "memgraph.h"

#include <memory>

struct buffer_holder_t {
  buffer_holder_t(uint64_t size): size(size) { data = new float[size]; }
  ~buffer_holder_t() { delete[] data; }

  void zeros() { std::fill(data, data + size, 0.0); }
  void ones()  { std::fill(data, data + size, 1.0); }
  void iota(int start = 0) { std::iota(data, data + size, start); }
  void random(float lower = 0.0, float upper = 1.0);

  float sum() const { return std::accumulate(data, data + size, 0.0); }

  vector<float> as_vector() const;

  uint64_t size;
  float* data;
};

using buffer_t = std::shared_ptr<buffer_holder_t>;

bool operator==(buffer_t const& lhs, buffer_t const& rhs);
bool operator!=(buffer_t const& lhs, buffer_t const& rhs);
bool operator==(buffer_holder_t const& lhs, buffer_holder_t const& rhs);
bool operator!=(buffer_holder_t const& lhs, buffer_holder_t const& rhs);

bool is_close(buffer_t const& lhs, buffer_t const& rhs, float eps = 1e-3);
bool is_close(buffer_holder_t const& lhs, buffer_holder_t const& rhs, float eps = 1e-3);
bool is_close(float lhs, float rhs, float eps = 1e-3);
bool is_close(
  buffer_t const& lhs, uint64_t offset_lhs,
  buffer_t const& rhs, uint64_t offset_rhs,
  uint64_t size,
  float eps = 1e-3);

map<int, buffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, buffer_t> const& inputs);

map<int, buffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, buffer_t> const& inputs);

void reference_compute_memgraph(
  memgraph_t const& memgraph,
  vector<buffer_t>& compute_location_buffers);

tensor_t<buffer_t> partition_buffer(
  partition_t const& partition,
  buffer_t const& buffer);

buffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<buffer_t> const& buffer);

buffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<buffer_t> const& inputs);

void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  buffer_t& out,
  vector<buffer_t> const& inputs);

void reference_touch(
  touch_t const& touch,
  buffer_t& out,
  buffer_t const& inn);

tensor_t<buffer_t> get_partitioned_buffer(
  map<int, buffer_t> items,
  tensor_t<int> whiches);

map<int, buffer_t> init_buffer_map(
  tensor_t<int> keys,
  tensor_t<buffer_t> values);

void fill_buffer_map(
  map<int, buffer_t>& items,
  tensor_t<int> keys,
  tensor_t<buffer_t> values);

std::ostream& operator<<(std::ostream& out, buffer_t const& buffer);

