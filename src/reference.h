#pragma once
#include "setup.h"

#include "tensor.h"
#include "graph.h"
#include "taskgraph.h"

#include <memory>

struct buffer_holder_t {
  buffer_holder_t(uint64_t size): size(size) { data = new float[size]; }
  ~buffer_holder_t() { delete[] data; }

  void zeros() { std::fill(data, data + size, 0.0); }
  void ones()  { std::fill(data, data + size, 1.0); }
  void iota(int start = 0) { std::iota(data, data + size, start); }

  vector<float> as_vector() const {
    vector<float> ret;
    ret.reserve(size);
    std::copy(data, data + size, std::back_inserter(ret));
    return ret;
  }

  uint64_t size;
  float* data;
};

using buffer_t = std::shared_ptr<buffer_holder_t>;

map<int, buffer_t> reference_compute_graph(
  graph_t const& graph,
  map<int, buffer_t> const& inputs);

map<int, buffer_t> reference_compute_taskgraph(
  taskgraph_t const& taskgraph,
  map<int, buffer_t> const& inputs);

tensor_t<buffer_t> partition_buffer(
  partition_t const& partition,
  buffer_t const& buffer);

buffer_t unpartition_buffer(
  partition_t const& partition,
  tensor_t<buffer_t> const& buffer);

buffer_t reference_einsummable(
  einsummable_t const& einsummable,
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

