#pragma once
#include "../base/setup.h"

#include "../base/tensor.h"
#include "../base/buffer.h"

#include "graph.h"
#include "taskgraph.h"
#include "memgraph.h"

struct dbuffer_t {
  dbuffer_t();
  dbuffer_t(dtype_t, buffer_t);

  void zeros();
  void ones();
  void fill(scalar_t val);
  void iota(int start = 0);
  void random();
  void random(scalar_t lower, scalar_t upper);

  dbuffer_t view_c64_as_f32();
  dbuffer_t view_f32_as_c64();

  scalar_t sum() const;

  uint64_t nelem() const;

  void set(uint64_t which_elem, scalar_t const& val);
  void agg_into(uint64_t which_elem, castable_t, scalar_t const& val);
  scalar_t get(uint64_t which_elem) const;

  float16_t          * f16();
  float              * f32();
  double             * f64();
  std::complex<float>* c64();

  float16_t           const* f16() const;
  float               const* f32() const;
  double              const* f64() const;
  std::complex<float> const* c64() const;

  dtype_t dtype;
  buffer_t data;
};

dbuffer_t make_dbuffer(dtype_t, uint64_t num_elems);

template <typename T>
bool is_close(T const& lhs, T const& rhs, float eps = 1e-3) {
  return (lhs <= rhs + T(eps)) && (lhs >= rhs - T(eps));
}

bool is_close(dbuffer_t const& lhs, dbuffer_t const& rhs, float eps = 1e-3);

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
dbuffer_t reference_einsummable(
  einsummable_t const& einsummable,
  vector<dbuffer_t> const& inputs);

// TODO
void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  buffer_t& out,
  vector<buffer_t> const& inputs);
void reference_einsummable_inplace(
  einsummable_t const& einsummable,
  dbuffer_t& out,
  vector<dbuffer_t> const& inputs);

// TODO
buffer_t reference_concat(
  concat_t const& concat,
  vector<buffer_t> const& inputs);
dbuffer_t reference_concat(
  concat_t const& concat,
  vector<dbuffer_t> const& inputs);

void reference_touch(
  touch_t const& touch,
  buffer_t out,
  buffer_t const inn);
void reference_touch(
  touch_t const& touch,
  dbuffer_t out,
  dbuffer_t const inn);

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

