#pragma once
#include "../../einsummable/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" // touch_t

#include <thread>

void print_elementwise_function(scalarop_t op);

std::function<void(float*,vector<float const*>)>
build_unary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t unary_op);

std::function<void(float*,vector<float const*>)>
build_binary_elementwise_kernel(
  int num_threads,
  uint64_t n,
  scalarop_t binary_op);

std::function<void(float*, float const*)>
build_touch(touch_t const& touch);
