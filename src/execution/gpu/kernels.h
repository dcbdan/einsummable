#pragma once

#include "../../base/setup.h"

#include "../../einsummable/scalarop.h"

#include "../../einsummable/taskgraph.h" 

#include <cuda_runtime.h>

std::function<void(cudaStream_t, float*, float const*)>
build_touch(touch_t const& touch);