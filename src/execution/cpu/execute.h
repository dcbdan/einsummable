#pragma once
#include "../../einsummable/setup.h"

#include "../../einsummable/taskgraph.h"
#include "../../einsummable/reference.h" // buffer_t

// Every input node in taskgraph should be in tensors.
// After execution, only every save taskgraph node should be in tensors
void execute(taskgraph_t const& taskgraph, map<int, buffer_t>& tensors);

