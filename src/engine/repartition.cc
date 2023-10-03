#include "repartition.h"

void repartition(
  communicator_t& comm,
  remap_relations_t const& remap,
  map<int, buffer_t>& data)
{
  // 1. form a taskgraph
  // 2. remap data from inn_r_tid to inn_tg_tid
  // 3. build exec data with data
  // 4. execute the exec graph
  // 5. remap data from out_tg_tid to out_r_tid

  throw std::runtime_error("not implemented yet");
}

// Issue 1: yet more things into the exec graph
//   Solution 1: make std unique pointers and inherit from a virtual base_op
//   ^ TODO: Make this change
//   Solution 2: pass in a pair of functions and some "data" shared between the
//               two functions that is also managed by the node
// Issue 2: yet more things into the resource manager
//   Note: a resource is a list of resource units
//   Note: a (desc unit type, resource unit type, manager unit type) all come
//         together

