#pragma once

#include "../../einsummable/memgraph.h"
#include "gpu_kernel_manager.h"
#include "utility.h"

// 1. assume we have a kernel manager for each gpu
// 2. ignore storage for now
// 3. for workspace, assume that we can use maximum output size
// 4. for groups: use conditional nodes and handles

// What do we know about cudagraphs?
// * semaphores: unsure what they are for or how they work
// * there are conditional nodes.. for what?
//   `cudaGraphSetConditional`: runs on the device, but no stream


// What are all the cudagraphs docs?
//   6.30: graph management
//   6.5:  stream management
// Other Resources:
//   https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/

// Idea: 
//   TODO: need to incorporate first touch case as well !!!
//   For a group, execute
//     if(needs to execute op 1 in group) {
//       ...
//       turn off flag 1
//     }
//     if(needs to execute op 2 in group) {
//       ...
//       turn off flag 2
//     }
//     ...
//     if(needs to execute op n in group) {
//       ...
//       turn off flag n
//     }
// > This way, 
//   1. there is no contention since only one executes at a time
//   2. after `n``executions, it will have been completed
//      (assuming each time a flag is set, this node is added)
// Questions:
// 1. How do we set flags? > Use a handle
// 2. How do we create if nodes? > If node is a condtional node with a graph inside
  
////////////////////////////////

// Task 1: create a cudaGraph without touches with groups and without storage
// Task 2: create a cudaGraph with touches
// Task 3: create a cudagraph with storage

cudaGraph_t compile_cuda_graph(
  memgraph_t const& memgraph,
  vector<kernel_manager_t>& kms,
  vector<void*> mems,
  map<string, scalar_t> const& scalar_vars);

