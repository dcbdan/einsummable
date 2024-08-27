#pragma once

#include "../../einsummable/memgraph.h"
#include "gpu_kernel_manager.h"

struct streamlist_t {
  struct op_t {
    // wait for these events
    vector<int> wait;
    // execute this node on this stream
    int mid;  
    int device;
    int stream;
    // record that we finished to this event
    optional<int> event;

    // For apply nodes, compile the kernel info
    optional<kernel_manager_t::kernel_info_t> kernel_info;
    // NOTE: will not run if this is not set
  };

  memgraph_t const& memgraph;

  vector<op_t> ops;
  int num_streams_per_device;
  int num_devices;
  int num_events;

  static streamlist_t make(
    memgraph_t const& memgraph, 
    int num_streams_per_device);

  void compile_kernels(
    vector<kernel_manager_t>& kms,
    map<string, scalar_t> const& scalar_vars);

  void execute(
    vector<kernel_manager_t>& kms,
    vector<vector<cudaStream_t>>& stream_pools,
    vector<void*> mems, 
    bool loud) const;
};
