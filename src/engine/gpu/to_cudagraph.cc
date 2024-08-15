#include "to_cudagraph.h"

cudaGraph_t compile_cuda_graph(
  memgraph_t const& memgraph,
  vector<kernel_manager_t>& kms,
  vector<void*> mems,
  map<string, scalar_t> const& scalar_vars)
{
  int num_gpus = kms.size();
  if(mems.size() != num_gpus) {
    throw std::runtime_error("mems.size() != kms.size()");
  }

  vector<cudaGraphNode_t> cnodes(memgraph.nodes.size(), NULL);

  cudaGraph_t ret;
  handle_cuda_error(cudaGraphCreate(&ret,0), "cannot create graph");

  vector<cudaGraphNode_t> deps;
  deps.reserve(10);

  int n_nodes = memgraph.nodes.size();
  //for(int mid = 0; mid != n_nodes; ++mid) {
  //  auto const& node = memgraph.nodes[mid];
  //  std::cout << mid << ": ";
  //  node.op.print_type(std::cout);
  //  std::cout << std::endl;

  //  if(node.op.is_touch()) {
  //    auto const& apply = node.op.get_apply();
  //    auto const& touch = apply.get_touch();
  //    DOUT("device:       " << apply.loc);
  //    DOUT("has_castable: " << bool(touch.castable));
  //    DOUT("dtype:        " << touch.dtype);

  //    for(auto const& s: touch.selection) {
  //      DOUT("");
  //      s.print();
  //    }

  //    DOUT("");
  //  }
  //}

  for(int mid = 0; mid != n_nodes; ++mid) {
    auto const& node = memgraph.nodes[mid];

    deps.resize(0);
    // TODO TODO TODO for(int const& inn: node.inns) {
    // TODO TODO TODO   deps.push_back(cnodes[inn]);
    // TODO TODO TODO }
    if(mid > 0) {
      deps.push_back(cnodes[mid-1]);
    }

    if(node.op.is_inputmem()   || 
       node.op.is_partialize() ||
       node.op.is_alloc()      ||
       node.op.is_del())
    {
      handle_cuda_error(
        cudaGraphAddEmptyNode(&cnodes[mid], ret, deps.data(), deps.size()),
        "cuda graph add empty node");
     
    } else if(node.op.is_constant()) {
      cudaGraph_t g;
      cudaStream_t stream;
      handle_cuda_error(cudaGraphCreate(&g,0), "cannot create graph");

      int device = node.op.get_loc();
      handle_cuda_error(cudaSetDevice(device));
      handle_cuda_error(cudaStreamCreate(&stream));

      handle_cuda_error(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));

      auto const& info = node.op.get_constant();
      void* out = increment_void_ptr(mems[device], info.offset);
     
      if(info.fill.is_constant()) {
        kms[device].constant_fill(info.fill.get_constant(), stream, out);
      } else if(info.fill.is_lowertri()) {
        kms[device].lowerTri_fill(info.fill.get_lowertri(), stream, out);
      } else {
        throw std::runtime_error("should not reach");
      }

      handle_cuda_error(cudaStreamEndCapture(stream, &g));

      handle_cuda_error(cudaGraphAddChildGraphNode(&cnodes[mid], ret, deps.data(), deps.size(), g));

      handle_cuda_error(cudaStreamDestroy(stream), "");
      handle_cuda_error(cudaGraphDestroy(g), "");
    } else if(node.op.is_einsummable()) {
      auto const& apply = node.op.get_apply();
      if(apply.group >= 0) {
        throw std::runtime_error("einsummable should not have group");
      }

      int const& device = apply.loc;

      einsummable_t e = apply.get_einsummable()
        .replace_scalar_variables(scalar_vars)
        .merge_adjacent_dims();
      auto maybe_built = kms[device].build(e);
      if(!maybe_built) {
        throw std::runtime_error("could not compile einsum");
      }

      uint64_t wsz = 0;
      auto const& workspace_info = maybe_built.value();
      if(workspace_info.known()) {
        wsz = workspace_info.value();
      } else {
        // TODO: how do we deal with this case?
        wsz = dtype_size(e.out_dtype()) * product(e.join_shape);
      }

      optional<tuple<void*, uint64_t>> workspace;
      if(wsz > 0) {
        throw std::runtime_error("workspace is not supported...");
      } 

      void* global_buffer = mems[device];
      void* out_mem = increment_void_ptr(
        global_buffer,
        apply.mems[0].offset);

      vector<void const*> inn_mems;
      inn_mems.reserve(apply.mems.size() - 1);
      for(int i = 1; i != apply.mems.size(); ++i) {
        inn_mems.push_back(increment_void_ptr(
          global_buffer,
          apply.mems[i].offset));
      }

      {
        cudaGraph_t g;
        cudaStream_t stream;
        handle_cuda_error(cudaGraphCreate(&g,0), "cannot create graph");
  
        handle_cuda_error(cudaSetDevice(device));
        handle_cuda_error(cudaStreamCreate(&stream));
  
        handle_cuda_error(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 
        kms[device](e, stream, out_mem, inn_mems, workspace);
  
        handle_cuda_error(cudaStreamEndCapture(stream, &g));
  
        handle_cuda_error(
          cudaGraphAddChildGraphNode(
            &cnodes[mid], ret, deps.data(), deps.size(), g));
  
        handle_cuda_error(cudaStreamDestroy(stream));
        handle_cuda_error(cudaGraphDestroy(g));
      }

    } else if(node.op.is_touch()) {
      auto const& apply = node.op.get_apply();
      auto const& touch = apply.get_touch();
      int const& device = apply.loc;
      if(apply.group >= 0) {
        throw std::runtime_error("touch with group: not implemented");
      } else {
        if(touch.castable) {
          throw std::runtime_error("should not have a castable here");
        }
      }

      void* out_mem = increment_void_ptr(
        mems[device],
        apply.mems[0].offset);

      void const* inn_mem = increment_void_ptr(
        mems[device],
        apply.mems[1].offset);

      {
        cudaGraph_t g;
        cudaStream_t stream;
        handle_cuda_error(cudaGraphCreate(&g,0), "cannot create graph");
  
        handle_cuda_error(cudaSetDevice(device));
        handle_cuda_error(cudaStreamCreate(&stream));
  
        handle_cuda_error(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 
        launch_touch_kernel(touch, stream, out_mem, inn_mem);
  
        handle_cuda_error(cudaStreamEndCapture(stream, &g));
  
        handle_cuda_error(
          cudaGraphAddChildGraphNode(
            &cnodes[mid], ret, deps.data(), deps.size(), g));
  
        handle_cuda_error(cudaStreamDestroy(stream), "");
        handle_cuda_error(cudaGraphDestroy(g), "");
      }
    } else if(node.op.is_move()) {
      auto const& move = node.op.get_move();
      auto const& [src_device, src_offset] = move.src;
      auto const& [dst_device, dst_offset] = move.dst;

      void const* src_mem = increment_void_ptr(
        mems[src_device],
        src_offset);

      void* dst_mem = increment_void_ptr(
        mems[dst_device],
        dst_offset);

      handle_cuda_error(cudaGraphAddMemcpyNode1D(
        &cnodes[mid],
        ret,
        deps.data(), deps.size(),
        dst_mem, src_mem, move.size,
        cudaMemcpyDeviceToDevice));
    } else {
      throw std::runtime_error("compile cuda graph: missing node implementation");
    }
  }

  // 1. For each memgraph node, create a cudaGraphNode
  // 2. For each op, create a cudaGraph and insert via AddChildGraphNode

  // How do we create a graph? Should we just use streams?
  // Can we capture op with stream, create a graph, append graph to main graph?
  //    Ez: cudaGraphAddChildGraphNode
  // How should we deal with dummy nodes?
  //   Either 1: create a new memgraph with no dummies  No
  //          2: add dummy cudaGraph nodes              EZ: cudaGraphAddEmptyNode
  //          3: reach past dummy nodes                 No

  // UGH. cudaGraphAddChildGraphNode cannot do memory stuff. BWOAH

  return ret;
}
