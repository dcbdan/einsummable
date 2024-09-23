#include "to_cudagraph.h"
#include <fstream>

cudaGraphNode_t get_capture_out_graph_node(cudaStream_t stream) {
  cudaGraphNode_t const* ds;
  cudaStreamCaptureStatus status;
  size_t n = 0;
  handle_cuda_error(cudaStreamGetCaptureInfo(stream, &status, 0, 0, &ds, &n));
  if(n != 1) {
    throw std::runtime_error("Invalid! n = " + write_with_ss(n));
  }
  return ds[0];
}

bool _is_dummy(memgraph_t const& memgraph, int mid) {
  auto const& node = memgraph.nodes[mid];
  return node.op.is_inputmem()   || 
         node.op.is_partialize() ||
         node.op.is_alloc()      ||
         node.op.is_del()        ||
         node.op.is_barrier()     ;
}

void _reach_past_dummies(memgraph_t const& memgraph, int mid, set<int>& ret) {
  if(_is_dummy(memgraph, mid)) {
    auto const& node = memgraph.nodes[mid];
    for(int const& inn: node.inns) {
      _reach_past_dummies(memgraph, inn, ret);
    }
  } else {
    ret.insert(mid);
  }
}

set<int> _get_deps(memgraph_t const& memgraph, int mid) {
  set<int> ret;
  auto const& node = memgraph.nodes[mid];
  for(int const& inn: node.inns) {
    _reach_past_dummies(memgraph, inn, ret);
  }
  return ret;
}

void create_dep_tree(cudaGraph_t& ret, vector<cudaGraphNode_t>& deps) {
  vector<cudaGraphNode_t> tmp;
  while(deps.size() > 2) {
    tmp = vector<cudaGraphNode_t>( (deps.size() + 1) / 2 );
    for(int i = 0; i != tmp.size(); ++i) {
      if(i + 1 == tmp.size()) {
        tmp[i] = deps[2*i];
      } else {
        handle_cuda_error(
          cudaGraphAddEmptyNode(&tmp[i], ret, deps.data() + (2*i), 2),
          "cuda graph add empty node");
      }
    }
    std::copy(tmp.begin(), tmp.end(), deps.begin());
    deps.resize(tmp.size());
  }
}

cudaGraph_t compile_cuda_graph(
  memgraph_t const& memgraph,
  vector<kernel_manager_t>& kms,
  vector<void*> mems,
  map<string, scalar_t> const& scalar_vars)
{
  //DLINEOUT("memgraph size: " << memgraph.nodes.size());
  //{
  //  std::ofstream f("mg.gv");
  //  memgraph.print_graphviz(f);
  //  DOUT("printed mg.gv");
  //}
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

  for(int mid = 0; mid != n_nodes; ++mid) {
    if(_is_dummy(memgraph, mid)) {
      continue;
    }

    auto const& node = memgraph.nodes[mid];

    deps.resize(0);
    for(int const& inn: _get_deps(memgraph, mid)) {
      deps.push_back(cnodes[inn]);
    }
    // No actual performance improvement...
    //   if(deps.size() > 8) {
    //     // modify deps 
    //     create_dep_tree(ret, deps);
    //   }

    if(node.op.is_constant()) {
      cudaStream_t stream;

      int device = node.op.get_loc();
      handle_cuda_error(cudaSetDevice(device));
      handle_cuda_error(cudaStreamCreate(&stream));

      handle_cuda_error(cudaStreamBeginCaptureToGraph(
        stream, ret,
        deps.data(), NULL, deps.size(), 
        cudaStreamCaptureModeGlobal));

      auto const& info = node.op.get_constant();
      void* out = increment_void_ptr(mems[device], info.offset);
     
      if(info.fill.is_constant()) {
        kms[device].constant_fill(info.fill.get_constant(), stream, out);
      } else if(info.fill.is_lowertri()) {
        kms[device].lowerTri_fill(info.fill.get_lowertri(), stream, out);
      } else {
        throw std::runtime_error("should not reach");
      }

      // DOUT("constant fill");
      cnodes[mid] = get_capture_out_graph_node(stream);

      handle_cuda_error(cudaStreamEndCapture(stream, &ret));

      handle_cuda_error(cudaStreamDestroy(stream), "");
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
      // maybe_built will also tell us the worksize, but we assume
      // that has been provided as part of the memgraph

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

      optional<tuple<void*, uint64_t>> workspace;
      if(apply.workspace) {
        mem_t const& w = apply.workspace.value();
        workspace = {
          increment_void_ptr(global_buffer, w.offset),
          w.size
        };
      }

      cudaStream_t stream;
  
      handle_cuda_error(cudaSetDevice(device));
      handle_cuda_error(cudaStreamCreate(&stream));
  
      handle_cuda_error(cudaStreamBeginCaptureToGraph(
        stream,ret,
        deps.data(), NULL, deps.size(),
        cudaStreamCaptureModeGlobal));

      kms[device](e, stream, out_mem, inn_mems, workspace);

      // DOUT("einsum: " << e);
      cnodes[mid] = get_capture_out_graph_node(stream);

      handle_cuda_error(cudaStreamEndCapture(stream, &ret));
 
      handle_cuda_error(cudaStreamDestroy(stream));
    } else if(node.op.is_touch()) {
      auto const& apply = node.op.get_apply();
      auto const& touch = apply.get_touch();
      int const& device = apply.loc;

      if(apply.workspace) {
        throw std::runtime_error("should not have a workspace for touch");
      }

      if(apply.group >= 0) {
        if(!touch.castable) {
          throw std::runtime_error("with group, must have castable!");
        }
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

      cudaStream_t stream;
      handle_cuda_error(cudaSetDevice(device));
      handle_cuda_error(cudaStreamCreate(&stream));
  
      handle_cuda_error(cudaStreamBeginCaptureToGraph(
        stream, ret,
        deps.data(), NULL, deps.size(),
        cudaStreamCaptureModeGlobal));

      // DOUT("touch");
      // DOUT(touch);
 
      launch_touch_kernel(touch, stream, out_mem, inn_mem);

      cnodes[mid] = get_capture_out_graph_node(stream);

      handle_cuda_error(cudaStreamEndCapture(stream, &ret));

      handle_cuda_error(cudaStreamDestroy(stream), "");
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

  return ret;
}

