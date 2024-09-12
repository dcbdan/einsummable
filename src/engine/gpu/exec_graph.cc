#include "exec_nodes.h"
#include "../exec_graph.h"
#include "managers.h"
#include "resource_manager.h"
#include "workspace.h"
#include "storage_manager.h"
#include "stream_pool.h"
#include "utility.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/types.h>
#include <unordered_map>

static int w = 0;
bool debug_exec_graph = false;

void print_exec_graph(exec_graph_t exec_graph){
  for (int i = 0; i < exec_graph.nodes.size(); ++i) {
    auto node = exec_graph.nodes[i];
    std::cout << "Node " << i << " has input: ";
    for (auto in : node.inns) {
      std::cout << in << " ";
    }
    std::cout << "and output: ";
    for (auto out : node.outs) {
      std::cout << out << " ";
    }
    std::cout << std::endl;
  }
}

exec_graph_t exec_graph_t::make_gpu_super_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  vector<kernel_manager_t>& gpu_kms,
  int num_gpus_per_node,
  vector<void*> gpu_mems,
  map<string, scalar_t> const& scalar_vars)
{
  if(num_gpus_per_node != gpu_mems.size()) {
    throw std::runtime_error("must have world size 1");
  }
  if(gpu_kms.size() != gpu_mems.size()) {
    throw std::runtime_error("must have 1 km per gpu");
  }
  if(this_rank != 0) {
    throw std::runtime_error("this rank must be zero");
  }

  super_graph_t super = create_super_graph(memgraph); 

  exec_graph_t graph;
  map<int, int> sid_to_eid;

  auto insert = [&](op_ptr_t op, int sid) {
    vector<int> inns;
    for(auto const& inn_sid: super.nodes[sid].inns) {
      inns.push_back(sid_to_eid.at(inn_sid));
    }
    int eid = graph.insert(op, inns);
    sid_to_eid.insert({sid, eid});
  };

  auto create_super_op = [&](vector<int> const& mids) {
    vector<memgraph_t::op_t> ops;
    int loc = -1;
    uint64_t workspace_size = 0;
    for(auto const& mid: mids) {
      auto const& oo = memgraph.nodes[mid].op;
      if(oo.is_partialize() || oo.is_alloc() || oo.is_del() ||
         oo.is_inputmem() || oo.is_inputsto())
      {
        continue;
      }

      ops.push_back(oo);

      auto& op = ops.back();

      int l;
      if(op.is_move()) {
        l = op.get_move().get_src_loc();
      } else {
        l = op.get_loc();
      }

      if(loc < 0) {
        loc = l;
      } else if(loc != l) {
        throw std::runtime_error("invalid super node location...");
      }

      if(op.is_einsummable()) {
        einsummable_t einsum = op.get_apply()
          .get_einsummable()
          .replace_scalar_variables(scalar_vars)
          .merge_adjacent_dims();

        // rewrite the einsummable on the stored op 
        op.get_apply().op = einsum;

        auto maybe_built = gpu_kms[loc].build(einsum);
        if(!maybe_built) {
          throw std::runtime_error("could not compile einsum");
        }

        uint64_t wsz = 0;
        auto const& workspace_info = maybe_built.value();
        if(workspace_info.known()) {
          wsz = workspace_info.value();
        } else {
          // TODO: how do we deal with this case?
          wsz = dtype_size(einsum.out_dtype()) * product(einsum.join_shape);
        }

        workspace_size = std::max(workspace_size, wsz);
      }
    }

    if(loc < 0) {
      if(ops.size() != 0) {
        throw std::runtime_error("should not be an op!");
      }

      dummy_t* op = new dummy_t();
      return op_ptr_t(op);
    }

    gpu_super_t* op = new gpu_super_t(loc, gpu_kms[loc], workspace_size, ops);
    return op_ptr_t(op);
  };

  for(int sid = 0; sid != super.nodes.size(); ++sid) {
    auto const& node = super.nodes[sid];

    insert(
      create_super_op(node.ops),
      sid);
  }

  return graph;  
}

exec_graph_t exec_graph_t::make_gpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  vector<kernel_manager_t>& gpu_kms,
  int num_gpus_per_node,
  vector<void*> gpu_mems,
  map<string, scalar_t> const& scalar_vars)
{
  exec_graph_t graph;

  map<int, int> mid_to_eid;

  auto is_dummy = [](memgraph_t::node_t const& node) {
    return node.op.is_inputmem()   ||
           node.op.is_inputsto()   ||
           node.op.is_partialize() ||
           node.op.is_alloc()      ||
           node.op.is_del()        ||
           node.op.is_barrier()     ;
  };

  auto insert = [&](op_ptr_t op, int mid)
  {
    auto const& node = memgraph.nodes[mid];
    vector<int> mid_inns(node.inns.begin(), node.inns.end());

    set<int> inns;
    for(int idx = 0; idx != mid_inns.size(); ++idx) {
      int mid_inn = mid_inns[idx];
      auto const& inn_node = memgraph.nodes[mid_inn];
      if(is_dummy(inn_node)) {
        vector<int> more_inns(inn_node.inns.begin(), inn_node.inns.end());
        vector_concatenate_into(mid_inns, more_inns);
      } else {
        inns.insert(mid_to_eid.at(mid_inn));
      }
    }

    int eid = graph.insert(op, vector<int>(inns.begin(), inns.end()));

    mid_to_eid.insert({mid, eid});
  };

  if(this_rank != 0) {
    throw std::runtime_error("world size must be zero...");
  }

  auto is_local_to_here = [&](int mid) {
    return true;
  //  for(int i = 0; i != num_gpus_per_node; ++i) {
  //    int which_gpu = this_rank*num_gpus_per_node + i;
  //    // DOUT("Checking if node " << mid << " is local to gpu " << which_gpu);
  //    if(memgraph.is_local_to(mid, which_gpu)) {
  //      return true;
  //    }
  //  }

  //  return false;
  };

  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    if(!is_local_to_here(mid)) {
      if(!memgraph.nodes[mid].op.is_inputsto()) {
        throw std::runtime_error("this mg node doesn't occur here...expecing 1 gpu node with multi-gpus only; not impl");
      }
     continue;
    }

    auto const& node = memgraph.nodes[mid];

    if(is_dummy(node)){
      continue;
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      int loc = node.op.get_apply_loc();
      if(apply.is_einsummable()) {
        if(apply.group >= 0) {
          throw std::runtime_error("only allowing touches to have a group");
        }

        einsummable_t einsum = apply
          .get_einsummable()
          .replace_scalar_variables(scalar_vars)
          .merge_adjacent_dims();

        // build the op (except the workspace size)

        // Note: op->einsummable and einsum should be the same here
        auto maybe_built = gpu_kms[loc].build(einsum);
        if(!maybe_built) {
          throw std::runtime_error("could not build einsum: exec graph gpu make");
        }

        gpu_einsummable_t* op = new gpu_einsummable_t(
          gpu_kms[loc],
          einsum,
          apply.mems,
          apply.workspace,
          node.op.get_apply_loc(),
          gpu_kms[loc].get_built_kernel_info(einsum)
        );
        
        // insert into the graph
        insert(op_ptr_t(op), mid);
      } else if(apply.is_touch()) {
        gpu_touch_t* op = new gpu_touch_t(
          gpu_kms[loc],
          apply.get_touch(),
          apply.group,
          apply.mems,
          node.op.get_apply_loc()
        );

        // Any touch that does not have a group id is the only write to
        // the output bytes, so make sure it's castable is none so that
        // it does a copy and not a sum
        if(op->group_id < 0) {
          op->touch.castable = std::nullopt;
        }

        insert(op_ptr_t(op), mid);
      } else {
        throw std::runtime_error("should not reach: node op is apply but not einsummable or touch");
      }
    } else if(node.op.is_move()) {
      // check if the move is local to this gpu
      auto const& move = node.op.get_move();
      auto const& src = move.get_src_loc();
      auto const& dst = move.get_dst_loc();
      if (std::floor(src/num_gpus_per_node) != std::floor(dst/num_gpus_per_node)) {
        throw std::runtime_error("node to node communication not supported yet");
      }
      gpu_copy_t* op = new gpu_copy_t(node.op.get_move());
      insert(op_ptr_t(op), mid);
    } else if(node.op.is_evict()) {
      gpu_evict_t* op = new gpu_evict_t(node.op.get_evict());
      insert(op_ptr_t(op), mid);
    } else if(node.op.is_load()) {
      gpu_load_t* op = new gpu_load_t(node.op.get_load());
      insert(op_ptr_t(op), mid);
    } else if(node.op.is_constant()) {
      // could be a constant or a lower triangular fill
      auto loc = node.op.get_constant().loc;
      auto const& fill = node.op.get_constant().fill;
      if(fill.is_constant()) {
        gpu_constant_t* op = new gpu_constant_t(gpu_kms[loc], node.op.get_constant());
        insert(op_ptr_t(op), mid);
      } else if(fill.is_lowertri()) {
        gpu_lowerTri_t* op = new gpu_lowerTri_t(gpu_kms[loc], node.op.get_constant());
        insert(op_ptr_t(op), mid);
      } else {
        throw std::runtime_error("should not reach: constant fill is neither constant nor lowertri");
      }
    } else {
      throw std::runtime_error("unknown (and unimplemented) op in the memgraph");
    }
  }

  return graph;
}

desc_ptr_t gpu_super_t::resource_description() const
{
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(super_loc));
  // TODO: device != loc in multi-node multi-gpu-per-node setting
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{super_loc}));
  if(workspace_size > 0) {
    gpu_workspace_desc_t workspace_desc;
    workspace_desc.device = super_loc;
    workspace_desc.size = workspace_size;
    ret.emplace_back(gpu_workspace_manager_t::make_desc(workspace_desc));
  }

  {
    set<int> groups;
    for(auto const& op: ops) {
      if(op.is_touch()) {
        int const& group_id = op.get_apply().group;
        if(group_id >= 0 && groups.count(group_id) == 0) {
          groups.insert(group_id);
          ret.emplace_back(group_manager_t::make_desc(group_id));
        }
      }
    }
  }

  {
    set<int> dsts;
    for(auto const& op: ops) {
      if(op.is_move()) {
        int dst = op.get_move().get_dst_loc();
        dsts.insert(dst);
        ret.emplace_back(global_buffers_t::make_desc(dst));
      }
    }
  }

  return resource_manager_t::make_desc(ret);
}

void gpu_super_t::launch(
  resource_ptr_t rsrc, 
  std::function<void()> callback) const
{
  // Resources:
  // 1. global buffer
  // 2. stream
  // 3. maybe the workspace.
  // 4. the groups
  // 5. the dst buffers
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  int resource_id = 0;

  void* global_buffer = global_buffers_t::get_resource(resources[resource_id++]);

  cudaStream_t stream = streampool_manager_t::get_resource(resources[resource_id++]).stream;

  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(workspace_size > 0) {
    maybe_workspace = gpu_workspace_manager_t::get_resource(
      resources[resource_id++]).as_tuple();
  }
  
  // NOTE: It may be the case that multiple touches on this super op
  //       have the same group_id. Make sure that only the first touch
  //       serves as initialization. So after each touch, 
  //       set group_infos[group_id] to false
  //       !!!!!!!!!!!!!!!
  map<int, bool> group_infos{ {-1, false} }; // make sure we have -1 as a key
  for(auto const& op: ops) {
    if(op.is_touch()) {
      int const& group_id = op.get_apply().group;
      if(group_id >= 0 && group_infos.count(group_id) == 0) {
        auto const& [group_id_, is_first] =
          group_manager_t::get_resource(resources[resource_id++]);
        if(group_id != group_id_) {
          throw std::runtime_error("invalid group id from the resource manager");
        }
        group_infos.insert({group_id, is_first});
      }
    }
  }

  map<int, void*> loc_to_buffer;
  for(auto const& op: ops) {
    if(op.is_move()) {
      int dst = op.get_move().get_dst_loc();
      if(loc_to_buffer.count(dst) == 0) {
        loc_to_buffer.insert({
          dst,
          global_buffers_t::get_resource(resources[resource_id++])});
      }
    }
  }

  for(auto const& op: ops) {
    if(op.is_constant()) {
      auto const& c = op.get_constant();
      if(c.fill.is_constant()) {
        // TODO
        throw std::runtime_error("constant constant super not impl");
      } else if(c.fill.is_lowertri()) {
        // TODO
        throw std::runtime_error("constant lowertri super not impl");
      }
    } else if(op.is_einsummable()) {
      auto const& a = op.get_apply();
      auto const& mems = a.mems;
      auto const& e = a.get_einsummable();

      void* out_mem = increment_void_ptr(
        global_buffer,
        mems[0].offset);

      vector<void const*> inn_mems;
      inn_mems.reserve(mems.size() - 1);
      for(int i = 1; i != mems.size(); ++i) {
        inn_mems.push_back(increment_void_ptr(
          global_buffer,
          mems[i].offset));
      }

      km(
        e,
        stream,
        out_mem,
        inn_mems,
        maybe_workspace);
    } else if(op.is_touch()) {
      auto const& a = op.get_apply();
      auto const& mems = a.mems;

      void* out_mem = increment_void_ptr(
        global_buffer,
        mems[0].offset);
      void const* inn_mem = increment_void_ptr(
        global_buffer,
        mems[1].offset);

      int group_id = a.group;
      if(group_id < 0) {
        group_id = -1;
      }
      bool& is_first = group_infos.at(group_id);

      touch_t touch = a.get_touch();
      if(is_first) {
        // if this is the first touch, make sure the touch becomes a copy
        touch.castable = std::nullopt;
        // Make sure that the next touch will not be incorrectly treated 
        // as a copy!
        is_first = false;
      }

      if(group_id < 0 && touch.castable != std::nullopt) {
        // Just in case!
        throw std::runtime_error("without group id, touch must be a copy");
      }

      km(
        touch,
        stream,
        out_mem,
        inn_mem);
    } else if(op.is_move()) {
      auto const& move = op.get_move();
      auto const& [src_loc, src_offset] = move.src;
      auto const& [dst_loc, dst_offset] = move.dst;

      void* src_buffer = global_buffer;
      void* dst_buffer = loc_to_buffer.at(dst_loc);

      void* src_mem = increment_void_ptr(
        src_buffer,
        src_offset);

      void* dst_mem = increment_void_ptr(
        dst_buffer,
        dst_offset);

      cudaError_t cudaError = cudaMemcpyAsync(
        dst_mem, src_mem, move.size, cudaMemcpyDeviceToDevice, stream);
      if(cudaError != cudaSuccess) {
        throw std::runtime_error("CudaMemcpyAsync in super node failed");
      }
    } else if(op.is_evict()) {
      // TODO
      throw std::runtime_error("gpu_super_t::launch: evict not implemented");
    } else if(op.is_load()) {
      // TODO
      throw std::runtime_error("gpu_super_t::launch: load not implemented");
    } else if(op.is_partialize() || op.is_alloc() || op.is_del()) {
      // Nothing to do, these are dummy ops
    } else {
      throw std::runtime_error("gpu_super_t::launch: missing mg op case");
    }
  }

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_copy callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_copy_t: adding callback");
}

desc_ptr_t
gpu_einsummable_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: a workspace (if workspace_size > 0)
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));

  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  return resource_manager_t::make_desc(ret);
}

void gpu_einsummable_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  string sxsx = "gpu_einsummable_t::launch";
  if(einsummable.is_contraction()) {
    sxsx += "contraction";
  } else if(einsummable.has_aggregation()) {
    sxsx += "aggregation";
  } else {
    sxsx += "elementwise";
  }
  auto gremlin = get_rm_timetracker().make_totals_gremlin(sxsx);

  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);
  // DOUT("number of resources: " << resources.size());

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(workspace) {
    mem_t const& workspace_mem = workspace.value();
    void* m = increment_void_ptr(global_buffer, workspace_mem.offset);
    maybe_workspace = tuple<void*, uint64_t>{m, workspace_mem.size};
  }

  cudaStream_t stream = streampool_manager_t::get_resource(resources[1]).stream;

  void* out_mem = increment_void_ptr(
    global_buffer,
    mems[0].offset);

  vector<void const*> inn_mems;
  inn_mems.reserve(mems.size() - 1);
  for(int i = 1; i != mems.size(); ++i) {
    inn_mems.push_back(increment_void_ptr(
      global_buffer,
      mems[i].offset));
  }

  {
    auto gremlin = get_rm_timetracker().make_totals_gremlin("gpu_km");
    gpu_km(
      my_kernel_info,
      stream,
      out_mem,
      inn_mems,
      maybe_workspace);
  }

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
    // DOUT("in gpu_einsummable callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_einsummable_t: callback");
}

desc_ptr_t
gpu_touch_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));

  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  return resource_manager_t::make_desc(ret);
}

void gpu_touch_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  auto gremlin = get_rm_timetracker().make_totals_gremlin("gpu_touch_t::launch");
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* out_mem = increment_void_ptr(
    global_buffer,
    mems[0].offset);

  void const* inn_mem = increment_void_ptr(
    global_buffer,
    mems[1].offset);

  // print the offsets
  // DOUT("Touch Output offset: " << mems[0].offset);
  // DOUT("Touch Input offset: " << mems[1].offset);

  // create stream and launch
  // cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;

  gpu_km(
    touch,
    stream,
    out_mem,
    inn_mem);

  // cudaDeviceSynchronize();
  // callback();

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_touch callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_touch_t: callback");
}

desc_ptr_t
gpu_copy_t::resource_description() const
{
  // 1st: gpu memory ptrs (this could be one resource or two depending on the design)
  // 2nd: a stream
  vector<desc_ptr_t> ret;
  auto my_move = move;
  auto [src_loc, src_offset] = my_move.src;
  auto [dst_loc, dst_offset] = my_move.dst;
  ret.emplace_back(global_buffers_t::make_desc(src_loc));
  ret.emplace_back(global_buffers_t::make_desc(dst_loc));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{dst_loc}));

  return resource_manager_t::make_desc(ret);
}

void gpu_copy_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  auto gremlin = get_rm_timetracker().make_totals_gremlin("gpu_copy_t::launch");
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto src_buffer = global_buffers_t::get_resource(resources[0]);
  auto dst_buffer = global_buffers_t::get_resource(resources[1]);

  auto [src_loc, src_offset] = move.src;
  auto [dst_loc, dst_offset] = move.dst;

  auto move_size = move.size;

  void* src_mem = increment_void_ptr(
    src_buffer,
    src_offset);

  void* dst_mem = increment_void_ptr(
    dst_buffer,
    dst_offset);

  // print 100 elements of src

  auto stream = streampool_manager_t::get_resource(resources[2]).stream;

  cudaError_t cudaError = cudaMemcpyAsync(dst_mem, src_mem, move.size, cudaMemcpyDeviceToDevice, stream);

  if (cudaError != cudaSuccess) {
    // print the error code and error string
    DOUT("cudaMemcpy failed with error: " << cudaGetErrorString(cudaError));
    DOUT("src loc, offset: " << src_loc << " " << src_offset);
    DOUT("dst loc, offset: " << dst_loc << " " << dst_offset);
    DOUT("size:    " << move.size);

    throw std::runtime_error("CudaMemcpy failed @ gpu_copy_t");
  }

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_copy callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_copy_t: adding callback");
}

desc_ptr_t
gpu_evict_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: storage object
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));
  ret.emplace_back(gpu_storage_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

void gpu_evict_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* gpu_memory = increment_void_ptr(
    global_buffer,
    gpu_offset);

  // create stream and launch
  cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;

  auto& storage = *gpu_storage_manager_t::get_resource(resources[2]).ptr;
  buffer_t buffer = storage.alloc(size, storage_id);

  handle_cuda_error(
    cudaMemcpyAsync(buffer->raw(), gpu_memory, size, cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync in gpu_evict_t");

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_evict callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_evict_t: callback");
}

desc_ptr_t
gpu_load_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: storage object
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));
  ret.emplace_back(gpu_storage_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

struct gpu_load_info_t {
  std::function<void()> callback;
  gpu_storage_t* storage;
  int storage_id;
};

void gpu_load_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* gpu_memory = increment_void_ptr(
    global_buffer,
    gpu_offset);

  // create stream and launch
  cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;

  auto& storage = *gpu_storage_manager_t::get_resource(resources[2]).ptr;

  buffer_t buffer = storage.reference(storage_id);

  handle_cuda_error(
    cudaMemcpyAsync(gpu_memory, buffer->data, size, cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync in gpu_load_t");

  gpu_load_info_t* info = new gpu_load_info_t;
  info->callback = callback;
  info->storage = &storage;
  info->storage_id = storage_id;

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      gpu_load_info_t* info_ptr = reinterpret_cast<gpu_load_info_t*>(user_data);
      auto& info = *info_ptr;
      info.storage->remove(info.storage_id);
      info.callback();
      delete info_ptr;
    },
    reinterpret_cast<void*>(info), 0),
    "gpu_load_t: callback");
}

// constant and lowerTri need the same resources since they are the same memgraph nodes

desc_ptr_t
gpu_constant_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  return resource_manager_t::make_desc(ret);
}

desc_ptr_t
gpu_lowerTri_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));
  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  return resource_manager_t::make_desc(ret);
}

void gpu_constant_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* gpu_memory = increment_void_ptr(
    global_buffer,
    gpu_offset);

  // create stream and launch
  cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;

  auto num_elements = 1;
  for (auto dim : fill.shape) {
    num_elements *= dim;
  }

  // DOUT("fill: " << fill.value << " offset: " << gpu_offset << " nelms: " << num_elements);

  gpu_km.constant_fill(fill, stream, gpu_memory);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  // cudaDeviceSynchronize();
  // DOUT("Output from constant: ");
  // printFloatGPU(gpu_memory, 20);

  // callback();

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_constant callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_constant_t: callback");
}

void gpu_lowerTri_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* gpu_memory = increment_void_ptr(
    global_buffer,
    gpu_offset);

  // create stream and launch
  cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;

  auto gpu_fill = fill;

  if (fill.upper == scalar_t::negative_inf(dtype_t::f32)){
    // DOUT("Found negative infinity lower triangular fill");
    scalar_t new_upper(-1 * 1e30);
    auto l = fill_t::lowertri_t {
      .lower = fill.lower,
      .upper = new_upper,
      .nrow = fill.nrow,
      .ncol = fill.ncol,
      .start = fill.start
    };
    gpu_fill = l;
  }
  // DOUT("new upper: " << gpu_fill.upper);

  gpu_km.lowerTri_fill(gpu_fill, stream, gpu_memory);

  // DOUT("lower tri fill lower: " << fill.lower << " upper: " << fill.upper << " offset: " << gpu_offset
  //   << " nrows: " << fill.nrow << " ncols: " << fill.ncol);

  // cudaDeviceSynchronize();
  // DOUT("Output from lower_tri fill: ");
  // printFloatGPU(gpu_memory, 20);
  // callback();

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_lowerTri callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
    "gpu_lowerTri_t: callback");
}
