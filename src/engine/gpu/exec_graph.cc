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
#include <iostream>
#include <sys/types.h>
#include <unordered_map>

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

  auto insert = [&](op_ptr_t op, int mid)
  {
    auto const& node = memgraph.nodes[mid];
    auto const& mid_inns = node.inns;
    auto const& mid_outs = node.outs;

    vector<int> inns;
    for(auto const& mid_inn: mid_inns) {
      inns.push_back(mid_to_eid.at(mid_inn));
    }

    int eid = graph.insert(op, inns);

    mid_to_eid.insert({mid, eid});
  };

  auto is_local_to_here = [&](int mid) {
    for(int i = 0; i != num_gpus_per_node; ++i) {
      int which_gpu = this_rank*num_gpus_per_node + i;
      // DOUT("Checking if node " << mid << " is local to gpu " << which_gpu);
      if(memgraph.is_local_to(mid, which_gpu)) {
        return true;
      }
    }
    return false;
  };

  // std::unordered_set<einsummable_t> all_einsums;

  int evict_count = 0, load_count = 0;
  uint64_t evict_bytes = 0, load_bytes = 0;

  std::unordered_map<string, einsummable_t> einsums_not_compiled;
  // open a file
  std::ofstream failed_einsums;
  failed_einsums.open("failed_einsums.txt");

  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    if(!is_local_to_here(mid)) {
    //  DOUT("Skipping node " << mid << " because it is not local to this gpu")
     continue;
    }

    auto const& node = memgraph.nodes[mid];
    // DOUT("Making exec graph for node " << mid);

    if(
      node.op.is_inputmem()   ||
      node.op.is_inputsto()   ||
      node.op.is_partialize() ||
      node.op.is_alloc()      ||
      node.op.is_del())
    {
      op_ptr_t op = std::make_shared<dummy_t>();
      insert(op, mid);
      // DOUT("Inserted dummy op for node " << mid);
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

        // if (all_einsums.find(einsum) == all_einsums.end()){
        //   DOUT("einsum: " << einsum);
        //   DOUT("einsum join shape: " << einsum.join_shape);
        //   DOUT("einsum inns: " << einsum.inns);
        //   DOUT("einsum out rank: " << einsum.out_rank);
        //   DOUT("einsum join: " << einsum.join);
        //   DOUT("einsum castable: " << einsum.castable);
        //   DOUT("");
        //   all_einsums.insert(einsum);
        // }

        // build the op (except the workspace size)
        gpu_einsummable_t* op = new gpu_einsummable_t(
          gpu_kms[loc],
          einsum,
          apply.mems,
          node.op.get_apply_loc()
        );
        // Note: op->einsummable and einsum should be the same here
        // DOUT("Einsummable: " << einsum);
        auto maybe_built = gpu_kms[loc].build(einsum);
        if(!maybe_built) {
          // DOUT("Einsummable: " << einsum);
          // DOUT("Debug OP simplified: " << einsum.join.simplify().to_cppstr());
          string e_string = einsum.str();
          if (einsums_not_compiled.find(e_string) == einsums_not_compiled.end()){
            einsums_not_compiled.emplace(e_string, einsum);
            failed_einsums << "einsum:" << einsum << std::endl;
          }
          // throw std::runtime_error("GPU KM could not compile the kernel");
          op_ptr_t op = std::make_shared<dummy_t>();
          insert(op, mid);
          continue;
        }
        auto workspace_info = maybe_built.value();
        if (workspace_info.known()){
          // DOUT(einsum << " workspace size: " << workspace_info.value());
          op->workspace_size = workspace_info.value();
        }
        else{
          throw std::runtime_error("workspace size is not known; this should not happen in cutensor 2.0");
        }
        // insert into the graph
        insert(op_ptr_t(op), mid);
        // DOUT("Inserted einsummable op for node " << mid);
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
        // DOUT("Inserted touch op for node " << mid);
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
      // DOUT("Inserted copy op for node " << mid);
    } else if(node.op.is_evict()) {
      evict_count++;
      evict_bytes += node.op.get_evict().src.size;
      gpu_evict_t* op = new gpu_evict_t(node.op.get_evict());
      insert(op_ptr_t(op), mid);
      // DOUT("Inserted evict op for node " << mid);
    } else if(node.op.is_load()) {
      load_count++;
      load_bytes += node.op.get_load().dst.size;
      gpu_load_t* op = new gpu_load_t(node.op.get_load());
      insert(op_ptr_t(op), mid);
      // DOUT("Inserted load op for node " << mid);
    } else if(node.op.is_constant()) {
      // could be a constant or a lower triangular fill
      auto loc = node.op.get_constant().loc;
      auto const& fill = node.op.get_constant().fill;
      if(fill.is_constant()) {
        gpu_constant_t* op = new gpu_constant_t(gpu_kms[loc], node.op.get_constant());
        insert(op_ptr_t(op), mid);
        // DOUT("Inserted constant op for node " << mid);
      } else if(fill.is_lowertri()) {
        gpu_lowerTri_t* op = new gpu_lowerTri_t(gpu_kms[loc], node.op.get_constant());
        insert(op_ptr_t(op), mid);
        // DOUT("Inserted lowerTri op for node " << mid);
      } else {
        throw std::runtime_error("should not reach: constant fill is neither constant nor lowertri");
      }
    } else {
      throw std::runtime_error("unknown (and unimplemented) op in the memgraph");
    }
  }
  // Debug: print the exec_graph
  // print_exec_graph(graph);
  if (einsums_not_compiled.size() > 0){
    DOUT("The following einsums were not compiled:");
    for (auto [e_string, e_value]: einsums_not_compiled){
      DOUT("einsum: " << e_value);
      DOUT("einsum simplified: " << e_value.join.simplify().to_cppstr());
    }
    throw std::runtime_error("GPU KM could not compile some kernels");
  }
  // close the file
  failed_einsums.close();

  DOUT("The number of nodes in the exec_graph is " << graph.nodes.size());
  fprintf(stdout, "evict_count: %d, evict_bytes: %lu, load_count: %d, load_bytes: %lu\n",
    evict_count, evict_bytes, load_count, load_bytes);
  return graph;
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

  if (workspace_size > 0) {
    gpu_workspace_desc_t workspace_desc;
    workspace_desc.device = device;
    workspace_desc.size = workspace_size;
    ret.emplace_back(gpu_workspace_manager_t::make_desc(workspace_desc));
  }
  return resource_manager_t::make_desc(ret);
}

void gpu_einsummable_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  // DOUT("launching einsummable: " << einsummable);
  // DOUT("gpu_einsummable_t::launch: getting resources")
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);
  // DOUT("number of resources: " << resources.size());

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

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
  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(workspace_size > 0) {
    maybe_workspace = gpu_workspace_manager_t::get_resource(resources[2]).as_tuple();
  }

  cudaStream_t stream = streampool_manager_t::get_resource(resources[1]).stream;

  gpu_km(
    einsummable,
    stream,
    out_mem,
    inn_mems,
    maybe_workspace);

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

  // cudaDeviceSynchronize();

}

desc_ptr_t
gpu_touch_t::resource_description() const
{
  // 1st: gpu memory ptr
  // 2nd: a stream
  // 3rd: a group id (if group_id >= 0)
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));

  ret.emplace_back(streampool_manager_t::make_desc(streampool_desc_t{device}));

  if(group_id >= 0) {
    ret.emplace_back(group_manager_t::make_desc(group_id));
  }

  return resource_manager_t::make_desc(ret);
}

void gpu_touch_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  void* global_buffer = global_buffers_t::get_resource(resources[0]);

  void* out_mem = increment_void_ptr(
    global_buffer,
    mems[0].offset);

  void const* inn_mem = increment_void_ptr(
    global_buffer,
    mems[1].offset);

  bool is_first = false;
  if(group_id >= 0) {
    tuple<int, bool> const& info = group_manager_t::get_resource(resources[2]);
    is_first = std::get<1>(info);
  }

  touch_t this_touch = touch;
  if(is_first) {
    // if this is the first touch, make sure the touch becomes a copy
    this_touch.castable = std::nullopt;
  }

  // create stream and launch
  // cudaSetDevice(device);
  // cudaStream_t stream = cuda_create_stream();
  auto stream = streampool_manager_t::get_resource(resources[1]).stream;
  gpu_km(
    this_touch,
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

  storage.remove(storage_id);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
      // DOUT("in gpu_load callback");
      std::function<void()>* callback_ptr =
        reinterpret_cast<std::function<void()>*>(user_data);
      auto& callback = *callback_ptr;
      callback();
      delete callback_ptr;
    },
    reinterpret_cast<void*>(callback_copy), 0),
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

  gpu_km.constant_fill(fill, stream, gpu_memory);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

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

  gpu_km.lowerTri_fill(fill, stream, gpu_memory);

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
