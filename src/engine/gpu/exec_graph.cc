#include "exec_nodes.h"
#include "../exec_graph.h"
#include "workspace.h"
#include "utility.h"


// TODO: have a stream pool implementation once creating a stream on fly works
exec_graph_t exec_graph_t::make_gpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  kernel_manager_t& gpu_km,
  int num_gpus_per_node)
{
  exec_graph_t graph;

  map<int, int> mid_to_eid;

  auto insert = [&](op_ptr_t op, int mid)
  {
    auto const& node = memgraph.nodes[mid];
    auto const& mid_inns = node.inns;
    auto const& mid_outs = node.outs;

    vector<int> inns;
    for(auto const& mid: mid_inns) {
      inns.push_back(mid_to_eid.at(mid));
    }

    int eid = graph.insert(op, inns);

    mid_to_eid.insert({mid, eid});
  };

  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    auto const& node = memgraph.nodes[mid];
    if(!node.op.is_local_to_gpu(this_rank, num_gpus_per_node)) {
      continue;
    }

    if(
      node.op.is_inputmem()   ||
      node.op.is_inputsto()   ||
      node.op.is_partialize() ||
      node.op.is_alloc()      ||
      node.op.is_del())
    {
      op_ptr_t op = std::make_shared<dummy_t>();
      insert(op, mid);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      if(apply.is_einsummable()) {
        if(apply.group >= 0) {
          throw std::runtime_error("only allowing touches to have a group");
        }

        // build the op (except the workspace size)
        gpu_einsummable_t* op = new gpu_einsummable_t(
          gpu_km,
          apply.get_einsummable().merge_adjacent_dims(),
          apply.mems,
          node.op.get_apply_loc()
        );

        // compile the kernel (and update the workspace size)
        auto maybe_registered = gpu_km.build(op->einsummable);
        if(!maybe_registered) {
          throw std::runtime_error("GPU KM could not compile the kernel");
        }
        op->workspace_size = maybe_registered.value().value();

        // insert into the graph
        insert(op_ptr_t(op), mid);
      } else if(apply.is_touch()) {
        gpu_touch_t* op = new gpu_touch_t(
          gpu_km,
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
        throw std::runtime_error("should not reach");
      }
    } else if(node.op.is_move()) {
      gpu_copy_t* op = new gpu_copy_t(node.op.get_move());
      insert(op_ptr_t(op), mid);
    } else if(node.op.is_evict()) {
      throw std::runtime_error("GPU evict not implemented");
    } else if(node.op.is_load()) {
      throw std::runtime_error("GPU load not implemented");
    } else {
      throw std::runtime_error("should not reach");
    }
  }

  return graph;
}

desc_ptr_t
gpu_einsummable_t::resource_description() const
{
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc(device));

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
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

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
    maybe_workspace = gpu_workspace_manager_t::get_resource(resources[1]).as_tuple();
  }

  // create stream and launch
  cudaSetDevice(device);
  cudaStream_t stream = cuda_create_stream();
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
  vector<desc_ptr_t> ret;
  ret.emplace_back(global_buffers_t::make_desc());

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
    tuple<int, bool> const& info = group_manager_t::get_resource(resources[1]);
    is_first = std::get<1>(info);
  }

  touch_t this_touch = touch;
  if(is_first) {
    // if this is the first touch, make sure the touch becomes a copy
    this_touch.castable = std::nullopt;
  }

  // create stream and launch
  cudaSetDevice(device);
  cudaStream_t stream = cuda_create_stream();
  gpu_km(
    this_touch,
    stream,
    out_mem,
    inn_mem);

  std::function<void()>* callback_copy = new std::function<void()>(callback);

  handle_cuda_error(cudaStreamAddCallback(
    stream,
    [](cudaStream_t stream, cudaError_t status, void* user_data) {
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
  auto my_move = move;
  auto [src_loc, src_offset] = my_move.src;
  auto [dst_loc, dst_offset] = my_move.dst;

  return global_buffers_t::make_multi_desc({src_loc, dst_loc});
}

void gpu_copy_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto all_buffers_rsrc = resource_manager_t::get_resource(resources[0]);
  auto src_buffer = global_buffers_t::get_resource(all_buffers_rsrc[0]);
  auto dst_buffer = global_buffers_t::get_resource(all_buffers_rsrc[1]);

  auto [src_loc, src_offset] = move.src;
  auto [dst_loc, dst_offset] = move.dst;

  void* src_mem = increment_void_ptr(
    src_buffer,
    src_offset);

  void* dst_mem = increment_void_ptr(
    dst_buffer,
    dst_offset);

  cudaSetDevice(src_loc);
  cudaStream_t stream = cuda_create_stream();
  cudaError_t cudaError = cudaMemcpyAsync(dst_mem, src_mem, move.size, cudaMemcpyDeviceToDevice, stream);
  if (cudaError != cudaSuccess) {
    // print cpy size
    fprintf(stderr, "cpy size: %zu\n", move.size);
    // print the error code and error string
    fprintf(stderr, "cudaMemcpy failed with error: %s\n", cudaGetErrorString(cudaError));
    throw std::runtime_error("cudaMemcpy failed");
  }
}
