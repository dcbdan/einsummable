#include "exec_nodes.h"
#include "../exec_graph.h"
#include "gpu_kernel_manager.h"
#include "workspace.h"
#include "utility.h"


// TODO: have a stream pool implementation once creating a stream on fly works
exec_graph_t
exec_graph_t::make_gpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  kernel_manager_t& gpu_km)
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
    if(!node.op.is_local_to(this_rank)) {
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
        op->worksize = maybe_registered.value();

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
      gpu_copy_t* op = new gpu_copy_t(
        .move = node.op.get_move()
      );
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
  ret.emplace_back(global_buffers_t::make_desc());

  if (worksize.value() > 0) {
    ret.emplace_back(workspace_t::make_desc(worksize.value()));
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
  return resource_manager_t::make_desc(
    vecotr<desc_ptr_t>(
      global_buffers_t::make_desc()
    )
  );
}

void gpu_copy_t::launch(
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

  // TODO: do the actual thing here
}