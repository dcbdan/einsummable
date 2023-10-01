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
  exec_graph_t graph(gpu_km);

  map<int, int> mid_to_eid;

  auto insert = [&](op_t op, int mid)
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
      insert(dummy_t{}, mid);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      if(apply.is_einsummable()) {
        if(apply.group >= 0) {
          throw std::runtime_error("only allowing touches to have a group");
        }

        // build the op (except the workspace size)
        gpu_einsummable_t op {
          .gpu_km = gpu_km,
          .einsummable = apply.get_einsummable().merge_adjacent_dims(),
          .mems = apply.mems,
          .device = node.op.get_apply_loc()
        };

        // compile the kernel (and update the workspace size)
        auto maybe_registered = gpu_km.build(op.einsummable);
        if(!maybe_registered) {
          throw std::runtime_error("GPU KM could not compile the kernel");
        }
        op.worksize = maybe_registered.value();

        // insert into the graph
        insert(op, mid);
      } else if(apply.is_touch()) {
        gpu_touch_t op {
          .gpu_km = gpu_km,
          .touch = apply.get_touch(),
          .group_id = apply.group,
          .mems = apply.mems,
          .device = node.op.get_apply_loc()
        };

        // Any touch that does not have a group id is the only write to
        // the output bytes, so make sure it's castable is none so that
        // it does a copy and not a sum
        if(op.group_id < 0) {
          op.touch.castable = std::nullopt;
        }

        insert(op, mid);
      } else {
        throw std::runtime_error("should not reach");
      }
    } else if(node.op.is_move()) {
      gpu_copy_t op {
        .move = node.op.get_move()
      };
      insert(op, mid);
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

exec_graph_t::desc_t
exec_graph_t::gpu_einsummable_t::resource_description() const
{
  vector<desc_unit_t> ret;
  ret.emplace_back(global_buffers_t::desc_t{});

  if (worksize.value() > 0) {
    ret.emplace_back(workspace_t::desc_t{worksize.value()});
  }

  return ret;
}

void exec_graph_t::gpu_einsummable_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffers_t::resource_t>(resources[0]).ptr);

  void* out_mem = reinterpret_cast<void*>(
    ptr + mems[0].offset);

  vector<void const*> inn_mems;
  inn_mems.reserve(mems.size() - 1);
  for(int i = 1; i != mems.size(); ++i) {
    inn_mems.push_back(reinterpret_cast<void const*>(
      ptr + mems[i].offset));
  }

  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(worksize.value() > 0) {
    maybe_workspace =
      std::get<workspace_manager_t::resource_t>(resources[1]).as_tuple();
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

exec_graph_t::desc_t
exec_graph_t::gpu_touch_t::resource_description() const
{
  vector<desc_unit_t> ret;
  ret.emplace_back(global_buffers_t::desc_t{});

  if(group_id >= 0) {
    ret.emplace_back(group_manager_t::desc_t { group_id });
  }

  return ret;
}

void exec_graph_t::gpu_touch_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffers_t::resource_t>(resources[0]).ptr);

  void* out_mem = reinterpret_cast<void*>(
    ptr + mems[0].offset);

  void const* inn_mem = reinterpret_cast<void const*>(
    ptr + mems[1].offset);

  bool is_first = false;
  if(group_id >= 0) {
    is_first = std::get<group_manager_t::resource_t>(resources[1]).is_first;
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
    touch,
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

exec_graph_t::desc_unit_t
exec_graph_t::gpu_copy_t::resource_description() const
{
  return global_buffers_t::desc_t{};
}

void exec_graph_t::gpu_copy_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffers_t::resource_t>(resources[0]).ptr);

  void* dst_mem = reinterpret_cast<void*>(
    ptr + mems[0].offset);

  void const* inn_mem = reinterpret_cast<void const*>(
    ptr + mems[1].offset);
}
