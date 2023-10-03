#include "../exec_graph.h"

#include "exec_nodes.h"

exec_graph_t
exec_graph_t::make_cpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  cpu_kernel_executor_t& cpu_executor)
{
  exec_graph_t graph(cpu_executor);

  map<int, int> mid_to_eid;

  // Insert the op and get the dependencies from mid. Note that
  // this does not check for whether the mid deps are local or not
  auto insert_from_mid = [&](op_ptr_t op, int mid)
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
      insert_from_mid(op, mid);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      if(apply.is_einsummable()) {
        if(apply.group >= 0) {
          throw std::runtime_error("only allowing touches to have a group");
        }

        // build the op (except the workspace size)
        cpu_einsummable_t* op = new cpu_einsummable_t(
          cpu_executor,
          apply.get_einsummable().merge_adjacent_dims(),
          apply.mems
        );

        // compile the kernel (and update the workspace size)
        auto maybe_registered = cpu_executor.build(op->einsummable);
        if(!maybe_registered) {
          throw std::runtime_error("could not compile the kernel");
        }
        op->workspace_size = maybe_registered.value();

        // insert into the graph
        insert_from_mid(op_ptr_t(op), mid);
      } else if(apply.is_touch()) {
        cpu_touch_t* op = new cpu_touch_t (
          cpu_executor,
          apply.get_touch(),
          apply.group,
          apply.mems
        );

        // Any touch that does not have a group id is the only write to
        // the output bytes, so make sure it's castable is none so that
        // it does a copy and not a sum
        if(op->group_id < 0) {
          op->touch.castable = std::nullopt;
        }

        insert_from_mid(op_ptr_t(op), mid);
      } else {
        throw std::runtime_error("should not reach");
      }
    } else if(node.op.is_move()) {
      auto const& move = node.op.get_move();
      auto const& [src, src_offset] = move.src;
      auto const& [dst, dst_offset] = move.dst;

      vector<int> local_deps;
      for(auto const& dep_mid: node.inns) {
        if(memgraph.nodes[dep_mid].op.is_local_to(this_rank)) {
          local_deps.push_back(mid_to_eid.at(dep_mid));
        }
      }

      if(src == dst) {
        throw std::runtime_error("not supporting moves within self");
      }
      if(src == this_rank) {
        // On this machine, we are sending.
        mem_t mem {
          .offset = src_offset,
          .size = move.size
        };

        int recv_ready_eid = graph.insert(
          op_ptr_t(new exec_graph_t::wait_recv_ready_t(mid, dst)),
          local_deps);

        int send_eid = graph.insert(
          op_ptr_t(new exec_graph_t::send_t(mid, dst, mem)),
          { recv_ready_eid });

        mid_to_eid.insert({mid, send_eid});
      } else if(dst == this_rank) {
        // On this machine, we are recving.
        mem_t mem {
          .offset = dst_offset,
          .size = move.size
        };

        int recv_ready_eid = graph.insert(
          op_ptr_t(new notify_recv_ready_t(mid, src)),
          local_deps);

        int recv_eid = graph.insert(
          op_ptr_t(new recv_t(mid, src, mem)),
          { recv_ready_eid });

        mid_to_eid.insert({mid, recv_eid});
      } else {
        throw std::runtime_error("this move is not local!");
      }
    } else if(node.op.is_evict()) {
      auto const& evict = node.op.get_evict();
      cpu_evict_t* op = new cpu_evict_t(
        evict.dst.id,
        evict.src.as_mem()
      );
      insert_from_mid(op_ptr_t(op), mid);
    } else if(node.op.is_load()) {
      auto const& load = node.op.get_load();
      cpu_load_t* op = new cpu_load_t(
        load.src.id,
        load.dst.as_mem()
      );
      insert_from_mid(op_ptr_t(op), mid);
    } else {
      throw std::runtime_error("should not reach");
    }
  }

  return graph;
}

exec_graph_t::desc_t
cpu_einsummable_t::resource_description() const
{
  vector<desc_unit_t> ret;
  ret.emplace_back(global_buffers_t::desc_t{ .which = 0 });

  // TODO: insert threadpool resource description

  if(workspace_size > 0) {
    ret.emplace_back(cpu_workspace_manager_t::desc_t { .size = workspace_size });
  }
  return ret;
}

void cpu_einsummable_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  auto const& global_buffer = std::get<global_buffers_t::resource_t>(resources[0]);

  void* out_mem = global_buffer.at(mems[0].offset);

  vector<void const*> inn_mems;
  inn_mems.reserve(mems.size() - 1);
  for(int i = 1; i != mems.size(); ++i) {
    inn_mems.push_back(global_buffer.at(mems[i].offset));
  }

  optional<tuple<void*, uint64_t>> maybe_workspace;
  if(workspace_size > 0) {
    maybe_workspace =
      std::get<cpu_workspace_manager_t::resource_t>(resources[1]).as_tuple();
  }

  // TODO use threadpool resource.

  // But since we're not using a threadpool resource, we're just going
  // to launch a thread and let it float in the wind by calling detach
  std::thread thread(
    [this, callback, out_mem, inn_mems, maybe_workspace]
    {
      cpu_executor(einsummable, out_mem, inn_mems, maybe_workspace);
      callback();
    });
  // Note: capturing this under the assumption that the exec_graph will not
  //       change and invalidate the this pointer

  thread.detach();
}

exec_graph_t::desc_t
cpu_touch_t::resource_description() const
{
  vector<desc_unit_t> ret;

  ret.emplace_back(global_buffers_t::desc_t{ .which = 0 });

  // TODO add threadpool resource description

  if(group_id >= 0) {
    ret.emplace_back(group_manager_t::desc_t { group_id });
  }

  return ret;
}

void cpu_touch_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  auto const& global_buffer = std::get<global_buffers_t::resource_t>(resources[0]);

  void* out_mem = global_buffer.at(mems[0].offset);

  void const* inn_mem = global_buffer.at(mems[1].offset);

  bool is_first = false;
  if(group_id >= 0) {
    is_first = std::get<group_manager_t::resource_t>(resources[1]).is_first;
  }

  touch_t this_touch = touch;
  if(is_first) {
    // if this is the first touch, make sure the touch becomes a copy
    this_touch.castable = std::nullopt;
  }

  // TODO use threadpool resource.

  // But since we're not using a threadpool resource, we're just going
  // to launch a thread and let it float in the wind by calling detach
  std::thread thread([this, callback, this_touch, out_mem, inn_mem] {
    cpu_executor(this_touch, out_mem, inn_mem);
    callback();
  });

  thread.detach();
}

exec_graph_t::desc_t
cpu_evict_t::resource_description() const
{
  vector<desc_unit_t> ret;

  ret.emplace_back(global_buffers_t::desc_t{ .which = 0 });
  ret.emplace_back(cpu_storage_manager_t::desc_t{});

  return ret;
}

void cpu_evict_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffers_t::resource_t>(resources[0]).ptr);
  cpu_storage_t* storage =
    std::get<cpu_storage_manager_t::resource_t>(resources[1]).ptr;

  std::thread thread([this, callback, storage, ptr] {
    buffer_t data = make_buffer_reference(ptr + mem.offset, mem.size);
    storage->write(data, id);
    callback();
  });

  thread.detach();
}

exec_graph_t::desc_t
cpu_load_t::resource_description() const
{
  vector<desc_unit_t> ret;

  ret.emplace_back(global_buffers_t::desc_t{ .which = 0 });
  ret.emplace_back(cpu_storage_manager_t::desc_t{});

  return ret;
}

void cpu_load_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffers_t::resource_t>(resources[0]).ptr);
  cpu_storage_t* storage =
    std::get<cpu_storage_manager_t::resource_t>(resources[1]).ptr;

  std::thread thread([this, callback, storage, ptr] {
    buffer_t data = make_buffer_reference(ptr + mem.offset, mem.size);
    storage->load(data, id);
    callback();
  });

  thread.detach();
}

