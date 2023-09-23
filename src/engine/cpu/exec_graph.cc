#include "../exec_graph.h"

exec_graph_t
exec_graph_t::make_cpu_exec_graph(
  memgraph_t const& memgraph,
  int this_rank,
  cpu_kernel_executor_t& cpu_executor)
{
  exec_graph_t graph {
    .cpu_executor = cpu_executor
  };

  map<int, int> mid_to_eid;
  map<int, int> eid_to_mid; // TODO: is this one needed?

  auto insert = [&](op_t op, int mid)
  {
    auto const& node = memgraph.nodes[mid];
    auto const& mid_inns = node.inns;
    auto const& mid_outs = node.outs;

    vector<int> inns;
    for(auto const& mid: mid_inns) {
      inns.push_back(mid_to_eid.at(mid));
    }

    vector<int> outs;
    for(auto const& mid: mid_outs) {
      outs.push_back(mid_to_eid.at(mid));
    }

    graph.nodes.push_back(node_t {
      .op = op,
      .inns = inns,
      .outs = outs
    });
    int eid = graph.nodes.size() - 1;

    mid_to_eid.insert({mid, eid});
    eid_to_mid.insert({eid, mid});
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
        cpu_einsummable_t op {
          .cpu_executor = cpu_executor,
          .einsummable = apply.get_einsummable().merge_adjacent_dims(),
          .mems = apply.mems,
          .workspace_size = 0
        };

        // compile the kernel (and update the workspace size)
        auto maybe_registered = cpu_executor.build(op.einsummable);
        if(!maybe_registered) {
          throw std::runtime_error("could not compile the kernel");
        }
        op.workspace_size = maybe_registered.value();

        // insert into the graph
        insert(op, mid);
      } else if(apply.is_touch()) {
        cpu_touch_t op {
          .cpu_executor = cpu_executor,
          .touch = apply.get_touch(),
          .mems = apply.mems
        };
        insert(op, mid);
      } else {
        throw std::runtime_error("should not reach");
      }
    } else if(node.op.is_move()) {
      // TODO
      throw std::runtime_error("moves are not implemented");
    } else if(node.op.is_evict()) {
      // TODO
      throw std::runtime_error("evicts are not implemented");
    } else if(node.op.is_load()) {
      // TODO
      throw std::runtime_error("loads are not implemented");
    } else {
      throw std::runtime_error("should not reach");
    }
  }

  return graph;
}

exec_graph_t::desc_t
exec_graph_t::cpu_einsummable_t::resource_description() const
{
  vector<desc_unit_t> ret;
  ret.emplace_back(global_buffer_t::desc_t{});

  // TODO: insert threadpool resource description

  if(workspace_size > 0) {
    ret.emplace_back(cpu_workspace_manager_t::desc_t { .size = workspace_size });
  }
  return ret;
}

void exec_graph_t::cpu_einsummable_t::launch(
  exec_graph_t::rsrc_t resources,
  std::function<void()> callback) const
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(
    std::get<global_buffer_t::resource_t>(resources[0]).ptr);

  void* out_mem = reinterpret_cast<void*>(
    ptr + mems[0].size);

  vector<void const*> inn_mems;
  inn_mems.reserve(mems.size() - 1);
  for(int i = 1; i != mems.size(); ++i) {
    inn_mems.push_back(reinterpret_cast<void const*>(
      ptr + mems[i].size));
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

