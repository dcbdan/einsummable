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

