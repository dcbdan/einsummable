#include "exec_graph.h"

exec_graph_t::desc_t
exec_graph_t::node_t::resource_description() const
{
  return std::visit(
    [](auto const& op){ return op.resource_description(); },
    op);
}

void
exec_graph_t::node_t::launch(
  exec_graph_t::rsrc_t resource,
  std::function<void()> callback) const
{
  return std::visit(
    [resource, callback](auto const& op) {
      op.launch(resource, callback);
    },
    op);
}

int exec_graph_t::insert(
  exec_graph_t::op_t const& op,
  vector<int> const& inns)
{
  int ret = nodes.size();

  nodes.push_back(node_t {
    .op = op,
    .inns = inns,
    .outs = vector<int>{}
  });

  for(auto const& inn: inns) {
    nodes[inn].outs.push_back(ret);
  }

  return ret;
}

