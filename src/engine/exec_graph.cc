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
    [](
      auto const& op,
      rsrc_t resource,
      std::function<void()> callback)
      {
        return op.launch(resource, callback);
      },
      op, resource, callback);
}
