#include "exec_graph.h"

#include "communicator.h"
#include "notifier.h"

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

void
exec_graph_t::node_t::print(std::ostream& out) const
{
  return std::visit(
    [&out](auto const& op) {
      op.print(out);
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

exec_graph_t::desc_t
exec_graph_t::notify_recv_ready_t::resource_description() const {
  return vector<desc_unit_t>{ notifier_t::desc_t {} };
}

exec_graph_t::desc_t
exec_graph_t::wait_recv_ready_t::resource_description() const {
 return vector<desc_unit_t>{ notifier_t::desc_t {} };
}

exec_graph_t::desc_t
exec_graph_t::send_t::resource_description() const {
  // TODO: need to grab a thread resource
  return vector<desc_unit_t>{
    notifier_t::desc_t {},
    channel_manager_t::desc_t {
      .send = true,
      .loc = dst
    },
    global_buffers_t::desc_t {}
  };
}

exec_graph_t::desc_t
exec_graph_t::recv_t::resource_description() const {
  // TODO: need to grab a thread resource
  return vector<desc_unit_t>{
    notifier_t::desc_t {},
    channel_manager_t::desc_t {
      .send = false,
      .loc = src
    },
    global_buffers_t::desc_t {}
  };
}

void
exec_graph_t::notify_recv_ready_t::launch(
  exec_graph_t::rsrc_t resource,
  std::function<void()> callback) const
{
  notifier_t* notifier = std::get<notifier_t::resource_t>(resource[0]).self;

  std::thread thread([this, callback, notifier] {
    notifier->notify_recv_ready(this->dst, this->id);
    notifier->wait_send_ready(this->id);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::wait_recv_ready_t::launch(
  exec_graph_t::rsrc_t resource,
  std::function<void()> callback) const
{
  notifier_t* notifier = std::get<notifier_t::resource_t>(resource[0]).self;

  std::thread thread([this, callback, notifier] {
    notifier->wait_recv_ready(this->id);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::send_t::launch(
  exec_graph_t::rsrc_t resource,
  std::function<void()> callback) const
{
  // TODO: use a thread resource
  notifier_t* notifier = std::get<notifier_t::resource_t>(resource[0]).self;
  auto const& wire = std::get<channel_manager_t::resource_t>(resource[1]);

  void* ptr = std::get<global_buffers_t::resource_t>(resource[2]).at(mem.offset);

  std::thread thread([this, callback, notifier, wire, ptr] {
    notifier->notify_send_ready(this->dst, this->id, wire.get_channel());

    wire.send(ptr, this->mem.size);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::recv_t::launch(
  exec_graph_t::rsrc_t resource,
  std::function<void()> callback) const
{
  // TODO: use a thread resource
  notifier_t* notifier = std::get<notifier_t::resource_t>(resource[0]).self;
  auto const& wire = std::get<channel_manager_t::resource_t>(resource[1]);
  void* ptr = std::get<global_buffers_t::resource_t>(resource[2]).at(mem.offset);

  std::thread thread([this, callback, notifier, wire, ptr] {
    int channel = notifier->get_channel(this->id);

    wire.recv(ptr, this->mem.size, channel);

    callback();
  });

  thread.detach();
}
