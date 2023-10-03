#include "exec_graph.h"

#include "communicator.h"
#include "notifier.h"
#include "channel_manager.h"

int exec_graph_t::insert(
  exec_graph_t::op_ptr_t op,
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

desc_ptr_t
exec_graph_t::notify_recv_ready_t::resource_description() const {
  return resource_manager_t::make_desc(
    vector<desc_ptr_t>{
      notifier_t::make_desc(unit_t{})
    }
  );
}

desc_ptr_t
exec_graph_t::wait_recv_ready_t::resource_description() const {
  return resource_manager_t::make_desc(
    vector<desc_ptr_t>{
      notifier_t::make_desc(unit_t{})
    }
  );
}

desc_ptr_t
exec_graph_t::send_t::resource_description() const {
  // TODO: need to grab a thread resource
  return resource_manager_t::make_desc(
    vector<desc_ptr_t> {
      notifier_t::make_desc(unit_t{}),
      channel_manager_t::make_desc({ true, dst }),
      global_buffers_t::make_desc()
    }
  );
}

desc_ptr_t
exec_graph_t::recv_t::resource_description() const {
  // TODO: need to grab a thread resource
  return resource_manager_t::make_desc(
    vector<desc_ptr_t> {
      notifier_t::make_desc(unit_t{}),
      channel_manager_t::make_desc({ false, src }),
      global_buffers_t::make_desc()
    }
  );
}

void
exec_graph_t::notify_recv_ready_t::launch(
  resource_ptr_t resource,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  std::thread thread([this, callback, notifier] {
    notifier->notify_recv_ready(this->dst, this->id);
    notifier->wait_send_ready(this->id);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::wait_recv_ready_t::launch(
  resource_ptr_t resource,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  std::thread thread([this, callback, notifier] {
    notifier->wait_recv_ready(this->id);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::send_t::launch(
  resource_ptr_t resource,
  std::function<void()> callback) const
{
  // TODO: use a thread resource
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  auto const& wire = channel_manager_t::get_resource(resources[1]);

  void* ptr = increment_void_ptr(
    global_buffers_t::get_resource(resources[2]),
    mem.offset);

  std::thread thread([this, callback, notifier, wire, ptr] {
    notifier->notify_send_ready(this->dst, this->id, wire.get_channel());

    wire.send(ptr, this->mem.size);

    callback();
  });

  thread.detach();
}

void
exec_graph_t::recv_t::launch(
  resource_ptr_t resource,
  std::function<void()> callback) const
{
  // TODO: use a thread resource
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  auto const& wire = channel_manager_t::get_resource(resources[1]);

  void* ptr = increment_void_ptr(
    global_buffers_t::get_resource(resources[2]),
    mem.offset);

  std::thread thread([this, callback, notifier, wire, ptr] {
    int channel = notifier->get_channel(this->id);

    wire.recv(ptr, this->mem.size, channel);

    callback();
  });

  thread.detach();
}
