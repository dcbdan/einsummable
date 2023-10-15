#include "exec_graph.h"

#include "communicator.h"
#include "notifier.h"
#include "channel_manager.h"

#include "../base/buffer.h"
#include "../einsummable/dbuffer.h"

std::mutex einsummable_total_mutex;
double einsummable_total = 0.0;

ghost_t make_einsummable_ghost() {
  return ghost_t(einsummable_total_mutex, einsummable_total);
}

double get_einsummable_total() { return einsummable_total; }

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
  return resource_manager_t::make_desc(
    vector<desc_ptr_t> {
      notifier_t::make_desc(unit_t{}),
      send_channel_manager_t::make_desc(dst),
      global_buffers_t::make_desc(),
      threadpool_manager_t::make_desc()
    }
  );
}

desc_ptr_t
exec_graph_t::recv_t::resource_description() const {
  return resource_manager_t::make_desc(
    vector<desc_ptr_t> {
      recv_channel_manager_t::make_desc({ id, src }),
      global_buffers_t::make_desc(),
      threadpool_manager_t::make_desc()
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
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  auto const& wire = send_channel_manager_t::get_resource(resources[1]);

  void* ptr = increment_void_ptr(
    global_buffers_t::get_resource(resources[2]),
    mem.offset);

  auto& thread_resource = threadpool_manager_t::get_resource(resources[3]);

  thread_resource.launch([this, callback, notifier, wire, ptr] {
    notifier->notify_send_ready(this->dst, this->id, wire.channel);

    wire.send(ptr, this->mem.size);

    callback();
  });
}

void
exec_graph_t::recv_t::launch(
  resource_ptr_t resource,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(resource);

  auto const& wire = recv_channel_manager_t::get_resource(resources[0]);

  void* ptr = increment_void_ptr(
    global_buffers_t::get_resource(resources[1]),
    mem.offset);

  auto& thread_resource = threadpool_manager_t::get_resource(resources[2]);

  thread_resource.launch([this, callback, wire, ptr] {
    wire.recv(ptr, this->mem.size);

    callback();
  });
}
