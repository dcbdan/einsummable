#include "execute_state.h"
#include <variant>

void exec_state_t::event_loop() {
  std::queue<int> processing;
  while(true) {
    while(processing.size() > 0) {
      int id = processing.front();
      processing.pop();

      decrement_outs(id); // just adding to ready to run?

      num_remaining--;
    }
    if(num_remaining == 0) {
      return;
    }

    {
      auto iter = ready_to_run.begin();
      while(iter != ready_to_run.end()) {
        int const& id = *iter;
        if(try_to_launch(id)) {
          ready_to_run.erase(iter);
        } else {
          iter++;
        }
      }
    }

    // for each thing in ready to run:
    //   try to grab resource
    //   launch
    //     > inside call back, release resource
    //       and add to just_completed
    std::unique_lock lk(m_notify);
    cv_notify.wait(lk, [&, this] {
      if(just_completed.size() > 0) {
        processing = just_completed;
        just_completed = std::queue<int>();
        return true;
      } else {
        return false;
      }
    });
  }
}

void exec_state_t::decrement_outs(int id) {
  auto const& node = exec_graph.nodes[id];
  for(auto const& out_id: node.outs) {
    int& cnt = num_deps_remaining;
    cnt--;
    if(cnt == 0) {
      ready_to_run.push_back(out_id);
    }
  }
}

bool exec_state_t::try_to_launch(int id) {
  auto const& node = exec_graph.nodes[id];
  auto const& resource_desc = node.required_resources();
  auto resources =
    resource_manager.try_to_acquire_resources(resource_desc);
  if(resources != nullptr) {
    auto callback = [this, id, resources] {
      resources.release();

      {
        std::unique_lock lk(m_notify);
        this->just_completed.push(id);
      }

      cv_notify.notify_one();
    };

    node.launch(resources, callback);

    return true;
  } else {
    return false;
  }
}

void node_t::launch(
  std::shared_ptr<resource_manager_t::resource_holder_t> resource,
  std::function<void()> callback)
{
  using std::holds_alternative;
  using std::get;
​
  if(holds_alternative<dummy_t>(op)) {
    callback();
  } else if(holds_alternative<einsummable_cpu_t>(op)) {
    resource->launch_on_cpu_thread(/* ... */);
  }
}

resource_desc_t exec_graph_t::node_t::get_resource_desc()
{
  using std::holds_alternative;
  using std::get;

  // Checkinging which resource we need depending on the op
  // IF...ELSE... HERE WE GO AGAIN
  if(holds_alternative<dummy_t>(op)) {
    return resource_desc_t::make_empty();
  } 
  else if(holds_alternative<einsummable_cpu_t>(op)) {
    auto const& e = op.get();
    resource_desc_t ret;
    ret.add_cpu_memory();
    ret.add_cpu_thread();
    ret.add_cpu_kernel_manager();
    if(e.worksize > 0) {
      ret.add_cpu_workspace(e.worksize);
    }
    return ret;
  }
  else if (holds_alternative<move_cpu_t>(op)) {
    throw std::runtime_error("Move cpu not implemented");
  }
  else if (holds_alternative<touch_cpu_t>(op)){
    throw std::runtime_error("Touch cpu not implemented");
  }
  else if (holds_alternative<einsummable_gpu_t>(op)) {
    auto const& e = op.get();
    resource_desc_t ret;
    ret.add_gpu_memory();
    ret.add_gpu_stream();
    ret.add_gpu_kernel_manager();
    if(e.workspace_size > 0) {
      ret.add_gpu_workspace(e.workspace_size);
    }
    return ret;
  }
  else if (holds_alternative<touch_gpu_t>(op)) {
    auto const& e = op.get();
    resource_desc_t ret;
    ret.add_gpu_memory();
    ret.add_gpu_stream();
    ret.add_gpu_kernel_manager();
    if(e.workspace_size > 0) {
      ret.add_gpu_workspace(e.workspace_size);
    }
    return ret;
  }
  else if (holds_alternative<copy_gpu_t>(op)){
    auto const& e = op.get();
    resource_desc_t ret;
    ret.add_gpu_memory();
    ret.add_gpu_stream();
    return ret;
  }
  else {
    throw std::runtime_error("should not reach; undefined op type");
  }
}

// single node translation
exec_graph_t::node_t exec_graph_t::gpu_node_translate(
  memgraph_t::node_t const& n)
{
  node_t ret;
  auto mem_op = n.op;
  // see which op it is and translate
  // IF STATEMENTS
  if (mem_op.is_inputmem() || mem_op.is_inputsto() || mem_op.is_del() 
      || mem_op.is_partialize() || mem_op.is_alloc()) {
    ret.op = dummy_t{};
  }
  else if (mem_op.is_touch()){
    ret.op = touch_gpu_t{mem_op.get_touch(), {}, 0};
  }
  else if (mem_op.is_move()){

  }
  else if (mem_op.is_evict()){

  }
  else if (mem_op.is_load()){

  }
  else if (mem_op.is_einsummable()){

  }
  else{
    throw std::runtime_error("should not reach; should exhaust all memgraph ops");
  }

}

// translate memgraph to exec_graph
exec_graph_t make_from_memgraph_with_gpu(
    memgraph_t const& memgraph,
    int num_gpu_per_node,
    int this_node){
  exec_graph_t ret;
  ret.nodes.resize(memgraph.nodes.size());
  for(int i = 0; i < memgraph.nodes.size(); i++) {
    auto const& node = memgraph.nodes[i];
    auto& ret_node = ret.nodes[i];
    ret_node.op = node.op;
    ret_node.outs = node.outs;
    ret_node.num_deps_remaining = node.ins.size();
    ret_node.num_remaining = node.ins.size();
  }
}