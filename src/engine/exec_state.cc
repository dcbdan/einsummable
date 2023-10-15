#include "exec_state.h"

exec_state_t::exec_state_t(exec_graph_t const& g, rm_ptr_t r)
  : exec_graph(g), resource_manager(r)
{
  int num_nodes = exec_graph.nodes.size();

  num_remaining = num_nodes;

  num_deps_remaining.reserve(num_nodes);
  for(int id = 0; id != num_nodes; ++id) {
    int num_deps = g.nodes[id].inns.size();

    num_deps_remaining.push_back(num_deps);

    if(num_deps == 0) {
      ready_to_run.push_back(id);
    }
  }
}

void exec_state_t::event_loop() {
  std::queue<int> processing;
  while(true) {
    while(processing.size() > 0) {
      int id = processing.front();
      processing.pop();
      is_running.erase(id);

      decrement_outs(id);

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
          is_running.insert(id);
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
    int& cnt = num_deps_remaining[out_id];
    cnt--;
    if(cnt == 0) {
      ready_to_run.push_back(out_id);
    }
  }
}

bool exec_state_t::try_to_launch(int id) {
  auto const& node = exec_graph.nodes[id];
  desc_ptr_t resource_desc = node.resource_description();
  resource_ptr_t resources =
    resource_manager->try_to_acquire(resource_desc);
  if(resources) {
    auto callback = [this, id, resources] {
      resource_manager->release(resources);

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

