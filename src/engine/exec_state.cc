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

