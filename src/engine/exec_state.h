#pragma once
#include "../base/setup.h"

#include "exec_graph.h"
#include "resource_manager.h"

struct exec_state_t {
  exec_state_t(exec_graph_t const& g, resource_manager_t& r);

  // execute all nodes in exec graph
  void event_loop();

  // decrement the output nodes add add them
  // to ready_to_run
  void decrement_outs(int id);

  bool try_to_launch(int id);

  std::queue<int> just_completed;

  vector<int> ready_to_run;

  // this isn't needed but is kept for debugging
  set<int> is_running;

  // for every node in exec graph, the number of dependencies left
  vector<int> num_deps_remaining;

  // total number of things left to do
  int num_remaining;

  // for just_completed
  std::mutex m_notify;
  std::condition_variable cv_notify;

  exec_graph_t const& exec_graph;
  resource_manager_t& resource_manager;
};
