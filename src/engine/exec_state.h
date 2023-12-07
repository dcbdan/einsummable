#pragma once
#include "../base/setup.h"

#include "exec_graph.h"
#include "resource_manager.h"

// #define EXEC_STATE_PRINT

struct exec_state_t {
  enum class priority_t {
    given,
    bfs,
    dfs,
    random,
  };

  exec_state_t(
    exec_graph_t const& g,
    rm_ptr_t r,
    priority_t priority = priority_t::given,
    int rank = -1);

  struct compare_t {
    inline bool operator()(int const& lhs, int const& rhs) const {
      return self->priorities[lhs] > self->priorities[rhs];
    }

    exec_state_t* self;
  };

  struct queue_t :
    std::priority_queue<int, vector<int>, compare_t>
  {
    queue_t(exec_state_t* self)
      : std::priority_queue<int, vector<int>, compare_t>(compare_t { .self = self })
    {}

    vector<int> const& get() const { return c; }
    vector<int>&       get()       { return c; }
  };

  // execute all nodes in exec graph
  void event_loop();

  // decrement the output nodes add add them
  // to ready_to_run
  void decrement_outs(int id);

  bool try_to_launch(int id);

  vector<int> just_completed;

  queue_t ready_to_run;

  // for each node that is running, these are
  // the resources it is using
  map<int, resource_ptr_t> is_running;

  // for every node in exec graph, the number of dependencies left
  vector<int> num_deps_remaining;

  // for every node in exec graph, the priority of that node
  // (lower priorities come first)
  vector<int> priorities;

  // total number of things left to do
  int num_remaining;

  // for just_completed
  std::mutex m_notify;
  std::condition_variable cv_notify;

  exec_graph_t const& exec_graph;
  rm_ptr_t resource_manager;

#ifdef EXEC_STATE_PRINT 
  std::ofstream out;
#endif
};
