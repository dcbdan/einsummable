#pragma once
#include "../base/setup.h"

#include "../einsummable/taskgraph.h"

struct cluster_settings_t {
  cluster_settings_t(int n_node, int n_worker_per);

  double start_cost;
  vector<vector<double>> speed_per_byte;
  vector<int> nworkers_per_node;

  int num_nodes() const { return nworkers_per_node.size(); }
};

double bytes_cost(
  taskgraph_t const& taskgraph,
  cluster_settings_t const& settings);

struct bytes_cost_busy_t {
  double finish;
  int tid;
  int loc;
};

bool operator>(bytes_cost_busy_t const& lhs, bytes_cost_busy_t const& rhs);

struct bytes_cost_state_t {
  bytes_cost_state_t(
    taskgraph_t const& taskgraph,
    cluster_settings_t const& settings);

  // taskgraph has input / einsummable / partialize / move

  taskgraph_t const& taskgraph;
  cluster_settings_t const& settings;

  double time;
  priority_queue_least<bytes_cost_busy_t> busy_workers;
  vector<int> num_avail_workers;

  vector<int> counts;
  vector<int> pending_tasks;
  int num_finished;

  bool step();

  void decrement(set<int> const& outs);

  double get_cost(taskgraph_t::op_t const& op);
};
