#pragma once
#include "../base/setup.h"

#include "cluster.h"
#include "forwardsim.h"

#include "../base/placement.h"
#include "../einsummable/graph.h"

double simulate(
  cluster_t const& cluster,
  graph_t const& graph,
  vector<placement_t> const& pls);

vector<placement_t> single_loc_placements(graph_t const& graph);

struct mcmc_t {
  mcmc_t(
    cluster_t const& cl,
    graph_t const& gr,
    double bt,
    vector<placement_t> placements);

  mcmc_t(
    cluster_t const& cl,
    graph_t const& gr,
    double bt);

  bool step();

  vector<placement_t> random_change() const;

  static placement_t make_finer(placement_t const& pl);

  static placement_t make_coarser(placement_t const& pl);

  cluster_t const& cluster;
  graph_t const& graph;
  double beta;

  double best_makespan;
  vector<placement_t> best_placements;

  double current_makespan;
  vector<placement_t> current_placements;
};
