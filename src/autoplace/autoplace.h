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

// TODO: name this something else;
//       single_loc_no_partition maybe
vector<placement_t> single_loc_placements(graph_t const& graph);

vector<partition_t> autopartition(
  graph_t const& graph,
  equal_items_t<int> eqs,
  int nloc);

equal_items_t<int> construct_equal_placements(graph_t const& graph);
void construct_equal_placements_inplace(graph_t const& graph, equal_items_t<int>&);

struct mcmc_t {
  mcmc_t(
    cluster_t const& cl,
    graph_t const& gr,
    double bt,
    equal_items_t<int> const& equal_placements,
    vector<placement_t> const& initial_placements);

  static mcmc_t init_with_single_loc(
    cluster_t const& cl,
    graph_t const& gr,
    double bt,
    equal_items_t<int> eqs = {});
  static mcmc_t init_balanced(
    cluster_t const& cl,
    graph_t const& gr,
    double bt,
    equal_items_t<int> eqs = {});

  bool step();

  vector<placement_t> random_change() const;

  int num_locs() const;
  int num_workers() const;

  int random_gid() const;

  static placement_t make_finer(placement_t const& pl);

  static placement_t make_coarser(placement_t const& pl);

  cluster_t const& cluster;
  graph_t const& graph;
  double beta;
  equal_items_t<int> const equal_placements;
  vector<int> candidates;

  double best_makespan;
  vector<placement_t> best_placements;

  double current_makespan;
  vector<placement_t> current_placements;
};
