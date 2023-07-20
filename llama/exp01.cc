#include "misc.h"
#include "modules.h"
#include "builder.h"

#include "../src/einsummable/graph.h"
#include "../src/einsummable/reference.h"
#include "../src/autoplace/fsmcmc.h"
#include "../src/autoplace/rwmcmc.h"
#include "../src/autoplace/loadbalanceplace.h"
#include "../src/autoplace/autopart.h"

struct cluster_settings_t {
  int num_nodes;
  int num_threads_per_node;
  uint64_t compute_per_thread;
  uint64_t bandwidth;
};

cluster_t make_cluster(cluster_settings_t settings)
{
  int      const& num_nodes            = settings.num_nodes;
  int      const& num_threads_per_node = settings.num_threads_per_node;
  uint64_t const& compute_per_thread   = settings.compute_per_thread;
  uint64_t const& bandwidth            = settings.bandwidth;

  using device_t = cluster_t::device_t;
  using connection_t = cluster_t::connection_t;

  // all workers compute kernels single threaded
  auto f_cost = [compute_per_thread](einsummable_t const& e) {
    uint64_t flops = product(e.join_shape);
    double time = (1.0 / compute_per_thread) * flops;
    return tuple<int,double>{1, time};
  };

  vector<device_t> devices(num_nodes, device_t(
    num_threads_per_node,
    f_cost));

  vector<connection_t> connections;
  for(int i = 0; i != num_nodes; ++i) {
  for(int j = 0; j != num_nodes; ++j) {
    if(i != j) {
      connections.push_back(connection_t {
        .bandwidth = bandwidth,
        .src = i,
        .dst = j
      });
    }
  }}

  return cluster_t::make(devices, connections);
}

vector<placement_t> solve_forwardsim(
  graph_t const& graph,
  cluster_settings_t cluster_settings,
  int num_steps,
  double beta)
{
  cluster_t cluster = make_cluster(cluster_settings);

  auto mcmc = forwardsim_mcmc_t::init_balanced(cluster, graph, beta);
  for(int i = 0; i != num_steps; ++i) {
    mcmc.step();
  }

  return mcmc.best_placements;
}

vector<placement_t> solve_relationwise(
  graph_t const& graph,
  int nlocs,
  int n_threads_per_node,
  int max_blocks,
  int num_steps,
  double beta)
{
  auto kernel_coster = kernel_coster_t::for_cpu_cluster(nlocs);

  relationwise_mcmc_t mcmc(
    graph, kernel_coster,
    nlocs, n_threads_per_node, max_blocks,
    equal_items_t<int>());

  DOUT("single thread cost " << mcmc.cost());

  {
    uint64_t min_sizing = 1;
    vector<partition_t> parts = autopartition(
      graph, min_sizing, nlocs * n_threads_per_node, mcmc.get_equal_gids());

    // this ignores the equal gids
    vector<placement_t> pls = load_balanced_placement(graph, parts, nlocs, false);

    mcmc.set_placements(pls);
  }

  DOUT("balanced cost " << mcmc.cost());

  for(int i = 0; i != num_steps; ++i) {
    if(i % 10000 == 0) {
      DOUT( i << " / " << num_steps << "    " << mcmc.get_best_cost() );
    }
    mcmc.step(beta);
  }

  DOUT(num_steps << " / " << num_steps << "   " << mcmc.get_best_cost() );
  return mcmc.get_best_placements();
}

int main() {
  set_default_dtype(dtype_t::f16);

  uint64_t bsz    = 4;
  uint64_t seqlen = 8;

  auto args = model_args_t::llama_7B(bsz);

  int num_nodes = 8;
  int num_threads_per_node = 12;
  int num_steps = 30000;

  double beta = 10000.0;

  auto autoplace = [&](graph_t const& graph) {
    //uint64_t giga = 1e9;
    //cluster_settings_t cluster_settings {
    //  .num_nodes = num_nodes,
    //  .num_threads_per_node = num_threads_per_node,
    //  .compute_per_thread = 1*giga,
    //  .bandwidth = 10*giga
    //};
    //return solve_forwardsim(graph, cluster_settings, num_steps, beta);

    int max_blocks = 2 * num_nodes * num_threads_per_node;
    return solve_relationwise(
      graph, num_nodes, num_threads_per_node, max_blocks,
      num_steps,
      beta);
  };

  builder_t builder = builder_t::make_first_token(args, seqlen, autoplace);
}
