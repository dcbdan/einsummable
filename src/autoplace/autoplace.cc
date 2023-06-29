#include "autoplace.h"
#include "loadbalanceplace.h"
#include "autopart.h"

double simulate(
  cluster_t const& cluster,
  graph_t const& graph,
  vector<placement_t> const& pls)
{
  forward_state_t state(cluster, graph);

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    state.assign_placement(gid, pls[gid]);
  }

  double makespan = 0.0;
  while(!state.all_done()) {
    state.enqueue_all();
    auto completed = state.pop_work();
    makespan = std::max(completed.finish, makespan);
  }
  return makespan;
}

vector<placement_t> single_loc_placements(graph_t const& graph) {
  return graph.make_singleton_placement();
}

equal_items_t<int> construct_equal_placements(graph_t const& graph) {
  equal_items_t<int> ret;
  construct_equal_placements_inplace(graph, ret);
  return ret;
}
void construct_equal_placements_inplace(
  graph_t const& graph,
  equal_items_t<int>& ret)
{
  // Given (up,dwn), if
  // (1) dwn is elementwise straight or a formation and
  // (2) up has no aggregation,
  // then up,dwn should have the same placement
  for(int id = 0; id != graph.nodes.size(); ++id) {
    auto const& dwn = graph.nodes[id];
    if(dwn.inns.size() == 1) {
      int const& upp_id = dwn.inns[0];
      auto const& upp = graph.nodes[upp_id];
      if(!upp.op.has_aggregation()) {
        if(dwn.op.is_einsummable()) {
          auto const& e = dwn.op.get_einsummable();
          if(e.is_straight_elementwise()) {
            ret.insert(id, upp_id);
          }
        } else if(dwn.op.is_formation()) {
          ret.insert(id, upp_id);
        }
      }
    }
  }
}

mcmc_t::mcmc_t(
  cluster_t const& cl,
  graph_t const& gr,
  double bt,
  equal_items_t<int> const& eq_pls,
  vector<placement_t> const& init_placements):
  cluster(cl), graph(gr), beta(bt),
  equal_placements(eq_pls),
  current_placements(init_placements),
  candidates(eq_pls.candidates())
{
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    if(!equal_placements.has(gid)) {
      candidates.push_back(gid);
    }
  }

  current_makespan = simulate(cluster, graph, current_placements);

  best_makespan = current_makespan;
  best_placements = current_placements;
}

mcmc_t mcmc_t::init_with_single_loc(
  cluster_t const& cluster,
  graph_t const& graph,
  double beta,
  equal_items_t<int> eqs)
{
  construct_equal_placements_inplace(graph, eqs);

  return mcmc_t(cluster, graph, beta,
    eqs,
    single_loc_placements(graph));
}

mcmc_t mcmc_t::init_balanced(
  cluster_t const& cluster,
  graph_t const& graph,
  double beta,
  equal_items_t<int> eqs)
{
  construct_equal_placements_inplace(graph, eqs);

  int ncompute = 0;
  for(auto const& d: cluster.devices) {
    ncompute += d.capacity;
  }

  uint64_t min_sizing = 1;
  vector<partition_t> parts = autopartition(graph, min_sizing, ncompute, eqs);

  vector<placement_t> pls = load_balanced_placement(
    graph, parts, cluster.devices.size(), false);

  for(auto const& gid: eqs.candidates()) {
    auto const& pl = pls[gid];
    for(auto const& other_gid: eqs.get_at(gid)) {
      if(gid != other_gid) {
        pls[other_gid] = pl;
      }
    }
  }

  return mcmc_t(cluster, graph, beta, eqs, pls);
}

bool mcmc_t::step() {
  vector<placement_t> pls = random_change();
  double makespan = simulate(cluster, graph, pls);

  if(makespan < best_makespan) {
    best_makespan = makespan;
    //DOUT(best_makespan);
    best_placements = pls;
  }

  bool accept = makespan < current_makespan;
  if(!accept) {
    double diff = current_makespan - makespan;
    double prob_accept = std::exp(beta * diff);
    accept = runif({1-prob_accept, prob_accept});
  }

  if(accept) {
    current_makespan = makespan;
    current_placements = pls;
    return true;
  } else {
    return false;
  }
}

int mcmc_t::random_gid() const {
  return candidates[runif(candidates.size())];
}

int mcmc_t::num_locs() const {
  return cluster.devices.size();
}

int mcmc_t::num_workers() const {
  int ret = 0;
  for(auto const& dev: cluster.devices) {
    ret += dev.capacity;
  }
  return ret;
}

vector<placement_t> mcmc_t::random_change() const {
  int prob_change_partition = 10;

  vector<placement_t> ret = current_placements;

  int n_locs = num_locs();
  int n_workers = num_workers();

  int gid = random_gid();
  if(n_locs == 1 || runif(100) < prob_change_partition) {
    // change the partition: either make it coarser or finer
    placement_t& pl = ret[gid];
    int n_parts = pl.partition.num_parts();
    if(n_parts == 1) {
      pl = make_finer(pl);
    } else if(n_parts > n_workers + n_workers/2) {
      pl = make_coarser(pl);
    } else {
      if(runif(2) < 1) {
        pl = make_coarser(pl);
      } else {
        pl = make_finer(pl);
      }
    }
  } else {
    placement_t& pl = ret[gid];
    vector<int>& locs = pl.locations.get();
    int bid = runif(locs.size());

    // pick any loc except locs[bid]
    int loc = runif(n_locs-1);
    if(loc >= locs[bid]) {
      loc += 1;
    }

    // pick an adjacent location
    //int loc;
    //if(locs[bid] == 0) {
    //  loc = 1;
    //} else if(locs[bid] == n_locs-1) {
    //  loc = locs[bid]-1;
    //} else {
    //  if(runif(3) < 2) {
    //    loc = locs[bid]-1;
    //  } else {
    //    loc = locs[bid]+1;
    //  }
    //}

    locs[bid] = loc;
  }

  if(equal_placements.has(gid)) {
    for(int const& other_gid: equal_placements.get_at(gid)) {
      if(gid != other_gid) {
        ret[other_gid] = ret[gid];
      }
    }
  }

  return ret;
}

placement_t mcmc_t::make_finer(placement_t const& pl) {
  auto const& part = pl.partition;
  auto blk_shape = part.block_shape();
  vector<int> can_ds;
  for(int d = 0; d != blk_shape.size(); ++d) {
    auto const& pd = part.partdims[d];
    auto szs = pd.sizes();
    uint64_t sz = *std::min_element(szs.begin(), szs.end());
    if(sz >= 2) {
      can_ds.push_back(d);
    }
  }

  int d = can_ds[runif(can_ds.size())];
  auto new_partdims = part.partdims;
  new_partdims[d] = partdim_t::split_each(new_partdims[d], 2);
  partition_t new_partition(new_partdims);

  vtensor_t<int> const& locs = pl.locations;
  vtensor_t<int> new_locs(new_partition.block_shape());
  vector<int> index(blk_shape.size(), 0);
  do {
    int const& loc = locs.at(index);
    vector<int> new_index = index;

    new_index[d] *= 2;
    new_locs.at(new_index) = loc;
    new_index[d] += 1;
    new_locs.at(new_index) = loc;
  } while(increment_idxs(blk_shape, index));

  return placement_t(new_partition, new_locs);
}

placement_t mcmc_t::make_coarser(placement_t const& pl) {
  auto const& part = pl.partition;
  auto blk_shape = part.block_shape();
  vector<int> can_ds;
  for(int d = 0; d != blk_shape.size(); ++d) {
    auto const& pd = part.partdims[d];
    if(pd.num_parts() % 2 == 0) {
      can_ds.push_back(d);
    }
  }

  int d = can_ds[runif(can_ds.size())];
  auto new_partdims = part.partdims;
  {
    vector<uint64_t> new_sizes;
    auto const& pd = new_partdims[d];
    auto sizes = pd.sizes();
    int nr = pd.num_parts() / 2;
    for(int r = 0; r != nr; ++r) {
      new_sizes.push_back(sizes[2*r] + sizes[2*r+1]);
    }
    new_partdims[d] = partdim_t::from_sizes(new_sizes);
  }
  partition_t new_partition(new_partdims);

  auto new_blk_shape = new_partition.block_shape();
  vtensor_t<int> new_locs(new_blk_shape);

  vector<int> index(new_blk_shape.size(), 0);

  vtensor_t<int> const& locs = pl.locations;
  do {
    vector<int> old_index = index;

    old_index[d] *= 2;
    int loc1 = locs.at(old_index);
    old_index[d] += 1;
    int loc2 = locs.at(old_index);

    if(runif(2)){
      new_locs.at(index) = loc1;
    } else {
      new_locs.at(index) = loc2;
    }
  } while(increment_idxs(new_blk_shape, index));

  return placement_t(new_partition, new_locs);
}

