#include "autoplace.h"

double simulate(
  cluster_t const& cluster,
  graph_t const& graph,
  vector<placement_t> const& pls)
{
  forward_state_t state(cluster, graph);

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& pl = pls[gid];
    state.assign_partition(gid, pl.partition);
    vector<int> const& locs = pl.locations.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      state.assign_location({gid, bid}, locs[bid]);
    }
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
  vector<placement_t> pls;
  pls.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    pls.emplace_back(partition_t::singleton(node.op.shape()));
  }
  return pls;
}

mcmc_t::mcmc_t(
  cluster_t const& cl,
  graph_t const& gr,
  double bt,
  vector<placement_t> placements):
  cluster(cl), graph(gr), beta(bt), current_placements(placements)
{
  current_makespan = simulate(cluster, graph, current_placements);

  best_makespan = current_makespan;
  best_placements = current_placements;
}

mcmc_t::mcmc_t(
  cluster_t const& cl,
  graph_t const& gr,
  double bt)
  : mcmc_t(cl, gr, bt, single_loc_placements(gr))
{}

bool mcmc_t::step() {
  vector<placement_t> pls = random_change();
  double makespan = simulate(cluster,  graph, pls);

  if(makespan < best_makespan) {
    best_makespan = makespan;
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

vector<placement_t> mcmc_t::random_change() const {
  int prob_change_partition = 10;

  vector<placement_t> ret = current_placements;

  int n_locs = cluster.devices.size();
  int n_gids = graph.nodes.size();
  if(runif(100) < prob_change_partition) {
    // change the partition: either make it coarser or finer
    placement_t& pl = ret[runif(n_gids)];
    int n_parts = pl.partition.num_parts();
    if(n_parts == 1) {
      pl = make_finer(pl);
    } else if(n_parts > n_locs + n_locs/2) {
      pl = make_coarser(pl);
    } else {
      if(runif(2) < 1) {
        pl = make_coarser(pl);
      } else {
        pl = make_finer(pl);
      }
    }
  } else {
    placement_t& pl = ret[runif(n_gids)];
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

  tensor_t<int> const& locs = pl.locations;
  tensor_t<int> new_locs(new_partition.block_shape());
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
  tensor_t<int> new_locs(new_blk_shape);

  vector<int> index(new_blk_shape.size(), 0);

  tensor_t<int> const& locs = pl.locations;
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

