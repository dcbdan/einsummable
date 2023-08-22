#include "rwmcmc.h"

relationwise_mcmc_t::relationwise_mcmc_t(
  graph_t const& graph,
  kernel_coster_t const& kernel_coster,
  int nlocs,
  int n_threads_per_loc,
  int max_blocks,
  equal_items_t<int> equal_gids)
  : gwise(nlocs, n_threads_per_loc, graph, kernel_coster, graph.make_singleton_placement()),
    max_blocks(max_blocks), equal_gids(equal_gids)
{
  current_cost = gwise.total_cost();
  best_cost = current_cost;

  //DOUT("init cost " << best_cost);

  best_placements = gwise.get_placements();
}

bool relationwise_mcmc_t::step(double beta) {
  op_t op   = random_op();
  op_t undo = reverse(op);
  double prev_cost = cost();
  change(op);
  current_cost = cost();
  if(current_cost <= prev_cost) {
    if(current_cost < best_cost) {
      best_cost = current_cost;
      //DOUT("best cost " << best_cost);
      best_placements = gwise.get_placements();
    }
    return true;
  } else {
    double prob_accept = std::exp(beta * (prev_cost - current_cost));
    //DOUT("delta " << prev_cost - current_cost << "   | prob accept" << prob_accept);
    bool accept = runif({1-prob_accept, prob_accept});
    if(accept) {
      return true;
    } else {
      change(undo);
      return false;
    }
  }
}

void relationwise_mcmc_t::set_placements(vector<placement_t> const& pls)
{
  if(pls.size() != gwise.graph.nodes.size()) {
    throw std::runtime_error("must specify every placement");
  }

  for(int gid = 0; gid != pls.size(); ++gid) {
    gwise(gid, pls[gid]);
  }

  gwise.reset_cost();
  current_cost = gwise.total_cost();

  //DOUT("set cost " << current_cost);
  if(current_cost < best_cost) {
    best_cost = current_cost;
    best_placements = pls;
  }
}

int relationwise_mcmc_t::op_t::candidate_gid() const {
  if(is_greedy()) {
    return get_greedy().gid;
  }
  if(is_set_directly()) {
    throw std::runtime_error("set_directly has no candidate");
  }
  if(is_crement()) {
    return get_crement().gid;
  }
  if(is_set_notouch()) {
    return get_set_notouch().gid;
  }
  throw std::runtime_error("candidate_gid should not reach");
}

void relationwise_mcmc_t::change(relationwise_mcmc_t::op_t const& op) {
  if(op.is_greedy()) {
    int const& candidate_gid = op.get_greedy().gid;
    if(equal_gids.has(candidate_gid)) {
      set<int> const& gids = equal_gids.get_at(candidate_gid);
      if(gids.size() == 1) {
        greedy_solve(candidate_gid);
      } else {
        throw std::runtime_error("not implemented: placement constraints");
      }
    } else {
      greedy_solve(candidate_gid);
    }
  } else if(op.is_crement()) {
    auto const& c = op.get_crement();
    if(equal_gids.has(c.gid)) {
      set<int> const& gids = equal_gids.get_at(c.gid);
      if(gids.size() != 1) {
        // TODO
        throw std::runtime_error("not implemented: placement constraints");
      }
    }
    partition_t pp = crement_partition(c);
    current_cost += gwise(c.gid, pp);
  } else if(op.is_set_notouch()) {
    auto const& gid = op.get_set_notouch().gid;
    if(equal_gids.has(gid)) {
      set<int> const& gids = equal_gids.get_at(gid);
      if(gids.size() != 1) {
        // TODO
        throw std::runtime_error("not implemented: placement constraints");
      }
    }
    partition_t pp = notouch_partition(gid).value();
    current_cost += gwise(gid, pp);
  } else if(op.is_set_directly()) {
    int64_t cd = 0;
    int64_t md = 0;
    for(auto const& [gid, pl]: op.get_set_directly().items) {
      current_cost += gwise(gid, pl);
    }
  } else {
    throw std::runtime_error("rwmcmc: missing case in change");
  }
}

relationwise_mcmc_t::op_t
relationwise_mcmc_t::random_op() const
{
  int prob_change_partition = 80;
  int gid = runif(gwise.ginfos.size());
  if(runif(100) < prob_change_partition) {
    int prob_set_notouch = 10;
    bool can_no_touch = bool(gwise.ginfos[gid].refinement_partition);
    if(can_no_touch && runif(100) < prob_set_notouch) {
      auto maybe_pp = notouch_partition(gid);
      if(!maybe_pp || (maybe_pp && maybe_pp.value().num_parts() > max_blocks)) {
        return op_t(op_t::greedy_t { gid });
      } else {
        if(maybe_pp.value() == gwise.ginfos[gid].partition) {
          return op_t(op_t::greedy_t { gid });
        } else {
          return op_t(op_t::set_notouch_t { gid });
        }
      }
    } else {
      auto const& partition = gwise.ginfos[gid].partition;
      int rank = partition.partdims.size();
      int d = runif(rank);
      int nn = partition.partdims[d].spans.size();
      uint64_t total = partition.partdims[d].total();
      bool incr = (runif(5) < 3);
      if(incr) {
        int nparts = partition.num_parts();
        int nparts_after = (nparts / nn) * (nn + 1);
        if(nparts_after > max_blocks) {
          return op_t(op_t::greedy_t { gid } );
        }
        if(uint64_t(nn) == total) {
          return op_t(op_t::greedy_t { gid } );
        }
      } else {
        if(nn == 1) {
          return op_t(op_t::greedy_t { gid } );
        }
      }
      return op_t(op_t::crement_t { gid, d, incr });
    }
  } else {
    return op_t(op_t::greedy_t { gid } );
  }
}

relationwise_mcmc_t::op_t
relationwise_mcmc_t::reverse(
  relationwise_mcmc_t::op_t const& op) const
{
  vector<tuple<int, placement_t>> items;

  int candidate_gid = op.candidate_gid();
  if(equal_gids.has(candidate_gid)) {
    set<int> const& gids = equal_gids.get_at(candidate_gid);
    for(auto const& gid: gids) {
      items.emplace_back(gid, gwise.get_placement_at(gid));
    }
  } else {
    items.emplace_back(candidate_gid, gwise.get_placement_at(candidate_gid));
  }

  return op_t(op_t::set_directly_t { std::move(items) });
}

void relationwise_mcmc_t::greedy_solve(int gid) {
  int nbids = gwise.ginfos[gid].locations.size();
  for(int bid = 0; bid != nbids; ++bid) {
    jid_t jid { gid, bid };
    int orig_loc = gwise.ginfos[gid].locations[bid];
    double best = cost();
    int best_loc = orig_loc;
    for(int loc = 0; loc != gwise.nlocs; ++loc) {
      if(loc != orig_loc) {
        current_cost += gwise(jid, loc);
        double current_cost = cost();
        if(current_cost < best) {
          best = current_cost;
          best_loc = loc;
        }
      }
    }
    current_cost += gwise(jid, best_loc);
  }
}

partition_t
relationwise_mcmc_t::crement_partition(op_t::crement_t const& crement) const
{
  auto const& [gid, i, do_increase] = crement;
  partition_t ret(gwise.ginfos[gid].partition.partdims);
  auto& pds = ret.partdims;
  int n = pds[i].num_parts();
  if(do_increase) {
    pds[i] = partdim_t::split(pds[i].total(), n+1);
  } else {
    if(n == 1) {
      throw std::runtime_error("cannot decrement this partdim");
    }
    pds[i] = partdim_t::split(pds[i].total(), n-1);
  }
  if(ret.num_parts() > max_blocks) {
    throw std::runtime_error(
      "crement_partition returned partition has more than max blocks");
  }
  return ret;
}

optional<partition_t>
relationwise_mcmc_t::notouch_partition(int gid) const
{
  return gwise.notouch_partition(gid);
}


