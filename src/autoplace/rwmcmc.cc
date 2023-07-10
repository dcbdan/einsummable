#include "rwmcmc.h"

relationwise_mcmc_t::relationwise_mcmc_t(
  graph_t const& graph,
  int nlocs,
  int max_blocks,
  double scale_compute,
  double scale_move,
  equal_items_t<int> equal_gids)
  : gwise(nlocs, graph, graph.make_singleton_placement()),
    max_blocks(max_blocks), scale_compute(scale_compute), scale_move(scale_move),
    equal_gids(equal_gids)
{
  auto [compute_cost, move_cost] = gwise.total_cost();
  current_compute = compute_cost;
  current_move = move_cost;

  best_cost = cost();
  //DOUT("init cost " << best_cost);

  best_placements = gwise.get_placements();
}

bool relationwise_mcmc_t::step(double beta) {
  op_t op   = random_op();
  op_t undo = reverse(op);
  double prev_cost = cost();
  change(op);
  double current_cost = cost();
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
        // TODO
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
    update_cost(gwise(c.gid, pp));
  } else if(op.is_set_directly()) {
    int64_t cd = 0;
    int64_t md = 0;
    for(auto const& [gid, pl]: op.get_set_directly().items) {
      auto [cd_, md_] = gwise(gid, pl);
      cd += cd_;
      md += md_;
    }
    update_cost(cd, md);
  } else {
    throw std::runtime_error("rwmcmc: missing case in change");
  }
}

relationwise_mcmc_t::op_t
relationwise_mcmc_t::random_op() const
{
  int prob_change_partition = 90;
  int gid = runif(gwise.ginfos.size());
  if(runif(100) < prob_change_partition) {
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
        update_cost(gwise(jid, loc));
        double current_cost = cost();
        if(current_cost < best) {
          best = current_cost;
          best_loc = loc;
        }
      }
    }
    update_cost(gwise(jid, best_loc));
  }
}

void relationwise_mcmc_t::update_cost(
  int64_t compute_delta, int64_t move_delta)
{
  current_compute += compute_delta;
  current_move += move_delta;
}

void relationwise_mcmc_t::update_cost(tuple<int64_t, int64_t> delta)
{
  return update_cost(std::get<0>(delta), std::get<1>(delta));
}
double relationwise_mcmc_t::cost_from_scores(
  int64_t compute_cost, int64_t move_cost) const
{
  return compute_cost * scale_compute + move_cost * scale_move;
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

