#include "alocate.h"

#include "autoplace.h" // equal_holder_t

struct _alocate01_rw_t {
  _alocate01_rw_t(
    int nlocs,
    uint64_t flops_per_byte_moved,
    graph_t const& graph,
    vector<partition_t> const& parts,
    bool set_inputs_everywhere = true);

  vector<int> const&
  locations(int gid) const {
    return _rw.ginfos[gid].locations;
  }
  vector<int>&
  locations(int gid) {
    return _rw.ginfos[gid].locations;
  }

  vector<placement_t> get_placements() const {
    return _rw.get_placements();
  }

  vector<uint64_t> cost_agg_plan(
    int gid,
    int out_bid,
    agg_plan_t const& plan,
    int agg_loc) const;

private:
  int const nlocs;
  uint64_t const flops_per_byte_moved;
  bool set_inputs_everywhere;
  relationwise_t _rw;
};

vector<placement_t> alocate01(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved)
{
  _alocate01_rw_t rw(nlocs, flops_per_byte_moved, graph, parts);

  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_formation()) {
      if(node.inns.size() != 1) {
        throw std::runtime_error("a formation must have one input");
      }
      continue;
    }

    int join_rank = node.op.rank();
    int out_rank = node.op.out_rank();

    optional<int> out_formation_gid;
    if(node.outs.size() == 1) {
      int const& out_gid = *node.outs.begin();
      auto const& out_node = graph.nodes[out_gid];
      if(out_node.op.is_formation()) {
        out_formation_gid = out_gid;
      }
    } else {
      for(int const& out_gid: node.outs) {
        auto const& out_node = graph.nodes[out_gid];
        if(out_node.op.is_formation()) {
          throw std::runtime_error("a formation must be a single out");
        }
      }
    }

    int num_out_blocks;
    int num_agg;
    if(out_formation_gid) {
      auto const& join_part = parts[gid];
      partition_t out_part(vector<partdim_t>(
        join_part.partdims.begin(),
        join_part.partdims.begin() + out_rank));
      if(out_part != parts[out_formation_gid.value()]) {
        throw std::runtime_error("formation node incorrect partition");
      }
      num_out_blocks = out_part.num_parts();
      num_agg = join_part.num_parts() / num_out_blocks;
    } else if(out_rank != join_rank) {
      throw std::runtime_error("has agg must have out formation");
    } else {
      auto const& join_part = parts[gid];
      num_out_blocks = join_part.num_parts();
      num_agg = 1;
    }

    if(node.op.is_input()) {
      // Note: an input node may have a formation output!
      vector<int>& locs = rw.locations(gid);
      for(int bid = 0; bid != locs.size(); ++bid) {
        locs[bid] = bid % nlocs;
      }

      if(out_formation_gid) {
        int const& out_gid = out_formation_gid.value();
        vector<int>& out_locs = rw.locations(out_gid);
        out_locs = locs;
      }

      continue;
    }

    vector<uint64_t> cost_per_sites(nlocs, 0);
    for(int out_bid = 0; out_bid != num_out_blocks; ++out_bid) {
      agg_plan_t best_plan;
      int best_agg_loc;
      vector<uint64_t> best_cost_per_sites;
      optional<uint64_t> best_cost;

      for(agg_plan_t const& plan: gen_agg_plans(nlocs, num_agg)) {
        for(int agg_loc = plan.bloc; agg_loc != plan.eloc; ++agg_loc) {
          vector<uint64_t> just_agg_costs =
            rw.cost_agg_plan(gid, out_bid, plan, agg_loc);
          vector<uint64_t> new_cost_per_sites =
            vector_add(just_agg_costs, cost_per_sites);
          uint64_t cost = *std::max_element(
            new_cost_per_sites.begin(),
            new_cost_per_sites.end());
          if(!bool(best_cost) || cost < best_cost.value()) {
            best_cost = cost;
            best_plan = plan;
            best_agg_loc = agg_loc;
            best_cost_per_sites = new_cost_per_sites;
          }
        }
      }

      cost_per_sites = best_cost_per_sites;
      vector<int>& locs = rw.locations(gid);
      for(int i = 0; i != num_agg; ++i) {
        int bid = out_bid*num_agg + i;
        locs[bid] = best_plan.loc_at(i);
      }
      if(out_formation_gid) {
        int const& out_gid = out_formation_gid.value();
        vector<int>& out_locs = rw.locations(out_gid);
        out_locs[out_bid] = best_agg_loc;
      }
    }
  }
  return rw.get_placements();
}

_alocate01_rw_t::_alocate01_rw_t(
  int nls,
  uint64_t f,
  graph_t const& graph,
  vector<partition_t> const& parts,
  bool s)
  : nlocs(nls), flops_per_byte_moved(f),
    set_inputs_everywhere(s),
    _rw(graph, parts)
{}

vector<uint64_t> _alocate01_rw_t::cost_agg_plan(
  int gid,
  int out_bid,
  agg_plan_t const& plan,
  int agg_loc) const
{
  vector<uint64_t> ret(nlocs, 0);

  auto const& ginfos = _rw.ginfos;
  auto const& graph  = _rw.graph;

  auto const& ginfo = ginfos[gid];

  int nagg = _rw.num_agg_blocks_at(gid);

  // add the compute cost at each location
  for(int i = 0; i != nagg; ++i) {
    int bid = out_bid*nagg + i;
    auto const& join = ginfo.joins[bid];
    if(join.einsummable) {
      auto const& e = join.einsummable.value();
      int l = plan.loc_at(i);
      ret[l] += product(e.join_shape);
    }
  }

  // add the cost to move inputs into site
  {
    vector<set<rid_t>> inns_per_loc(nlocs);
    for(int i = 0; i != nagg; ++i) {
      int l = plan.loc_at(i);

      int bid = out_bid*nagg + i;
      auto const& join = ginfo.joins[bid];

      inns_per_loc[l].insert(join.deps.begin(), join.deps.end());
    }

    for(int l = 0; l != nlocs; ++l) {
      set<rid_t> const& rids = inns_per_loc[l];

      for(auto const& rid: rids) {
        auto const& inn_node = graph.nodes[rid.gid];
        if(inn_node.op.is_input() && set_inputs_everywhere) {
          // we assume it is free for all inputs to be everywhere
          continue;
        }

        set<int> refi_usage_locs = _rw.get_refi_usage_locs(rid);
        if(refi_usage_locs.count(l) != 0) {
          // this input was already at the site from a previous
          // set location
          continue;
        }
        vector<int> const& join_locs = ginfos[rid.gid].locations;
        auto const& refi = _rw.get_refi(rid);
        for(auto const& unit: refi.units) {
          if(unit.deps.size() > 1) {
            throw std::runtime_error("aggs should have been taken care of");
          }
          int const& join_bid = unit.deps[0];
          int const& join_loc = join_locs[join_bid];
          if(join_loc != l) {
            ret[l] += flops_per_byte_moved*unit.size;
          }
        }
      }
    }
  }

  // add the cost to agg from site
  {
    uint64_t bytes = _rw.get_join_out_bytes(jid_t { .gid = gid, .bid = out_bid*nagg });

    set<int> srcs;
    for(int i = 0; i != nagg; ++i) {
      int bid = out_bid*nagg + i;
      int l = plan.loc_at(i);
      srcs.insert(l);
    }

    for(auto const& src: srcs) {
      if(src != agg_loc) {
        // way 1: the recving side incurs the cost
        ret[agg_loc] += flops_per_byte_moved*bytes;

        // way 2: the sending side incurs the cost
        ret[src] += flops_per_byte_moved*bytes;

        // way 3: both..
      }
    }
  }

  return ret;
}

struct _alocate02_rw_t {
  _alocate02_rw_t(
    int nlocs,
    uint64_t flops_per_byte_moved,
    graph_t const& graph,
    vector<partition_t> const& parts);

  vector<uint64_t> move_cost(
    set<int> const& src_locs,
    set<int> const& dst_locs,
    uint64_t bytes) const;

  vector<uint64_t> refi_cost(rid_t rid) const;

  vector<uint64_t> contributing_cost(jid_t jid) const;

  void set_locs(int gid, vtensor_t<int> const& locs);

  void set_loc(jid_t jid, int loc);

  uint64_t get_cost() const { return *std::max_element(cost.begin(), cost.end()); }

  vector<placement_t> get_placements() const {
    return _rw.get_placements();
  }

private:
  int const nlocs;
  uint64_t const flops_per_byte_moved;
  relationwise_t _rw;

  vector<uint64_t> cost;
};

_alocate02_rw_t::_alocate02_rw_t(
  int nls,
  uint64_t f,
  graph_t const& graph,
  vector<partition_t> const& parts)
  : nlocs(nls), flops_per_byte_moved(f),
    cost(nlocs),
    _rw(graph, parts)
{}

vector<uint64_t> _alocate02_rw_t::move_cost(
  set<int> const& src_locs,
  set<int> const& dst_locs,
  uint64_t bytes) const
{
  vector<uint64_t> ret(nlocs);
  for(int const& src: src_locs) {
    for(int const& dst: dst_locs) {
      if(src != dst) {
        ret[src] += flops_per_byte_moved*bytes;
        ret[dst] += flops_per_byte_moved*bytes;
      }
    }
  }
  return ret;
}

vector<uint64_t> _alocate02_rw_t::refi_cost(rid_t rid) const
{
  refinement_t const& refi = _rw.get_refi(rid);

  set<int> dst_locs = _rw.get_refi_usage_locs(rid);

  vector<uint64_t> ret(nlocs);
  for(agg_unit_t const& unit: refi.units) {
    set<int> src_locs;
    for(int const& bid: unit.deps) {
      int const& loc = _rw.get_loc(jid_t{ rid.gid, bid });
      if(loc != -1) {
        src_locs.insert(loc);
      }
    }

    vector_add_into(
      ret,
      move_cost(src_locs, dst_locs, unit.size));
  }

  return ret;
}

vector<uint64_t> _alocate02_rw_t::contributing_cost(jid_t jid) const
{
  vector<uint64_t> ret(nlocs);
  int const& loc = _rw.get_loc(jid);
  if(loc == -1) {
    return ret;
  }

  join_t const& join = _rw.get_join(jid);
  if(join.einsummable) {
    int64_t join_cost = product(join.einsummable.value().join_shape);
    ret[loc] += join_cost;
  }

  for(int const& bid: join.outs) {
    vector_add_into(ret, refi_cost(rid_t { jid.gid, bid }));
  }
  for(rid_t const& rid: join.deps) {
    vector_add_into(ret, refi_cost(rid));
  }

  return ret;
}

void _alocate02_rw_t::set_locs(int gid, vtensor_t<int> const& locations)
{
  vector<int> block_shape = locations.get_shape();

  if(!vector_equal(
    block_shape,
    _rw.ginfos[gid].partition.block_shape()))
  {
    throw std::runtime_error("invalid block shapes provided!");
  }

  int nbid = product(block_shape);
  vector<int> const& locs = locations.get();
  for(int bid = 0; bid != nbid; ++bid) {
    set_loc(jid_t { gid, bid }, locs[bid]);
  }
}

void _alocate02_rw_t::set_loc(jid_t jid, int loc) {
  {
    int prev_loc = _rw.get_loc(jid);
    if(prev_loc == loc) {
      return;
    }
  }

  vector<uint64_t> prev_contrib = contributing_cost(jid);
  _rw.get_loc(jid) = loc;
  vector<uint64_t> curr_contrib = contributing_cost(jid);

  for(int i = 0; i != nlocs; ++i) {
    cost[i] += curr_contrib[i];
    cost[i] -= prev_contrib[i];
  }
}

vector<placement_t> alocate02(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved,
  map<int, vtensor_t<int>> const& fixed_pls,
  vector<tuple<int,int>> const& equal_pls)
{
  _alocate02_rw_t rw(nlocs, flops_per_byte_moved, graph, parts);

  equal_holder_t eqs(equal_pls);
  for(auto const& [id, locs]: fixed_pls) {
    if(eqs.has(id)) {
      throw std::runtime_error("move equal set to fixed...");
    }

    rw.set_locs(id, locs);
  }

  // Note: this will solve eq gids multiple times
  for(int const& base_gid: graph.get_order()) {
    if(fixed_pls.count(base_gid) > 0) {
      continue;
    }

    set<int> gids;
    if(eqs.has(base_gid)) {
      gids = eqs[base_gid];
    } else {
      gids.insert(base_gid);
    }

    int nbid = parts[base_gid].num_parts();
    for(int bid = 0; bid != nbid; ++bid) {
      auto set_loc = [&](int loc) {
        for(int const& gid: gids) {
          rw.set_loc({gid, bid}, loc);
        }
      };

      set_loc(0);
      int best_loc = 0;
      uint64_t best_cost = rw.get_cost();
      for(int loc = 1; loc != nlocs; ++loc) {
        set_loc(loc);
        uint64_t cost = rw.get_cost();
        if(cost < best_cost) {
          best_loc  = loc;
          best_cost = cost;
        }
      }

      if(best_loc != nlocs - 1) {
        set_loc(best_loc);
      }
    }
  }

  return rw.get_placements();
}

struct _alocate03_rw_t {
  _alocate03_rw_t(
    graph_t const& graph,
    vector<partition_t> const& parts,
    int nlocs, 
    bool with_goofy_topology);

  vector<int> const&
  locations(int gid) const {
    return _rw.ginfos[gid].locations;
  }
  vector<int>&
  locations(int gid) {
    return _rw.ginfos[gid].locations;
  }

  int& get_loc(jid_t const& jid) {
    return _rw.get_loc(jid);
  }

  int const& get_loc(jid_t const& jid) const {
    return _rw.get_loc(jid);
  }

  vector<placement_t> get_placements() const {
    return _rw.get_placements();
  }

  uint64_t compute_at(jid_t const& jid, int loc) const;

  uint64_t get_move_cost_multiplier(int src, int dst) const;

private:
  relationwise_t _rw;
  int nlocs;
  optional<vtensor_t<int>> topo_info;
};

_alocate03_rw_t::_alocate03_rw_t(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool with_goofy_topology)
  : _rw(graph, parts), nlocs(nlocs)
{
  if(with_goofy_topology) {
    if(nlocs > 8) {
      throw std::runtime_error("invalid with goofy topology");
    }
    topo_info = vtensor_t<int>({8,8}, 0);
    topo_info.value().get() = vector<int>{0,20,40,20,40,5,5,5,20,0,20,40,5,40,5,5,40,20,0,40,5,5,20,5,20,40,40,0,5,5,5,20,40,5,5,5,0,20,40,20,5,40,5,5,20,0,20,40,5,5,20,5,40,20,0,40,5,5,5,20,20,40,40,0}; // these are approx gb/s numbers
  }
}

uint64_t _alocate03_rw_t::get_move_cost_multiplier(int src, int dst) const
{
  if(src == dst) {
    throw std::runtime_error("can't send to self");
  }

  if(topo_info) {
    int const& gbs = topo_info.value().at({src,dst});
    if(gbs == 40) { 
      return 1;
    }
    if(gbs == 20) {
      return 2;
    }
    if(gbs == 5) {
      return 8;
    }
    throw std::runtime_error("should not reach...");
  } else {
    return 1;
  }
}

uint64_t _alocate03_rw_t::compute_at(jid_t const& jid, int this_loc) const
{
  uint64_t ret = 0;
  for(rid_t const& rid: _rw.get_join(jid).deps) {
    // this guy will need to be at the join location
  
    set<int> refi_usage_locs = _rw.get_refi_usage_locs(rid);
    if(refi_usage_locs.count(this_loc) != 0) {
      // already here, nothing to do
      continue;
    }

    vector<int> const& join_locs = _rw.ginfos[rid.gid].locations;
    auto const& refi = _rw.get_refi(rid);
    for(auto const& unit: refi.units) {
      set<int> src_locs;
      for(int const& join_bid: unit.deps) {
        int const& join_loc = join_locs[join_bid];
        if(join_loc != this_loc) {
          src_locs.insert(join_loc);
        }
      }

      // If the multiplier always equals 1...
      //   ret += src_locs.size()*unit.size;

      for(int const& src: src_locs) {
        ret += unit.size * get_move_cost_multiplier(src, this_loc);
      }
    }
  }

  return ret;
}

struct join_agg_info_t {
  uint64_t cost;
  vector<int> join_locs;
  vector<int> agg_locs;
};

vector<vector<int>> all_permutations(int nval) {
  if(nval < 0) {
    throw std::runtime_error("nval must be positive");
  }

  if(nval == 1) {
    return { {0} };
  }

  auto fix = [&](vector<int> const& xs) {
    vector<int> ret;
    vector<char> taken(xs.size() + 1, 0);
    for(int x: xs) {
      int y = 0;
      for(; y != taken.size(); ++y) {
        if(x == 0 && !taken[y]) {
          taken[y] = 1;
          break;
        }
        if(!taken[y]) {
          x--;
        }
      }
      ret.push_back(y);
    }
    for(int y = 0; y != taken.size(); ++y) {
      if(!taken[y]) {
        ret.push_back(y);
        break;
      }
    }
    return ret;
  };

  vector<int> shape;
  for(int i = nval; i >= 2; --i) {
    shape.push_back(i);
  }
  vector<int> idxs(nval-1, 0);

  vector<vector<int>> ret;
  do {
    ret.push_back(fix(idxs));
  } while(indexer_utils<int>::increment_idxs(shape, idxs));

  return ret;
}

vector<vector<int>> 
alocate03_create_join_locs(int nlocs, int n_join_bid) 
{
  if(n_join_bid < nlocs) {
    throw std::runtime_error("join bid less than number of locs");
  }

  int num_per_loc = n_join_bid / nlocs;
  if(n_join_bid % nlocs != 0) {
    throw std::runtime_error("join bid not divisible by nlocs");
  }

	if(num_per_loc != 1) {
    throw std::runtime_error("invalid: too many ops per loc!");
  }
  if(nlocs > 8) {
    throw std::runtime_error("too many permutations to choose from");
  }

  return all_permutations(nlocs);
}

vector<placement_t> alocate03(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs, 
  bool with_goofy_topology)
{
  _alocate03_rw_t rw(graph, parts, nlocs, with_goofy_topology);

  auto is_full_plan = [&](int gid) {
    auto const& op = graph.nodes[gid].op;
    if(op.is_einsummable()) {
      auto const& e = op.get_einsummable();
      if(e.is_contraction()) {
        return true;
      }
      if(e.has_aggregation() && parts[gid].num_parts() == nlocs) {
        return true;
      }
    }
    return false;
  };

  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];

    // TODO: how to deal with inputs?
    if(node.op.is_input()) {
      int nbid = parts[gid].num_parts();
      for(int bid = 0; bid != nbid; ++bid) {
        jid_t jid{ gid, bid };
        rw.get_loc(jid) = bid % nlocs;
      }

      continue;
    }

    if(node.op.is_formation()) {
      if(node.inns.size() != 1) {
        throw std::runtime_error("a formation must have one input");
      }
      int inn_gid = node.inns[0];
      if(is_full_plan(inn_gid)) {
        // when doing the input node, this node was already set
        continue;
      }
    }

    if(is_full_plan(gid))
    {
      if(node.outs.size() != 1) {
        throw std::runtime_error("must have one out");
      }

      int const& join_gid = gid;
      int        form_gid = *node.outs.begin();

      if(!graph.nodes[form_gid].op.is_formation()) {
        throw std::runtime_error("must have one formation out");
      }

      int n_join_bid = parts[join_gid].num_parts();
      int n_form_bid = parts[form_gid].num_parts();
      optional<join_agg_info_t> best;
      for(vector<int> const& join_locs: alocate03_create_join_locs(nlocs, n_join_bid)) {
        uint64_t total_cost = 0;

        // 1. set all the joins and update the total_cost
        for(int bid = 0; bid != n_join_bid; ++bid) {
          jid_t jid{ join_gid, bid };
          int const& loc = join_locs[bid];
          total_cost += rw.compute_at(jid, loc);
          rw.get_loc(jid) = loc;
        }

        // 2. set all the aggs and update the total_cost
        for(int bid = 0; bid != n_form_bid; ++bid) {
          jid_t jid{ form_gid, bid };
          int best_loc = 0;
          uint64_t best_cost = rw.compute_at(jid, 0);
          for(int loc = 1; loc != nlocs; ++loc) {
            uint64_t cost = rw.compute_at(jid, loc);
            if(cost < best_cost) {
              best_loc = loc;
              best_cost = cost;
            }
          }
          rw.get_loc(jid) = best_loc;
          total_cost += best_cost;
        }

        if(!best || total_cost < best.value().cost) {
          best = join_agg_info_t { 
            .cost = total_cost, 
            .join_locs = rw.locations(join_gid),
            .agg_locs  = rw.locations(form_gid)
          };
        }

        rw.locations(join_gid) = vector<int>(n_join_bid, -1);
        rw.locations(form_gid) = vector<int>(n_form_bid, -1);
      }

      auto const& [_, join_locs, form_locs] = best.value();
      rw.locations(join_gid) = join_locs;
      rw.locations(form_gid) = form_locs;
    } else {
      int nbid = parts[gid].num_parts();
      for(int bid = 0; bid != nbid; ++bid) {
        jid_t jid{ gid, bid };
        int best_loc = 0;
        uint64_t best_cost = rw.compute_at(jid, 0);
        for(int loc = 1; loc != nlocs; ++loc) {
          uint64_t cost = rw.compute_at(jid, loc);
          if(cost < best_cost) {
            best_loc = loc;
            best_cost = cost;
          }
        }
        rw.get_loc(jid) = best_loc;
      }
    }
  }

  return rw.get_placements();
}
