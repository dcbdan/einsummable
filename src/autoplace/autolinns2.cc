#include "autolinns2.h"

bool site_costs_lt(
  vector<uint64_t> const& lhs,
  vector<uint64_t> const& rhs)
{
  uint64_t const& mxl = *std::max_element(lhs.begin(), lhs.end());
  uint64_t const& mxr = *std::max_element(rhs.begin(), rhs.end());
  if(mxl == mxr) {
    return vector_sum(lhs) < vector_sum(rhs);
  } else {
    return mxl < mxr;
  }

  //int nlocs = lhs.size();

  //uint64_t mxl = nlocs * (*std::max_element(lhs.begin(), lhs.end()));
  //uint64_t mxr = nlocs * (*std::max_element(rhs.begin(), rhs.end()));

  //uint64_t ssl = vector_sum(lhs);
  //uint64_t ssr = vector_sum(rhs);

  //uint64_t alpha = 1;
  //uint64_t beta  = 1;

  //uint64_t score_l = alpha * mxl + beta * ssl;
  //uint64_t score_r = alpha * mxr + beta * ssr;

  //return score_l < score_r;
};

relationwise3_t::relationwise3_t(
  int nls,
  uint64_t fpbm,
  graph_t const& g,
  vector<partition_t> const& parts)
  : nlocs(nls), flops_per_byte_moved(fpbm), graph(g)
{
  std::function<partition_t const&(int)> get_partition =
    [&parts](int gid) -> partition_t const&
  {
    return parts[gid];
  };

  auto get_refinement_partition = f_get_refinement_partition();
  auto get_refis = f_get_mutable_refis();

  ginfos.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    partition_t const& part = parts[gid];
    int nblocks = part.num_parts();

    // Note: all locations are initially filled to negative 1

    ginfos.push_back(ginfo_t {
      .partition = part,
      .joins =
        twolayer_construct_joins(graph, gid, part),
      .locations = vector<int>(nblocks, -1),
      .rinfo = std::nullopt
    });

    ginfo_t& ginfo = ginfos.back();

    twolayer_insert_join_deps(
      graph, gid, ginfo.joins, ginfo.partition, get_refinement_partition);

    bool has_refinement = graph.nodes[gid].outs.size() > 0;
    if(has_refinement) {
      ginfo.rinfo = rinfo_t {
        .partition =
          twolayer_construct_refinement_partition(graph, gid, get_partition),
        .refis = {},
        .locations = {}
      };

      rinfo_t& rinfo = ginfo.rinfo.value();

      rinfo.refis = twolayer_construct_refis_and_connect_joins(
        graph, gid, ginfo.joins, ginfo.partition, rinfo.partition);

      rinfo.locations = vector<set<int>>(rinfo.partition.num_parts());
    }

    twolayer_insert_refi_outs_from_join_deps(
      graph, gid, ginfo.joins, get_refis);
  }
}

vector<placement_t> relationwise3_t::get_placements() const
{
  vector<placement_t> ret;
  ret.reserve(ginfos.size());
  for(auto const& ginfo: ginfos) {
    ret.emplace_back(
      ginfo.partition,
      vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
  }
  return ret;
}

placement_t relationwise3_t::get_placement_at(int gid) const
{
  ginfo_t const& ginfo = ginfos[gid];
  return placement_t(
    ginfo.partition,
    vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
}

std::function<partition_t const&(int)>
relationwise3_t::f_get_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).partition;
  };
}

std::function<partition_t const&(int)>
relationwise3_t::f_get_refinement_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).rinfo.value().partition;
  };
}

std::function<vector<refinement_t>&(int)>
relationwise3_t::f_get_mutable_refis() {
  return [this](int gid) -> vector<refinement_t>& {
    return ginfos.at(gid).rinfo.value().refis;
  };
}

void relationwise3_t::update_cost(
  vector<uint64_t>& cost,
  optional<einsummable_t> const& maybe,
  int loc) const
{
  if(maybe) {
    auto const& e = maybe.value();
    cost[loc] += product(e.join_shape);
  }
}

void relationwise3_t::update_cost(
  vector<uint64_t>& cost,
  uint64_t bytes,
  int src,
  int dst) const
{
  uint64_t move_cost = bytes * flops_per_byte_moved;

  // Seems best not to penalize the sender; Just penalizing the recver here {{{
  cost[dst] += move_cost;
  // }}}
}

void relationwise3_t::update(builder_t const& builder) {
  for(auto const& [jid, loc]: builder.join_locs) {
    get_join_loc(jid) = loc;
  }
  for(auto const& [rid, locs]: builder.refi_locs) {
    get_refi_locs(rid).insert(locs.begin(), locs.end());
  }
}

builder_t::builder_t(relationwise3_t const& rw_)
  : rw(rw_), site_costs(rw.nlocs, 0)
{}

builder_t::builder_t(
  relationwise3_t const& rw_, vector<uint64_t> const& scs)
  : rw(rw_), site_costs(scs)
{}

builder_t::builder_t(builder_t const& other)
  : site_costs(other.site_costs),
    join_locs(other.join_locs),
    refi_locs(other.refi_locs),
    rw(other.rw)
{}

builder_t& builder_t::operator=(builder_t const& other) {
  if(this == &other) {
    return *this;
  }
  if(&rw != &other.rw) {
    throw std::runtime_error("incompatiable builders");
  }

  site_costs = other.site_costs;
  join_locs = other.join_locs;
  refi_locs = other.refi_locs;

  return *this;
}

optional<int> builder_t::get_join_loc( jid_t const& jid) const {
  auto iter = join_locs.find(jid);
  if(iter == join_locs.end()) {
    int ret = rw.get_join_loc(jid);
    if(ret < 0) {
      return std::nullopt;
    } else {
      return ret;
    }
  } else {
    return iter->second;
  }
}

set<int> builder_t::get_refi_locs(rid_t const& rid) const {
  set<int> rw_locs = rw.get_refi_locs(rid);
  auto iter = refi_locs.find(rid);
  if(iter != refi_locs.end()) {
    auto const& new_locs = iter->second;
    rw_locs.insert(new_locs.begin(), new_locs.end());
  }
  return rw_locs;
}

int builder_t::jid_at(jid_t const& jid, int loc, bool assert_) {
  optional<int> maybe = get_join_loc(jid);
  if(maybe) {
    if(assert_) {
      if(maybe.value() != loc) {
        throw std::runtime_error("build jid_at assert tripped");
      }
    }
    return maybe.value();
  }

  // Ok, this join has not been set; make sure all the refis end up
  // at this location
  join_t const& join = get_join(jid);
  for(auto const& rid: join.deps) {
    rid_at(rid, loc);
  }

  // Now that we have the data, we must do the computation
  rw.update_cost(site_costs, join.einsummable, loc);

  // And we must set the loc
  join_locs.insert({jid, loc});

  return loc;
}

void builder_t::rid_at(rid_t const& rid, int loc) {
  if(get_refi_locs(rid).count(loc) > 0) {
    // we already are at this location, so nothing to do
    return;
  }

  refinement_t const& refi = get_refi(rid);
  for(agg_unit_t const& unit: refi.units) {
    set<int> srcs;
    for(int const& bid: unit.deps) {
      jid_t jid { .gid = rid.gid, .bid = bid };
      int join_loc = jid_at(jid, loc);
      if(join_loc != loc) {
        srcs.insert(join_loc);
      }
    }
    // we must move this agg to this location.
    for(int const& src: srcs) {
      rw.update_cost(site_costs, unit.size, src, loc);
    }
  }

  // all the data has been aggregated to loc, so update
  refi_locs[rid].insert(loc);
}

builder_t
solve_refi_at(
  relationwise3_t const& rw,
  vector<uint64_t> init_cost,
  rid_t const& rid,
  int loc)
{
  if(rw.get_refi_locs(rid).count(loc) > 0) {
    return builder_t(rw, init_cost);
  }

  refinement_t const& refi = rw.get_refi(rid);

  // for any joins that have already been computed,
  // add move costs we know will already happen.

  vector<set<int>> agg_unit_srcs(refi.units.size());
  for(int which_unit = 0; which_unit != refi.units.size(); ++which_unit) {
    auto const& unit = refi.units[which_unit];

    set<int>& srcs = agg_unit_srcs[which_unit];
    for(int const& bid: unit.deps) {
      int maybe = rw.get_join_loc(jid_t{ rid.gid, bid });
      if(maybe >= 0 && maybe != loc) {
        srcs.insert(maybe);
      }
    }

    for(int const& src: srcs) {
      rw.update_cost(init_cost, unit.size, src, loc);
    }
  }

  // for each join dependency, try all the locations and pick the best
  builder_t ret(rw, init_cost);
  for(int which_unit = 0; which_unit != refi.units.size(); ++which_unit) {
    auto const& unit = refi.units[which_unit];
    set<int>& srcs = agg_unit_srcs[which_unit];

    for(int const& bid: unit.deps) {
      jid_t jid { .gid = rid.gid, .bid = bid };
      if(bool(ret.get_join_loc(jid))) {
        continue;
      }

      int which_join_loc;
      optional<builder_t> best_builder;
      for(int join_loc = 0; join_loc != rw.nlocs; ++join_loc) {
        builder_t builder = ret;

        // do the join at join_loc, recursively doing wtvr necc
        int _actual_join_loc = builder.jid_at(jid, join_loc);
        if(join_loc != _actual_join_loc) {
          throw std::runtime_error("should be prevented");
        }

        // if a move will have to occur from join_loc, incur the cost now
        if(join_loc != loc && srcs.count(join_loc) == 0) {
          srcs.insert(join_loc);
          rw.update_cost(builder.site_costs, unit.size, join_loc, loc);
        }

        if(!bool(best_builder) ||
           site_costs_lt(builder.site_costs, best_builder.value().site_costs))
        {
          which_join_loc = join_loc;
          best_builder = builder;
        }
      }

      ret = best_builder.value();
    }
  }

  // we've done all this extra work to put the rid at this loc,
  // so make sure to update the builder accordingly
  ret.refi_locs[rid].insert(loc);

  return ret;
}

builder_t
solve_refi(
  relationwise3_t const& rw,
  vector<uint64_t> const& init_cost,
  rid_t const& rid)
{
  if(rw.get_refi_locs(rid).size() > 0) {
    throw std::runtime_error("this refi has already been solved");
  }

  int which_loc;
  optional<builder_t> ret;
  for(int loc = 0; loc != rw.nlocs; ++loc) {
    builder_t b = solve_refi_at(rw, init_cost, rid, loc);
    if(!bool(ret) || site_costs_lt(b.site_costs, ret.value().site_costs)) {
      which_loc = loc;
      ret = b;
    }
  }

  DOUT("solving refi " << rid);
  for(int i = 0; i != 4; ++i) {
    rid_t rid0{0,i};
    set<int> locs_ = ret.value().get_refi_locs(rid0);
    vector<int> locs(locs_.begin(), locs_.end());
    DOUT("  " << rid0 << ": " << locs);
  }
  //for(auto const& [rid, loc]: ret.value().refi_locs) {
  //  if(rid.gid == 0) {
  //    DOUT("  " << rid << ": " << vector<int>(loc.begin(), loc.end()));
  //  }
  //}

  return ret.value();
}

builder_t
solve_join(
  relationwise3_t const& rw,
  vector<uint64_t> const& init_cost,
  jid_t const& jid)
{
  optional<builder_t> ret;
  for(int loc = 0; loc != rw.nlocs; ++loc) {
    builder_t b(rw, init_cost);
    b.jid_at(jid, loc);

    uint64_t cost = b.max_site_cost();
    if(!bool(ret) ||
       site_costs_lt(b.site_costs, ret.value().site_costs))
    {
      ret = b;
    }
  }

  return ret.value();
}

vector<placement_t> autolocate_bipartite(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved)
{
  relationwise3_t rw(nlocs, flops_per_byte_moved, graph, parts);

  vector<uint64_t> costs(rw.nlocs, 0);
  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      continue;
    }

    if(rw.has_refinement(gid) && node.op.is_formation()) {
      continue;
    }

    if(rw.has_refinement(gid)) {
      int num_refis = rw.get_num_refis(gid);
      for(int bid = 0; bid != num_refis; ++bid) {
        builder_t b = solve_refi(rw, costs, rid_t{ .gid = gid, .bid = bid});
        costs = b.site_costs;
        rw.update(b);
      }

      if(node.outs.size() == 1) {
        int const& out_gid = *node.outs.begin();
        auto const& out_node = graph.nodes[out_gid];
        if(out_node.op.is_formation()) {
          vector<int>& locs = rw.ginfos[out_gid].locations;
          for(int bid = 0; bid != locs.size(); ++bid) {
            int loc = rw.get_singleton_refi_loc(rid_t{ gid, bid });
            locs[bid] = loc;
          }
        }
      }
    } else {
      int num_joins = rw.get_num_joins(gid);
      for(int bid = 0; bid != num_joins; ++bid) {
        builder_t b = solve_join(rw, costs, jid_t{ .gid = gid, .bid = bid});
        costs = b.site_costs;
        rw.update(b);
      }
    }
  }

  // TODO: remove this check
  auto ret = rw.get_placements();
  for(auto const& pl: ret) {
    for(auto const& loc: pl.locations.get()) {
      if(loc < 0) {
        throw std::runtime_error("not all locations have been set");
      }
    }
  }

  // TODO: remove this print
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& pl = ret[gid];
    vector<int> cnts(nlocs, 0);
    for(int const& loc: pl.locations.get()) {
      cnts[loc]++;
    }
    DOUT(gid << ": locs = " << pl.locations.get());
    //DOUT(gid << ": " << cnts); // pl.locations.get());
  }

  return ret;
}

//////////////////////////////////////////////////////////////////////
#include "loadbalanceplace.h" // aggplan
#include "alocate.h" // _get_possible_placements

void solve_from_agg_plans(
  builder_t& ret,
  int gid,
  int num_out_blocks,
  int num_agg,
  bool has_formation)
{
  auto const& rw = ret.rw;

  for(int out_bid = 0; out_bid != num_out_blocks; ++out_bid) {
    optional<builder_t> best_agg_builder;
    for(agg_plan_t const& plan: gen_agg_plans(rw.nlocs, num_agg)) {
      // agg_loc isn't used if there is no formation!
      int eloc = has_formation ? plan.eloc : (plan.bloc + 1);
      for(int agg_loc = plan.bloc; agg_loc != eloc; ++agg_loc) {
        builder_t agg_builder = ret;

        for(int which_agg = 0; which_agg != num_agg; ++which_agg) {
          int bid = out_bid * num_agg + which_agg;
          int join_loc = plan.loc_at(which_agg);
          agg_builder.jid_at(jid_t{ gid, bid }, join_loc, true);
        }
        if(has_formation) {
          agg_builder.rid_at(rid_t { gid, out_bid }, agg_loc);
        }

        if(!bool(best_agg_builder) ||
           site_costs_lt(agg_builder.site_costs, best_agg_builder.value().site_costs))
        {
          best_agg_builder = agg_builder;
        }
      }
    }
    ret = best_agg_builder.value();
  }
}

vector<vector<tuple<int, placement_t>>>
gen_inn_plans(
  relationwise3_t const& rw,
  set<int> const& inns_)
{
  vector<int> inn_gids(inns_.begin(), inns_.end());

  vector<vector<placement_t>> all_pls;
  for(auto const& inn: inn_gids) {
    all_pls.push_back(_get_possible_placements(rw.nlocs, rw.ginfos[inn].partition));
  }
  vector<int> shape;
  for(auto const& pls: all_pls) {
    shape.push_back(pls.size());
  }

  vector<vector<tuple<int, placement_t>>> ret;
  vector<int> idxs(inn_gids.size(), 0);
  do {
    ret.emplace_back();
    auto& v = ret.back();
    for(int which_inn = 0; which_inn != idxs.size(); ++which_inn) {
      int const& inn_gid = inn_gids[which_inn];
      auto const& pls = all_pls[which_inn];
      int const& which_pl = idxs[which_inn];
      placement_t const& pl = pls[which_pl];
      v.emplace_back(inn_gid,pl);
    }
  } while(increment_idxs(shape, idxs));

  return ret;
}

vector<placement_t> autolocate_agg_at_a_time_from_inns_v2(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved)
{
  relationwise3_t rw(nlocs, flops_per_byte_moved, graph, parts);

  vector<uint64_t> site_costs(rw.nlocs, 0);
  set<int> seen_inn_gids;
  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];

    if(node.op.is_input()) {
      continue;
    }

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

    set<int> new_inn_gids;
    for(auto const& inn_gid: node.inns) {
      auto const& node = graph.nodes[inn_gid];
      if(node.op.is_input() && seen_inn_gids.count(inn_gid) == 0) {
        new_inn_gids.insert(inn_gid);
      }
    }

    if(new_inn_gids.size() > 0) {
      optional<builder_t> best_builder;
      auto inn_plans = gen_inn_plans(rw, new_inn_gids);
      for(vector<tuple<int, placement_t>> const& inn_pls: inn_plans) {
        builder_t builder(rw, site_costs);
        for(auto const& [inn_gid, inn_pl]: inn_pls) {
          vector<int> const& locs = inn_pl.locations.get();
          for(int bid = 0; bid != locs.size(); ++bid) {
            builder.jid_at(jid_t{ inn_gid, bid }, locs[bid], true);
          }
        }

        solve_from_agg_plans(
          builder, gid, num_out_blocks, num_agg, bool(out_formation_gid));

        if(!bool(best_builder) ||
           site_costs_lt(builder.site_costs, best_builder.value().site_costs))
        {
          best_builder = builder;
        }
      }
      rw.update(best_builder.value());
      site_costs = best_builder.value().site_costs;

      for(auto const& inn_gid: new_inn_gids) {
        seen_inn_gids.insert(inn_gid);
      }
    } else {
      builder_t builder(rw, site_costs);
      solve_from_agg_plans(
        builder, gid, num_out_blocks, num_agg, bool(out_formation_gid));
      rw.update(builder);
      site_costs = builder.site_costs;
    }

    if(out_formation_gid) {
      int const& out_gid = out_formation_gid.value();
      vector<int>& out_locs = rw.ginfos[out_gid].locations;
      for(int bid = 0; bid != out_locs.size(); ++bid) {
        out_locs[bid] = rw.get_singleton_refi_loc(rid_t{ gid, bid });
      }
    }
  }

  // TODO: remove this check
  auto ret = rw.get_placements();
  for(auto const& pl: ret) {
    for(auto const& loc: pl.locations.get()) {
      if(loc < 0) {
        throw std::runtime_error("not all locations have been set");
      }
    }
  }

  // TODO: remove this print
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& pl = ret[gid];
    vector<int> cnts(nlocs, 0);
    for(int const& loc: pl.locations.get()) {
      cnts[loc]++;
    }
    DOUT(gid << ": " << cnts); // pl.locations.get());
  }

  return ret;
}

