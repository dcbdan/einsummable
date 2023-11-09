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

  // Here, the send is incurring cost
  cost[src] += move_cost;
  // Here, the recv is incurring cost
  cost[dst] += move_cost;
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
  if(iter == refi_locs.end()) {
    return rw_locs;
  } else {
    auto const& new_locs = iter->second;
    rw_locs.insert(new_locs.begin(), new_locs.end());
    return iter->second;
  }
}

int builder_t::jid_at(jid_t const& jid, int loc) {
  optional<int> maybe = get_join_loc(jid);
  if(maybe) {
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
    if(rid.gid == 4) {
      DLINEOUT("with loc " << loc << ": " << b.site_costs);
    }
    if(!bool(ret) || site_costs_lt(b.site_costs, ret.value().site_costs)) {
      which_loc = loc;
      ret = b;
    }
  }

  if(rid.gid == 4) {
    DOUT("solve refi " << rid << " best loc of " << which_loc);
    {
      DOUT("  " << ret.value().site_costs);
      auto const& refi = rw.get_refi(rid);
      for(auto const& unit: refi.units) {
        for(auto const& bid: unit.deps) {
          jid_t jid { rid.gid, bid };
          DOUT("  " << jid << " at " << ret.value().get_join_loc(jid).value())
        }
      }
    }
  }

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
    DOUT(gid << ": " << pl.locations.get());
  }

  return ret;
}

