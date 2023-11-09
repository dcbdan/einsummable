#include "autolinns.h"

relationwise2_t::relationwise2_t(
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

    bool has_refinement = graph.nodes[gid].outs.size() > 0;

    // Note: all locations are initially filled to negative 1

    ginfos.push_back(ginfo_t {
      .partition = part,
      .joins =
        twolayer_construct_joins(graph, gid, part),
      .locations = vector<int>(nblocks, -1),
      .refinement_partition =
        has_refinement                                                        ?
        optional<partition_t>(
          twolayer_construct_refinement_partition(graph, gid, get_partition)) :
        std::nullopt                                                          ,
      .refis = std::nullopt
    });

    ginfo_t& ginfo = ginfos.back();

    twolayer_insert_join_deps(
      graph, gid, ginfo.joins, ginfo.partition, get_refinement_partition);

    if(ginfo.has_refinement()) {
      ginfo.refis = twolayer_construct_refis_and_connect_joins(
        graph, gid, ginfo.joins, ginfo.partition, ginfo.refinement_partition.value());
    }

    twolayer_insert_refi_outs_from_join_deps(
      graph, gid, ginfo.joins, get_refis);
  }
}

vector<placement_t> relationwise2_t::get_placements() const
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

placement_t relationwise2_t::get_placement_at(int gid) const
{
  ginfo_t const& ginfo = ginfos[gid];
  return placement_t(
    ginfo.partition,
    vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
}

std::function<partition_t const&(int)>
relationwise2_t::f_get_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).partition;
  };
}

std::function<partition_t const&(int)>
relationwise2_t::f_get_refinement_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).refinement_partition.value();
  };
}

std::function<vector<refinement_t>&(int)>
relationwise2_t::f_get_mutable_refis() {
  return [this](int gid) -> vector<refinement_t>& {
    return ginfos.at(gid).refis.value();
  };
}

int relationwise2_t::num_agg_blocks_at(int gid) const {
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];
  int out_rank = node.op.out_rank();
  partition_t agg_partition(vector<partdim_t>(
    ginfo.partition.partdims.begin() + out_rank,
    ginfo.partition.partdims.end()));
  return agg_partition.num_parts();
}

set<int> relationwise2_t::get_refi_usage_locs(rid_t const& rid) const {
  auto const& [gid, bid] = rid;
  auto const& ginfo = ginfos[gid];
  refinement_t const& refi = ginfo.refis.value()[bid];

  set<int> ret;
  for(auto const& jid: refi.outs) {
    auto const& [join_gid, join_bid] = jid;
    auto const& join_ginfo = ginfos[join_gid];
    int const& loc = join_ginfo.locations[join_bid];
    if(loc != -1) {
      ret.insert(loc);
    }
  }
  return ret;
}

uint64_t relationwise2_t::get_refi_bytes(rid_t const& rid) const {
  auto const& [gid, bid] = rid;
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];

  auto dtype = node.op.out_dtype();
  uint64_t dsz = dtype_size(dtype);
  if(dtype_is_complex(dtype)) {
    dsz *= 2;
  }

  uint64_t nelem = ginfo.refinement_partition.value().block_size_at_bid(bid);
  return nelem * dsz;
}

uint64_t relationwise2_t::get_join_out_bytes(jid_t const& jid) const {
  auto const& [gid, bid] = jid;
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];

  auto dtype = node.op.out_dtype();
  uint64_t dsz = dtype_size(dtype);
  if(dtype_is_complex(dtype)) {
    dsz *= 2;
  }

  auto const& join_part = ginfo.partition;
  auto shape = join_part.tensor_shape_at(join_part.from_bid(bid));
  shape.resize(node.op.out_rank());
  uint64_t nelem = product(shape);

  return nelem * dsz;
}

vector<uint64_t> relationwise2_t::cost_agg_plan(
  int gid,
  int out_bid,
  agg_plan_t const& plan,
  int agg_loc) const
{
  vector<uint64_t> ret(nlocs, 0);

  auto const& ginfo = ginfos[gid];

  int nagg = num_agg_blocks_at(gid);

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
        set<int> refi_usage_locs = get_refi_usage_locs(rid);
        if(refi_usage_locs.count(l) != 0) {
          // this input was already at the site from a previous
          // set location
          continue;
        }
        vector<int> const& join_locs = ginfos[rid.gid].locations;
        auto const& refi = get_refi(rid);
        for(auto const& unit: refi.units) {
          if(unit.deps.size() > 1) {
            throw std::runtime_error("aggs should have been taken care of");
          }
          int const& join_bid = unit.deps[0];
          int const& join_loc = join_locs[join_bid];
          if(join_loc != l) {
            ret[l] += unit.size;
          }
        }
      }
    }
  }

  // add the cost to agg from site
  {
    uint64_t bytes = get_join_out_bytes(jid_t { .gid = gid, .bid = out_bid*nagg });

    set<int> srcs;
    for(int i = 0; i != nagg; ++i) {
      int bid = out_bid*nagg + i;
      int l = plan.loc_at(i);
      srcs.insert(l);
    }

    for(auto const& src: srcs) {
      if(src != agg_loc) {
        // way 1: the recving side incurs the cost
        //ret[agg_loc] += flops_per_byte_moved*bytes;

        // way 2: the sending side incurs the cost
        ret[src] += flops_per_byte_moved*bytes;
      }
    }
  }

  return ret;
}

void relationwise2_t::print_info() const
{
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    std::cout << "gid: " << gid << std::endl;
    auto const& ginfo = ginfos[gid];
    int nbid = ginfo.locations.size();
    for(int bid = 0; bid != nbid; ++bid) {
      auto const& join = ginfo.joins[bid];
      vector<rid_t> rids(join.deps.begin(), join.deps.end());
      std::cout << "   J " << bid << ": " << rids << std::endl;
    }
    if(ginfo.refis) {
      auto const& refis = ginfo.refis.value();
      for(int bid = 0; bid != refis.size(); ++bid) {
        auto const& refi = refis[bid];
        for(auto const& unit: refi.units) {
          std::cout << "  R " << bid << ": " << unit.deps << std::endl;
        }
      }
    }
  }
}

vector<placement_t> autolocate_agg_at_a_time_from_inns(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved)
{
  relationwise2_t rw(nlocs, flops_per_byte_moved, graph, parts);

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
      vector<int>& locs = rw.ginfos[gid].locations;
      for(int bid = 0; bid != locs.size(); ++bid) {
        locs[bid] = bid % nlocs;
      }

      if(out_formation_gid) {
        int const& out_gid = out_formation_gid.value();
        vector<int>& out_locs = rw.ginfos[out_gid].locations;
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
      vector<int>& locs = rw.ginfos[gid].locations;
      for(int i = 0; i != num_agg; ++i) {
        int bid = out_bid*num_agg + i;
        locs[bid] = best_plan.loc_at(i);
      }
      if(out_formation_gid) {
        int const& out_gid = out_formation_gid.value();
        vector<int>& out_locs = rw.ginfos[out_gid].locations;
        out_locs[out_bid] = best_agg_loc;
      }
    }
  }

  return rw.get_placements();
}
