#include "relationwise.h"
#include "../base/copyregion.h"

int64_t einsummable_cost(einsummable_t const& e) {
  int64_t ret = 1;
  for(uint64_t const& n: e.join_shape) {
    ret *= int64_t(n);
  }
  return ret;
}

int64_t einsummable_cost(optional<einsummable_t> const& maybe) {
  if(maybe) {
    return einsummable_cost(maybe.value());
  }
  return 0;
}

relationwise_t::relationwise_t(
  int nls,
  graph_t const& g,
  vector<placement_t> const& pls)
  : nlocs(nls), graph(g)
{
  std::function<partition_t const&(int)> get_partition =
    [&pls](int gid) -> partition_t const&
  {
    return pls[gid].partition;
  };

  auto get_refinement_partition = f_get_refinement_partition();
  auto get_refis = f_get_mutable_refis();

  ginfos.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    placement_t const& placement = pls[gid];

    bool has_refinement = graph.nodes[gid].outs.size() > 0;

    ginfos.push_back(ginfo_t {
      .partition = placement.partition,
      .joins =
        twolayer_construct_joins(graph, gid, placement.partition),
      .locations = placement.locations.get(),
      .refinement_partition =
        has_refinement                                                        ?
        optional<partition_t>(
          twolayer_construct_refinement_partition(graph, gid, get_partition)) :
        std::nullopt                                                          ,
      .refis = std::nullopt,
      .compute_cost = {},
      .move_cost = {}
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

  // add all the join and move costs
  // add all the move costs
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    reset_compute_cost(gid);
    reset_move_cost(gid);
  }
}

tuple<int64_t, int64_t> relationwise_t::operator()(jid_t jid, int loc)
{
  auto const& [gid, bid] = jid;
  auto& ginfo = ginfos[gid];
  auto& join = ginfo.joins[bid];
  auto& join_loc = ginfo.locations[bid];

  if(join_loc == loc) {
    // nothing is being changed!
    return {0, 0};
  }

  int64_t join_cost = einsummable_cost(join.einsummable);

  // compute the change in the compute cost & update ginfo.compute_cost
  int64_t compute_delta;
  {
    int64_t compute_before = vector_max_element(ginfo.compute_cost);
    ginfo.compute_cost[join_loc] -= join_cost;
    ginfo.compute_cost[loc]      += join_cost;
    int64_t compute_after = vector_max_element(ginfo.compute_cost);
    compute_delta = compute_after - compute_before;
  }

  // the corresponding ginfos that may have different move costs:
  //   this node, the gids that have an agg that use this join bid

  // compute the move cost before & subtract from all the corresponding
  // ginfos move_cost

  int64_t move_before = 0;

  move_before += vector_max_element(ginfo.move_cost);
  for(auto const& refi_bid: join.outs) {
    vector_sub_into(
      ginfo.move_cost,
      move_cost_at({gid, refi_bid}));
  }

  map<int, set<int>> inn_gid_to_refi_bids;
  for(auto const& [inn_gid, inn_refi_bid]: join.deps) {
    inn_gid_to_refi_bids[inn_gid].insert(inn_refi_bid);
  }

  for(auto const& [inn_gid, inn_refis]: inn_gid_to_refi_bids) {
    auto& inn_ginfo = ginfos[inn_gid];
    move_before += vector_max_element(inn_ginfo.move_cost);
    for(auto const& inn_refi_bid: inn_refis) {
      vector_sub_into(
        inn_ginfo.move_cost,
        move_cost_at({inn_gid, inn_refi_bid}));
    }
  }

  // update the join loc
  join_loc = loc;

  // now that the join loc has been updated, add to all the corresponding
  // ginfos move cost

  int64_t move_after = 0;

  for(auto const& refi_bid: join.outs) {
    vector_add_into(
      ginfo.move_cost,
      move_cost_at({gid, refi_bid}));
  }
  move_after += vector_max_element(ginfo.move_cost);

  for(auto const& [inn_gid, inn_refis]: inn_gid_to_refi_bids) {
    auto& inn_ginfo = ginfos[inn_gid];
    for(auto const& inn_refi_bid: inn_refis) {
      vector_add_into(
        inn_ginfo.move_cost,
        move_cost_at({inn_gid, inn_refi_bid}));
    }
    move_after += vector_max_element(inn_ginfo.move_cost);
  }

  int64_t move_delta = move_after - move_before;

  return {compute_delta, move_delta};
}

tuple<int64_t, int64_t>
relationwise_t::operator()(int gid, partition_t const& new_partition)
{
  auto const& ginfo = ginfos[gid];

  if(new_partition == ginfo.partition) {
    return {0, 0};
  }

  // A location for each block in the partition must be chosen,
  // then dispatch to the placement overload of this method
  //
  // To pick a location, just go with the first for each
  auto new_block_shape = new_partition.block_shape();
  vector<int> new_locations(product(new_block_shape), -1);
  copyregion_full_t copyregion(ginfo.partition, new_partition);
  do {
    int const& idx_new = copyregion.idx_bb;
    if(new_locations[idx_new] != -1) {
      int const& idx_old = copyregion.idx_aa;
      new_locations[idx_new] = ginfo.locations[idx_old];
    }
  } while(copyregion.increment());

  return this->operator()(gid,
    placement_t(
      new_partition,
      vtensor_t<int>(new_block_shape, new_locations)));
}

tuple<int64_t, int64_t>
relationwise_t::operator()(int gid, placement_t const& new_placement)
{
  ginfo_t& ginfo = ginfos[gid];

  if(new_placement.partition == ginfo.partition) {
    // In this case, only locations are changing, so offload to the method
    // that only changes a location as that should be faster
    auto const& new_locs = new_placement.locations.get();

    int64_t compute_delta = 0;
    int64_t move_delta = 0;

    for(int bid = 0; bid != ginfo.locations.size(); ++bid) {
      auto [cd, md] = this->operator()(jid_t { gid, bid }, new_locs[bid]);
      compute_delta += cd;
      move_delta += md;
    }

    return {compute_delta, move_delta};
  }

  set<int> inn_gids = graph.nodes[gid].get_inns_set();

  int64_t compute_cost_before = vector_max_element(ginfo.compute_cost);
  int64_t move_cost_before = vector_max_element(ginfo.move_cost);
  for(auto const& inn_gid: inn_gids) {
    move_cost_before += vector_max_element(ginfos[inn_gid].move_cost);
  }

  //////////

  // add this j
  ginfo.partition = new_placement.partition;
  ginfo.joins = twolayer_construct_joins(graph, gid, ginfo.partition);
  ginfo.locations = new_placement.locations.get();

  // rewire this r
  if(ginfo.has_refinement()) {
    twolayer_erase_refi_deps(ginfo.refis.value());
    twolayer_connect_join_to_refi(
      graph, gid, ginfo.joins, ginfo.partition,
      ginfo.refis.value(), ginfo.refinement_partition.value());
  }

  set<int> all_out_joins;
  for(int const& inn_gid: inn_gids) {
    // collect the join gids whose deps have changed
    set<int> const& inn_outs = graph.nodes[inn_gid].outs;
    all_out_joins.insert(inn_outs.begin(), inn_outs.end());

    ginfo_t& inn_ginfo = ginfos[inn_gid];

    // recompute the refinement partition
    inn_ginfo.refinement_partition = twolayer_construct_refinement_partition(
      graph, inn_gid, f_get_partition());

    // delete inn's r
    twolayer_erase_join_outs(inn_ginfo.joins);

    // rewrite inn's j-r
    inn_ginfo.refis = twolayer_construct_refis_and_connect_joins(
      graph, inn_gid,
      inn_ginfo.joins, inn_ginfo.partition,
      inn_ginfo.refinement_partition.value());
  }

  for(int const& join_gid: all_out_joins) {
    ginfo_t& join_ginfo = ginfos[join_gid];

    // remove join deps
    twolayer_erase_join_deps(join_ginfo.joins);
    // reinsert the join deps
    twolayer_insert_join_deps(
      graph, join_gid, join_ginfo.joins, join_ginfo.partition,
      f_get_refinement_partition());
    // call insert_refi_outs_from_join_deps
    twolayer_insert_refi_outs_from_join_deps(
      graph, join_gid, join_ginfo.joins, f_get_mutable_refis());
  }

  // the compute cost here
  reset_compute_cost(gid);

  // all the new move costs
  reset_move_cost(gid);
  for(int const& inn_gid: inn_gids) {
    reset_move_cost(inn_gid);
  }

  //////////

  int64_t compute_cost_after = vector_max_element(ginfo.compute_cost);
  int64_t move_cost_after = vector_max_element(ginfo.move_cost);
  for(auto const& inn_gid: inn_gids) {
    move_cost_after += vector_max_element(ginfos[inn_gid].move_cost);
  }

  return {
    compute_cost_after - compute_cost_before,
    move_cost_after - move_cost_before
  };
}

vector<placement_t> relationwise_t::get_placements() const
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

vector<int64_t> relationwise_t::move_cost_at(rid_t rid) const
{
  auto const& [gid,bid] = rid;
  auto const& ginfo = ginfos[gid];
  auto const& refi = ginfos[gid].refis.value()[bid];

  vector<char> dst_locs(nlocs, 0);
  for(auto const& [out_gid, out_bid]: refi.outs) {
    dst_locs[ginfos[out_gid].locations[out_bid]] = 1;
  }

  vector<int64_t> ret(nlocs, 0);
  vector<char> src_locs(nlocs);
  for(auto const& [sz, deps]: refi.units) {
    std::fill(src_locs.begin(), src_locs.end(), 0);
    for(auto const& inn_bid: deps) {
      src_locs[ginfo.locations[inn_bid]] = 1;
    }

    for(int src = 0; src != nlocs; ++src) {
      if(src_locs[src]) {
        for(int dst = 0; dst != nlocs; ++dst) {
          if(dst_locs[dst] && src != dst) {
            ret[dst] += sz;
          }
        }
      }
    }
  }

  // Here is another, most likely slower implementation
  //
  //  set<int> dst_locs;
  //  for(auto const& [out_gid, out_bid]: refi.outs) {
  //    dst_locs.insert(ginfos[out_gid].locations[out_bid]);
  //  }
  //
  //  vector<int64_t> ret(nlocs, 0);
  //  for(auto const& [sz, deps]: refi.units) {
  //    set<int> src_locs;
  //    for(auto const& inn_bid: deps) {
  //      src_locs.insert(ginfo.locations[inn_bid]);
  //    }
  //
  //    for(int const& src: src_locs) {
  //      for(int const& dst: dst_locs) {
  //        if(src != dst) {
  //          ret[dst] += sz;
  //        }
  //      }
  //    }
  //  }

  return ret;
}

tuple<int64_t, int64_t> relationwise_t::total_cost() const {
  int64_t compute = 0;
  int64_t move = 0;
  for(auto const& ginfo: ginfos) {
    compute += vector_max_element(ginfo.compute_cost);
    move += vector_max_element(ginfo.move_cost);
  }
  return {compute, move};
}

std::function<partition_t const&(int)>
relationwise_t::f_get_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).partition;
  };
}

std::function<partition_t const&(int)>
relationwise_t::f_get_refinement_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).refinement_partition.value();
  };
}

std::function<vector<refinement_t>&(int)>
relationwise_t::f_get_mutable_refis() {
  return [this](int gid) -> vector<refinement_t>& {
    return ginfos.at(gid).refis.value();
  };
}

void relationwise_t::reset_compute_cost(int gid) {
  ginfo_t& ginfo = ginfos[gid];
  ginfo.compute_cost = vector<int64_t>(nlocs, 0);
  for(int bid = 0; bid != ginfo.joins.size(); ++bid) {
    int const& loc = ginfo.locations[bid];
    int64_t join_cost = einsummable_cost(ginfo.joins[bid].einsummable);
    ginfo.compute_cost[loc] += join_cost;
  }
}

void relationwise_t::reset_move_cost(int gid) {
  ginfo_t& ginfo = ginfos[gid];
  if(ginfo.has_refinement()) {
    ginfo.move_cost = vector<int64_t>(nlocs, 0);
    int nbids = ginfo.refis.value().size();
    for(int bid = 0; bid != nbids; ++bid) {
      vector_add_into(
        ginfo.move_cost,
        move_cost_at(rid_t { gid, bid }));
    }
  } else {
    ginfo.move_cost = vector<int64_t>(nlocs, 0);
  }
}



