#include "relationwise.h"
#include "../base/copyregion.h"
#include "../einsummable/taskgraph.h" // double_last_dim_inplace

kernel_coster_t
kernel_coster_t::for_cpu_cluster(int nlocs)
{
  // 100 Mbps=1e8
  double bw = 1e9;

  double rw = 1e9;

  double fl = 1e10;

  double startup = 5e-4; // this many seconds to start doing anything

  return kernel_coster_t {
    .bandwidths = vector<vector<double>>(nlocs, vector<double>(nlocs, bw)),
    .flops = fl,
    .rw = rw,
    .compute_start = startup,
    .touch_start = startup,
    .move_start = startup
  };
}

double kernel_coster_t::compute(einsummable_t const& e) const {
  uint64_t elem = e.out_nelem();
  for(auto const& inn_shape: e.inn_shapes()) {
    elem += product(inn_shape);
  }
  return compute_start + \
    double(elem) / rw + \
    double(product(e.join_shape)) / flops;
}

double kernel_coster_t::move(uint64_t n, int src, int dst) const {
  return move_start + double(n) / bandwidths[src][dst];
}

double kernel_coster_t::touch(uint64_t n) const {
  return touch_start + double(n) / flops;
}

threads_costs_t::threads_costs_t(int n_threads)
  : max_cost(0), cnt(0), n_threads(n_threads)
{}

void threads_costs_t::add(double cost) {
  cnt++;
  max_cost = std::max(cost, max_cost);
}

void threads_costs_t::pop(double cost) {
  if(cost > max_cost) {
    throw std::runtime_error("invalid cost to pop");
  }
  cnt--;
  if(cnt == 0) {
    max_cost = 0.0;
  }
}

double threads_costs_t::cost() const {
  return ((cnt + n_threads - 1) / n_threads) * max_cost;
}

void threads_costs_t::clear() {
  max_cost = 0.0;
  cnt = 0;
}

void relationwise_stat_t::print_line(std::ostream& out) const {
  out << "no_touch:"   << double(num_no_touch) / double(num_can_no_touch) << ","
      << "join:"       << total_join                    << ","
      << "touch:"      << total_touch                   << ","
      << "move:"       << total_move                    << ","
      << "non-c avg blks:" << double(non_contraction.num_blocks) / double(non_contraction.num) << ","
      << "non-c avg locs:" << double(non_contraction.num_locs  ) / double(non_contraction.num) << ","
      << "contr avg blks:" << double(contraction.num_blocks)     / double(contraction.num)     << ","
      << "contr avg locs:" << double(contraction.num_locs  )     / double(contraction.num)     << ","
          << std::endl;
}

relationwise_t::relationwise_t(
  int nls,
  int ntp,
  graph_t const& g,
  kernel_coster_t const& kc,
  vector<placement_t> const& pls)
  : nlocs(nls), n_threads_per_loc(ntp), kernel_coster(kc), graph(g)
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
      .join_cost      = vector<threads_costs_t>(nlocs, threads_costs_t(n_threads_per_loc)),
      .touch_src_cost = vector<threads_costs_t>(nlocs, threads_costs_t(n_threads_per_loc)),
      .move_cost      = vector<double>(nlocs, 0.0),
      .touch_dst_cost = vector<threads_costs_t>(nlocs, threads_costs_t(n_threads_per_loc))
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
  reset_cost();
}

void relationwise_t::reset_cost() {
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    reset_join_cost(gid);
    reset_refi_cost(gid);
  }
}

double relationwise_t::ginfo_t::total_join_cost() const {
  return vector_max_method(join_cost, cost);
}
double relationwise_t::ginfo_t::total_refi_cost() const {
  return vector_max_method(touch_src_cost, cost) +
         vector_max_element(move_cost) +
         vector_max_method(touch_dst_cost, cost);
}

double relationwise_t::operator()(jid_t jid, int loc)
{
  auto const& [gid, bid] = jid;
  auto& ginfo = ginfos[gid];
  auto& join = ginfo.joins[bid];
  auto& join_loc = ginfo.locations[bid];

  if(join_loc == loc) {
    // nothing is being changed!
    return 0.0;
  }

  // compute the change in the compute cost & update ginfo.join_cost
  double join_delta = 0.0;
  if(join.einsummable) { // bool(join.einsummable) iff gid node is einsummable iff
                         //                            has_join_cost(gid)
    double j_cost = kernel_coster.compute(join.einsummable.value());
    double join_before = ginfo.total_join_cost();
    ginfo.join_cost[join_loc].pop(j_cost);
    ginfo.join_cost[loc].add(j_cost);
    double join_after = ginfo.total_join_cost();
    join_delta = join_after - join_before;
  }

  // the corresponding ginfos that may have different refi costs:
  //   this node, the gids that have an agg that use this join bid

  // compute the refi cost before & subtract from all the corresponding
  // ginfos refi cost terms

  double refi_before = 0.0;

  refi_before += ginfo.total_refi_cost();
  for(auto const& refi_bid: join.outs) {
    sub_refi_cost_at({gid, refi_bid});
  }

  map<int, set<int>> inn_gid_to_refi_bids;
  for(auto const& [inn_gid, inn_refi_bid]: join.deps) {
    inn_gid_to_refi_bids[inn_gid].insert(inn_refi_bid);
  }

  for(auto const& [inn_gid, inn_refis]: inn_gid_to_refi_bids) {
    auto& inn_ginfo = ginfos[inn_gid];
    refi_before += inn_ginfo.total_refi_cost();
    for(auto const& inn_refi_bid: inn_refis) {
      sub_refi_cost_at({inn_gid, inn_refi_bid});
    }
  }

  // update the join loc
  join_loc = loc;

  // now that the join loc has been updated, add to all the corresponding
  // ginfos refis costs

  double refi_after = 0;

  for(auto const& refi_bid: join.outs) {
    add_refi_cost_at({gid, refi_bid});
  }
  refi_after += ginfo.total_refi_cost();

  for(auto const& [inn_gid, inn_refis]: inn_gid_to_refi_bids) {
    auto& inn_ginfo = ginfos[inn_gid];
    for(auto const& inn_refi_bid: inn_refis) {
      add_refi_cost_at({inn_gid, inn_refi_bid});
    }
    refi_after += inn_ginfo.total_refi_cost();
  }

  double refi_delta = refi_after - refi_before;

  return join_delta + refi_delta;
}

double
relationwise_t::operator()(int gid, partition_t const& new_partition)
{
  auto const& ginfo = ginfos[gid];

  if(new_partition == ginfo.partition) {
    return 0.0;
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
    if(new_locations[idx_new] == -1) {
      int const& idx_old = copyregion.idx_aa;
      new_locations[idx_new] = ginfo.locations[idx_old];
    }
  } while(copyregion.increment());

  return this->operator()(gid,
    placement_t(
      new_partition,
      vtensor_t<int>(new_block_shape, new_locations)));
}

double
relationwise_t::operator()(int gid, placement_t const& new_placement)
{
  ginfo_t& ginfo = ginfos[gid];

  if(new_placement.partition == ginfo.partition) {
    // In this case, only locations are changing, so offload to the method
    // that only changes a location as that should be faster
    auto const& new_locs = new_placement.locations.get();

    double delta = 0.0;

    for(int bid = 0; bid != ginfo.locations.size(); ++bid) {
      delta += this->operator()(jid_t { gid, bid }, new_locs[bid]);
    }

    return delta;
  }

  set<int> inn_gids = graph.nodes[gid].get_inns_set();

  double join_cost_before = ginfo.total_join_cost();
  double refi_cost_before = ginfo.total_refi_cost();
  for(auto const& inn_gid: inn_gids) {
    refi_cost_before += ginfos[inn_gid].total_refi_cost();
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
  reset_join_cost(gid);

  // all the new refi costs
  reset_refi_cost(gid);
  for(int const& inn_gid: inn_gids) {
    reset_refi_cost(inn_gid);
  }

  //////////

  double join_cost_after = ginfo.total_join_cost();
  double refi_cost_after = ginfo.total_refi_cost();;
  for(auto const& inn_gid: inn_gids) {
    refi_cost_after += ginfos[inn_gid].total_refi_cost();
  }

  return (join_cost_after - join_cost_before) +
         (refi_cost_after - refi_cost_before) ;
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

relationwise_stat_t relationwise_t::make_stat() const {
  relationwise_stat_t ret {
    .num_no_touch = 0,
    .num_can_no_touch = 0,
    .total_join = 0,
    .total_touch = 0,
    .total_move = 0,
    .non_contraction = { .num = 0, .num_blocks = 0, .num_locs = 0 },
    .contraction     = { .num = 0, .num_blocks = 0, .num_locs = 0 },
  };

  for(int id = 0; id != graph.nodes.size(); ++id) {
    auto const& node = graph.nodes[id];
    auto const& ginfo = ginfos[id];
    if(ginfo.has_refinement()) {
      ret.num_can_no_touch++;
      if(has_no_touch(id)) {
        ret.num_no_touch++;
      }
    }
    ret.total_join += vector_max_method(ginfo.join_cost, cost);
    ret.total_touch += vector_max_method(ginfo.touch_src_cost, cost);
    ret.total_touch += vector_max_method(ginfo.touch_dst_cost, cost);
    ret.total_move += vector_max_element(ginfo.move_cost);

    int num_blocks = ginfo.partition.num_parts();
    set<int> locs(ginfo.locations.begin(), ginfo.locations.end());

    bool is_contraction = [&] {
      if(node.op.is_einsummable()) {
        auto const& e = node.op.get_einsummable();
        return e.is_contraction();
      }
      return false;
    }();

    if(is_contraction) {
      ret.contraction.num += 1;
      ret.contraction.num_blocks += num_blocks;
      ret.contraction.num_locs += locs.size();
    } else {
      ret.non_contraction.num += 1;
      ret.non_contraction.num_blocks += num_blocks;
      ret.non_contraction.num_locs += locs.size();
    }
  }

  return ret;
}

placement_t relationwise_t::get_placement_at(int gid) const
{
  ginfo_t const& ginfo = ginfos[gid];
  return placement_t(
    ginfo.partition,
    vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
}

bool relationwise_t::has_no_touch(int gid) const {
  auto const& ginfo = ginfos[gid];
  if(!ginfo.refinement_partition) {
    return true;
  }

  auto const& partition = ginfo.partition;
  auto const& refi_partition = ginfo.refinement_partition.value();

  // Note that the refi_partition is with respect to the real
  // and the partition is with respect to the output dtype
  if(dtype_is_complex(graph.out_dtype(gid))) {
    int out_rank = refi_partition.partdims.size();
    auto iter = partition.partdims.begin();
    partition_t out_partition(vector<partdim_t>(iter, iter + out_rank));
    double_last_dim_inplace(out_partition);
    return out_partition == refi_partition;
  } else {
    int out_rank = refi_partition.partdims.size();
    if(out_rank == partition.partdims.size()) {
      return partition == refi_partition;
    } else {
      auto iter = partition.partdims.begin();
      partition_t out_partition(vector<partdim_t>(iter, iter + out_rank));
      return out_partition == refi_partition;
    }
  }
}

optional<partition_t>
relationwise_t::notouch_partition(int gid) const
{
  auto const& ginfo = ginfos[gid];

  if(!ginfo.refinement_partition) {
    return ginfo.partition;
  }

  // dtype issues
  // agg issues

  auto const& refi_partition = ginfos[gid].refinement_partition.value();
  auto const& refi_pds = refi_partition.partdims;

  partition_t ret = ginfo.partition;
  std::copy(refi_pds.begin(), refi_pds.end(), ret.partdims.begin());

  if(dtype_is_complex(graph.out_dtype(gid))) {
    int out_rank = refi_pds.size();
    partdim_t& pd = ret.partdims[out_rank-1];
    auto szs = pd.sizes();
    for(auto const& sz: szs) {
      if(sz % 2 != 0) {
        return std::nullopt;
      }
    }
    pd = partdim_t::from_sizes(vector_halve(szs));
  }

  return ret;
}

double relationwise_t::total_cost() const {
  double ret;
  for(auto const& ginfo: ginfos) {
    ret += ginfo.total_cost();
  }
  return ret;
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

void relationwise_t::reset_join_cost(int gid) {
  if(!has_join_cost(gid)) {
    return;
  }
  ginfo_t& ginfo = ginfos[gid];
  vector_domethod(ginfo.join_cost, clear);
  for(int bid = 0; bid != ginfo.joins.size(); ++bid) {
    int const& loc = ginfo.locations[bid];
    double join_cost = kernel_coster.compute(ginfo.joins[bid].einsummable.value());
    ginfo.join_cost[loc].add(join_cost);
  }
}

void relationwise_t::reset_refi_cost(int gid) {
  ginfo_t& ginfo = ginfos[gid];

  if(ginfo.has_refinement()) {
    vector_domethod(ginfo.touch_src_cost, clear);
    std::fill(ginfo.move_cost.begin(), ginfo.move_cost.end(), 0.0);
    vector_domethod(ginfo.touch_dst_cost, clear);

    int nbids = ginfo.refis.value().size();
    for(int bid = 0; bid != nbids; ++bid) {
      add_refi_cost_at(rid_t { gid, bid });
    }
  } else {
    // the costs should have been zero before and they should
    // still be zero
  }
}

void relationwise_t::_change_refi_cost_at(rid_t rid, bool add)
{
  // TODO: This code is a hot spot. It may be faster to cache
  //       src_locs, dst_locs or both instead of recomputing
  //       them each time. Or maybe the whole thing doesn't need
  //       to be computed.
  auto const& [gid,bid] = rid;
  auto&       ginfo = ginfos[gid];
  auto const& refi = ginfos[gid].refis.value()[bid];

  auto& touch_src_cost = ginfo.touch_src_cost;
  auto& move_cost = ginfo.move_cost;
  auto& touch_dst_cost = ginfo.touch_src_cost;

  bool no_touching = has_no_touch(gid);

  auto update_touch = [&add, &no_touching](threads_costs_t& t, double c) {
    if(no_touching) { return; }

    if(add) {
      t.add(c);
    } else {
      t.pop(c);
    }
  };
  auto update_move = [&add](double& t, double c) {
    if(add) {
      t += c;
    } else {
      t -= c;
    }
  };

  vector<char> dst_locs(nlocs, 0);
  for(auto const& [out_gid, out_bid]: refi.outs) {
    dst_locs[ginfos[out_gid].locations[out_bid]] = 1;
  }

  vector<char> src_locs(nlocs);
  for(auto const& [sz, deps]: refi.units) {
    double t_cost = kernel_coster.touch(sz);
    std::fill(src_locs.begin(), src_locs.end(), 0);
    for(auto const& inn_bid: deps) {
      src_locs[ginfo.locations[inn_bid]] = 1;
    }

    for(int src = 0; src != nlocs; ++src) {
      if(src_locs[src]) {
        update_touch(touch_src_cost[src], t_cost);
        for(int dst = 0; dst != nlocs; ++dst) {
          if(dst_locs[dst] && src != dst) {
            update_move(move_cost[dst], kernel_coster.move(sz, src, dst));
            update_touch(touch_src_cost[dst], t_cost);
          }
        }
      }
    }
  }

  // Here is another, most likely slower implementation
  //
  //set<int> dst_locs;
  //for(auto const& [out_gid, out_bid]: refi.outs) {
  //  dst_locs.insert(ginfos[out_gid].locations[out_bid]);
  //}

  //for(auto const& [sz, deps]: refi.units) {
  //  double t_cost = kernel_coster.touch(sz);

  //  set<int> src_locs;
  //  for(auto const& inn_bid: deps) {
  //    src_locs.insert(ginfo.locations[inn_bid]);
  //  }

  //  for(int const& src: src_locs) {
  //    update_touch(touch_src_cost[src], t_cost);
  //    for(int const& dst: dst_locs) {
  //      if(src != dst) {
  //        update_move(move_cost[dst], kernel_coster.move(sz, src, dst));
  //        update_touch(touch_src_cost[dst], t_cost);
  //      }
  //    }
  //  }
  //}
}


