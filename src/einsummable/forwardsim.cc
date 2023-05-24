#include "forwardsim.h"
#include "copyregion.h"

forward_state_t::forward_state_t(cluster_t const& c, graph_t const& g)
  : cluster(c), graph(g),
    ginfos(g.nodes.size()),
    apply_workers(c.devices.size()),
    move_workers(c.connections.size()),
    to_move_worker(cluster.to_connection),
    num_join_remaining(0),
    time(0.0)
{
  vector<int> cs(g.nodes.size());
  std::iota(cs.begin(), cs.end(), 0);
  can_partition.insert(cs.begin(), cs.end());
}

forward_state_t::unit_status_t::unit_status_t()
  : is_setup(false)
{}

set<int> forward_state_t::unit_status_t::dsts() const {
  set<int> ret;
  for(auto const& [dst,_]: num_move_rem) {
    ret.insert(dst);
  }
  return ret;
};

forward_state_t::move_status_t::move_status_t(int n)
  : unit_status(n)
{}

forward_state_t::graph_node_info_t::graph_node_info_t()
  : partition(std::nullopt),
    joins(std::nullopt),
    compute_status(std::nullopt),
    locs(std::nullopt),
    refinement_partition(std::nullopt),
    refis(std::nullopt),
    move_status(std::nullopt)
{}

bool forward_state_t::all_done() const {
  return can_partition.size() == 0 && num_join_remaining == 0;
}

set<int> const& forward_state_t::can_assign_partition() const {
  return can_partition;
}

void forward_state_t::assign_partition(int gid, partition_t const& new_part) {
  if(new_part.total_shape() != graph.nodes[gid].op.shape()) {
    throw std::runtime_error("given partition has incorrect total shape");
  }

  if(can_partition.count(gid) == 0) {
    throw std::runtime_error("cannot partition: not in can_partition");
  }

  auto& partition = ginfos[gid].partition;
  if(partition) {
    throw std::runtime_error("this has already been given a partition");
  }

  partition = new_part;

  ec_assign_partition(gid);
}

void forward_state_t::assign_location(jid_t jid, int loc) {
  auto const& [gid,bid] = jid;

  if(loc < 0 || loc >= cluster.devices.size()) {
    throw std::runtime_error("cannot assign with this loc");
  }

  auto& ginfo = ginfos[gid];
  if(!ginfo.locs) {
    throw std::runtime_error("must assign partition before assigning location");
  }

  vector<int>& locs = ginfo.locs.value();
  if(locs[bid] != -1) {
    throw std::runtime_error("this location has already been assigned");
  }

  locs[bid] = loc;

  ec_assign_location(jid);
}

void forward_state_t::enqueue_apply_worker(int loc, int which) {
  auto& worker = apply_workers[loc];
  auto const& [gid,bid] = worker.get_pending(which);
  uint64_t const& flops = ginfos[gid].joins.value()[bid].flops;
  double compute_time = cluster.compute(loc, flops);
  worker.start_work(which, time, compute_time);
}

void forward_state_t::enqueue_move_worker(int src, int dst, int which) {
  auto& worker = get_move_worker(src, dst);
  auto const& [rid,uid] = worker.get_pending(which);
  auto const& [gid, bid] = rid;
  auto const& refi = ginfos[gid].refis.value()[bid];
  uint64_t const& elems = refi.units[uid].size;
  double move_time = cluster.move(src, dst, sizeof(float)*elems);
  worker.start_work(which, time, move_time);
}

void forward_state_t::enqueue_all() {
  for(int loc = 0; loc != cluster.devices.size(); ++loc) {
    int n = apply_workers[loc].get_pending().size();
    for(int i = 0; i != n; ++i) {
      enqueue_apply_worker(loc, 0);
    }
  }

  for(auto const& connection: cluster.connections) {
    int src = connection.src;
    int dst = connection.dst;
    int n = get_move_worker(src, dst).get_pending().size();
    for(int i = 0; i != n; ++i) {
      enqueue_move_worker(src, dst, 0);
    }
  }
}

forward_state_t::completed_t
forward_state_t::pop_work()
{
  vector<tuple<double, bool, int>> items;

  for(int i = 0; i != apply_workers.size(); ++i) {
    auto const& apply_worker = apply_workers[i];
    if(apply_worker.is_in_progress()) {
      items.emplace_back(
        std::get<1>(apply_worker.get_in_progress()),
        true,
        i);
    }
  }

  for(int i = 0; i != move_workers.size(); ++i) {
    auto const& move_worker = move_workers[i];
    if(move_worker.is_in_progress()) {
      items.emplace_back(
        std::get<1>(move_worker.get_in_progress()),
        false,
        i);
    }
  }

  if(items.size() == 0) {
    throw std::runtime_error("pop work: no work to pop");
  }

  auto const& [_, is_apply, which] = *std::min_element(items.begin(), items.end());

  if(is_apply) {
    auto& apply_worker = apply_workers[which];

    int const& loc = which;

    auto [start,finish,jid] = apply_worker.get_in_progress();
    auto const& [gid, bid] = jid;

    uint64_t const& flops = ginfos[gid].joins.value()[bid].flops;

    apply_worker.finish_work();

    ec_join(jid);

    time = finish;

    return completed_t(start, finish, loc, gid, bid, flops);
  } else {
    auto& move_worker = move_workers[which];

    auto const& connection  = cluster.connections[which];
    int const& src = connection.src;
    int const& dst = connection.dst;

    auto [start,finish,move_info] = move_worker.get_in_progress();
    auto const& [rid, uid] = move_info;
    auto const& [gid, bid] = rid;
    uint64_t const& size = ginfos[gid].refis.value()[bid].units[uid].size;

    move_worker.finish_work();

    ec_move(rid, uid, dst);

    time = finish;

    return completed_t(start, finish, src, dst, gid, bid, uid, size);
  }
}

void forward_state_t::ec_assign_location(jid_t jid) {
  auto const& [gid,bid] = jid;
  auto const& ginfo = ginfos[gid];

  // if the join objects haven't been setup, then no
  // computation can get started and the src can't
  // be setup
  if(!ginfo.joins) {
    return;
  }

  int const& loc = ginfo.locs.value()[bid];
  join_t const& join = ginfo.joins.value()[bid];
  if(ginfo.refis) {
    vector<refinement_t> const& refis = ginfo.refis.value();

    // we may now be able to setup outgoing units
    for(int const& rid_bid: join.outs) {
      refinement_t const& refi = refis[rid_bid];
      for(int uid = 0; uid != refi.units.size(); ++uid) {
        if(vector_has(refi.units[uid].deps, bid)) {
          rid_t rid { gid, rid_bid };
          if(can_setup_unit_status(rid, uid)) {
            setup_unit_status(rid, uid);
          }
        }
      }
    }
  }

  // add a dst to the dependent refinements
  for(auto const& rid: join.deps) {
    auto const& [inn_gid, _] = rid;
    auto const& inn_ginfo = ginfos[inn_gid];
    if(inn_ginfo.refis) {
      add_refi_dst(rid, jid, loc);
    }
  }

  // Note: the schedule join may have been triggered from above,
  //       but can be called multiple times
  vector<int> const& compute_status = ginfo.compute_status.value();
  if(compute_status[bid] == 0) {
    schedule_join(jid_t {gid, bid}, loc);
  }
}

void forward_state_t::ec_assign_partition(int gid) {
  auto& ginfo = ginfos[gid];

  // can partition update
  can_partition.erase(gid);

  // num_join_remaining must get incremented
  num_join_remaining += product(ginfo.partition.value().block_shape());

  // setup the compute locations to all contain -1 initially
  auto const& node = graph.nodes[gid];
  ginfo.locs = vector<int>(
    product(ginfo.partition.value().block_shape()),
    -1);

  // see if the input refinements can be setup
  auto inns = node.get_inns_set();
  if(inns.size() == 0) {
    setup_joins(gid);
  } else {
    for(auto const& gid_inn: node.get_inns_set()) {
      if(can_setup_refinement_partition(gid_inn)) {
        setup_refinement_partition(gid_inn);
        // this will in turn call setup_joins for this gid
        // if applicable
      }
    }
  }
}

bool forward_state_t::can_setup_refinement_partition(int gid) const {
  if(ginfos[gid].refinement_partition) {
    // nothing to do if it has already been setup
    return false;
  }

  auto const& node = graph.nodes[gid];
  if(node.outs.size() == 0) {
    // if there is only one output, there is no refinement
    // partition as the node does not have a usage
    return false;
  }

  // see if each output has a partition
  for(auto const& gid_out: node.outs) {
    if(!ginfos[gid_out].partition) {
      return false;
    }
  }

  return true;
}

// Note: Almost a copy of union_partition_holders in src/taskgraph.cc
partition_t union_partitions(vector<partition_t> const& ps)
{
  if(ps.size() == 0) {
    throw std::runtime_error("union partitions: input is empty");
    return ps[0];
  }
  if(ps.size() == 1) {
    return ps[0];
  }

  vector<partdim_t> partdims;
  int rank = ps[0].block_shape().size();
  partdims.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    vector<partdim_t> xs;
    xs.reserve(ps.size());
    for(auto const& p: ps) {
      xs.push_back(p.partdims[i]);
    }
    partdims.push_back(partdim_t::unions(xs));
  }
  return partition_t(partdims);
}

void forward_state_t::setup_refinement_partition(int join_id) {
  auto const& join_node = graph.nodes[join_id];
  vector<partition_t> usage_partitions;
  usage_partitions.reserve(2*join_node.outs.size());
  for(auto const& out_id: join_node.outs) {
    auto const& out_node = graph.nodes[out_id];
    if(out_node.op.is_formation()) {
      usage_partitions.push_back(out_node.placement.partition);
    } else {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage partitions to collect
      auto const& einsummable = out_node.op.get_einsummable();
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_id) {
          usage_partitions.emplace_back(einsummable.get_input_from_join(
            out_node.placement.partition.partdims,
            which_input));
        }
      }
    }
  }

  ginfos[join_id].refinement_partition = union_partitions(usage_partitions);

  if(can_setup_refis(join_id)) {
    setup_refis(join_id);
  }

  ec_setup_refinement(join_id);
}

void forward_state_t::ec_setup_refinement(int gid) {
  auto const& node = graph.nodes[gid];
  for(auto const& out_id: node.outs) {
    if(can_setup_joins(out_id)) {
      setup_joins(out_id);
    }
  }
}

bool forward_state_t::can_setup_refis(int gid) const {
  auto const& ginfo = ginfos[gid];
  return bool(ginfo.joins) && bool(ginfo.refinement_partition);
}

void forward_state_t::setup_refis(int graph_id) {
  auto const& node = graph.nodes[graph_id];
  auto& ginfo = ginfos[graph_id];

  partition_t const& join_partition = ginfo.partition.value();
  partition_t const& refi_partition = ginfo.refinement_partition.value();

  int join_rank = node.op.rank();
  int out_rank  = node.op.out_rank();
  int agg_rank  = join_rank - out_rank;

  auto join_shape = join_partition.block_shape();
  auto const& _join_partdims = join_partition.partdims;

  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  std::optional<vector<int>> maybe_agg_shape;
  if(agg_rank > 0) {
    partition_t agg_partition(vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end()));

    maybe_agg_shape = agg_partition.block_shape();
  }

  // set up the refinement nodes
  auto refi_shape = refi_partition.block_shape();

  ginfo.refis = vector<refinement_t>(product(refi_shape));
  vector<refinement_t>& refis = ginfo.refis.value();

  auto& joins = ginfo.joins.value();

  vector<int> out_shape = out_partition.block_shape();
  vector<int> out_index(out_shape.size(), 0);
  do {
    copyregion_t get_regions(refi_partition, out_partition, out_index);
    do {
      vector<int> refi_index = vector_from_each_member(
        get_regions.info, int, idx);
      vector<uint64_t> read_shape = vector_from_each_member(
        get_regions.info, uint64_t, size);

      auto& refi = refis[idxs_to_index(refi_shape, refi_index)];
      refi.units.push_back(agg_unit_t {
        .size = product(read_shape),
        .deps = {}
      });

      vector<int>& deps = refi.units.back().deps;
      if(maybe_agg_shape) {
        vector<int> agg_index(agg_rank, 0);
        auto const& agg_shape = maybe_agg_shape.value();
        deps.reserve(product(agg_shape));
        do {
          vector<int> join_index = vector_concatenate(out_index, agg_index);
          int join_bid = idxs_to_index(join_shape, join_index);
          deps.push_back(join_bid);
        } while(increment_idxs(agg_shape, agg_index));
      } else {
        // the join index is the out index if there is no agg
        // and there is only one input
        auto const& join_index = out_index;
        int join_bid = idxs_to_index(join_shape, join_index);
        deps.push_back(join_bid);
      }

      // an agg unit has been added to refi, so let the input
      // joins know they have an output here
      int refi_bid = idxs_to_index(refi_shape, refi_index);
      for(auto const& dep_join_bid: deps) {
        joins[dep_join_bid].outs.insert(refi_bid);
      }
    } while(get_regions.increment());
  } while(increment_idxs(out_shape, out_index));

  // Now setup move_status where possible
  ginfo.move_status = vector<move_status_t>();
  vector<move_status_t>& move_status = ginfo.move_status.value();
  int n_bid = product(refi_shape);
  move_status.reserve(n_bid);
  for(int bid = 0; bid != n_bid; ++bid) {
    auto const& refi = refis[bid];
    int n_uid = refi.units.size();
    move_status.emplace_back(n_uid);
    rid_t rid { graph_id, bid };
    for(int uid = 0; uid != n_uid; ++uid) {
      if(can_setup_unit_status(rid, uid)) {
        setup_unit_status(rid, uid);
      }
    }
  }
}

bool forward_state_t::can_setup_joins(int gid) const {
  auto const& ginfo = ginfos[gid];
  if(ginfo.joins) {
    // nothing to do if it has already been setup
    return false;
  }

  if(!ginfo.partition) {
    // if this doesn't have a partition, then joins can't be setup
    return false;
  }

  // see if each input has the refinement partition setup
  auto const& node = graph.nodes[gid];
  for(auto const& gid_inn: node.get_inns_set()) {
    if(!ginfos[gid_inn].refinement_partition) {
      return false;
    }
  }

  return true;
}

void forward_state_t::setup_joins(int graph_id) {
  auto const& node = graph.nodes[graph_id];
  auto& ginfo = ginfos[graph_id];

  partition_t const& join_partition = ginfo.partition.value();

  auto join_block_shape = join_partition.block_shape();

  ginfo.joins = vector<join_t>(product(join_block_shape));
  vector<join_t>& join_infos = ginfo.joins.value();

  vector<int> join_index(join_block_shape.size(), 0);

  do {
    int join_bid = idxs_to_index(join_block_shape, join_index);
    join_t& join_info = join_infos[idxs_to_index(join_block_shape, join_index)];

    // flops
    //   input nodes: 0
    //   formation nodes: 0
    //   join nodes: the tensor block size
    if(node.op.is_einsummable()) {
      join_info.flops = product(join_partition.tensor_shape_at(join_index));
    } else {
      join_info.flops = 0;
    }

    // deps
    //   input nodes: {}
    //   formation nodes: same as a straight einsummable op
    //   einsummable nodes: reach into each input and grab it
    if(node.op.is_input()) {
      join_info.deps = {};
    } else {
      optional<einsummable_t> maybe_einsummable;
      if(node.op.is_formation()) {
        auto op_shape = node.op.shape();
        int rank = op_shape.size();

        vector<vector<int>> inns(1);
        inns[0] = vector<int>(rank);
        std::iota(inns[0].begin(), inns[0].end(), 0);

        maybe_einsummable = einsummable_t(
          op_shape,
          inns,
          rank,
          scalarop_t::make_identity()); // will not be used
      } else {
        maybe_einsummable = node.op.get_einsummable();
      }

      auto hrect = join_partition.get_hrect(join_index);

      // for each input, get the inputs refinement ids that map
      // onto the corresponding input hrect.
      for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
        int const& inn = node.inns[which_inn];

        auto& inn_ginfo = ginfos[inn];

        partition_t const& inn_partition = inn_ginfo.refinement_partition.value();

        auto inn_shape = inn_partition.block_shape();

        auto const& e = maybe_einsummable.value();
        auto inn_hrect = e.get_input_from_join(hrect, which_inn);

        auto inn_region = inn_partition.get_region(inn_hrect);
        vector<int> inn_index = vector_mapfst(inn_region);
        do {
          int inn_refi_bid = idxs_to_index(inn_shape, inn_index);
          rid_t dep_rid { inn, inn_refi_bid };
          join_info.deps.push_back(dep_rid);
          insert_refi_out(dep_rid, jid_t { graph_id, join_bid });
        } while(increment_idxs_region(inn_region, inn_index));
      }
    }
  } while(increment_idxs(join_block_shape, join_index));

  ec_setup_joins(graph_id);
}

void forward_state_t::insert_refi_out(rid_t rid, jid_t jid) {
  auto const& [r_gid, r_bid] = rid;
  auto const& [j_gid, j_bid] = jid;

  refinement_t& refi = ginfos[r_gid].refis.value()[r_bid];
  refi.outs.insert(jid);

  int const& loc = ginfos[j_gid].locs.value()[j_bid];
  if(loc != -1) {
    add_refi_dst(rid, jid, loc);
  }
}

void forward_state_t::ec_setup_joins(int gid) {
  auto& ginfo = ginfos[gid];
  auto const& joins = ginfo.joins.value();

  // setup the compute status
  {
    ginfo.compute_status = vector<int>(joins.size());
    vector<int>& compute_status = ginfo.compute_status.value();

    for(int bid = 0; bid != joins.size(); ++bid) {
      compute_status[bid] = joins[bid].deps.size();
    }
  }

  if(can_setup_refis(gid)) {
    setup_refis(gid);
  }

  // schedule what can be scheduled
  vector<int> const& locs = ginfo.locs.value();
  vector<int> const& status = ginfo.compute_status.value();
  for(int bid = 0; bid != locs.size(); ++bid) {
    int const& loc = locs[bid];
    if(loc >= 0 && status[bid] == 0) {
      schedule_join(jid_t {gid, bid}, loc);
    }
  }
}

bool forward_state_t::can_setup_unit_status(rid_t rid, int uid) const {
  auto const& [gid, bid] = rid;

  auto const& ginfo = ginfos[gid];
  if(!ginfo.move_status) {
    // got to set up the refinements first
    return false;
  }

  move_status_t const& ms = ginfo.move_status.value()[bid];
  unit_status_t const& us = ms.unit_status[uid];
  if(us.is_setup) {
    // already setup, can't do it again
    return false;
  }

  refinement_t const& refi = ginfo.refis.value()[bid];
  auto const& unit = refi.units[uid];
  vector<int> const& locs = ginfo.locs.value();
  for(int const& inn_join_bid: unit.deps) {
    if(locs[inn_join_bid] == -1) {
      return false;
    }
  }

  return true;
}

void forward_state_t::setup_unit_status(rid_t rid, int uid) {
  auto const& [gid,bid] = rid;
  auto& ginfo = ginfos[gid];

  refinement_t const& refi = ginfo.refis.value()[bid];
  agg_unit_t const& unit = refi.units[uid];
  vector<int> const& locs = ginfo.locs.value();
  vector<int> const& compute_status = ginfo.compute_status.value();

  move_status_t& move_status = ginfo.move_status.value()[bid];
  unit_status_t& unit_status = move_status.unit_status[uid];

  for(int const& join_bid: unit.deps) {
    int const& src = locs[join_bid];
    int& cnt = unit_status.num_join_rem[src];
    if(compute_status[join_bid] != -1) {
      cnt += 1;
    }
  }
  int num_srcs = unit_status.num_join_rem.size();

  for(auto const& [out_gid, out_bid]: refi.outs) {
    int const& dst = ginfos[out_gid].locs.value()[out_bid];
    if(dst >= 0) {
      unit_status.num_move_rem.insert({dst, num_srcs});
    }
  }

  for(auto const& [src,num_rem]: unit_status.num_join_rem) {
    if(num_rem == 0) {
      for(int const& dst: unit_status.dsts()) {
        schedule_move(rid, uid, src, dst);
      }
    }
  }

  unit_status.is_setup = true;
}

void forward_state_t::ec_move(rid_t rid, int uid, int dst) {
  auto const& [gid, bid] = rid;

  auto& ginfo = ginfos[gid];

  move_status_t& move_status = ginfo.move_status.value()[bid];
  unit_status_t& unit_status = move_status.unit_status[uid];

  int& cnt = unit_status.num_move_rem[dst];
  cnt -= 1;
  if(cnt < 0) {
    throw std::runtime_error("how can count go below zero: ec_move");
  } else if(cnt == 0) {
    ec_agg_unit(rid, dst);
  }
}

void forward_state_t::ec_agg_unit(rid_t rid, int dst) {
  auto const& [gid, bid] = rid;

  auto& ginfo = ginfos[gid];

  move_status_t& move_status = ginfo.move_status.value()[bid];

  int& cnt = move_status.num_unit_rem[dst];
  cnt -= 1;
  if(cnt < 0) {
    throw std::runtime_error("how can count go below zero: ec_agg_unit");
  } else if(cnt == 0) {
    ec_refinement(rid, dst);
  }
}

void forward_state_t::ec_refinement(rid_t rid, int dst) {
  auto const& [gid, bid] = rid;

  auto& ginfo = ginfos[gid];

  refinement_t const& refi = ginfo.refis.value()[bid];
  for(auto const& [out_gid, out_bid]: refi.outs) {
    auto& out_ginfo = ginfos[out_gid];
    vector<int> const& locs = out_ginfo.locs.value();
    int const& loc = locs[out_bid];
    if(loc == dst) {
      vector<int>& compute_status = out_ginfo.compute_status.value();
      int& cnt = compute_status[out_bid];
      cnt -= 1;
      if(cnt < 0) {
        throw std::runtime_error("how can count go below zero: ec_refinement");
      } else if(cnt == 0) {
        schedule_join(jid_t { out_gid, out_bid }, loc);
      }
    }
  }
}

void forward_state_t::ec_join(jid_t jid) {
  num_join_remaining -= 1;

  auto const& [gid,bid] = jid;
  auto& ginfo = ginfos[gid];

  // Update the compute status
  {
    vector<int>& compute_status = ginfo.compute_status.value();
    int& cnt = compute_status[bid];
    cnt -= 1;
    if(cnt != -1) {
      throw std::runtime_error("invalid compute status in ec_join");
    }
  }

  if(!ginfo.joins || !ginfo.refis || !ginfo.move_status) {
    return;
  }

  auto const& join  = ginfo.joins.value()[bid];
  int const& loc    = ginfo.locs.value()[bid];

  auto const& refis = ginfo.refis.value();

  vector<move_status_t>& move_statuses = ginfo.move_status.value();
  for(int const& rid_bid: join.outs) {
    auto const& refi = refis[rid_bid];
    move_status_t& move_status = move_statuses[rid_bid];
    for(int uid = 0; uid != refi.units.size(); ++uid) {
      auto const& unit = refi.units[uid];
      unit_status_t& unit_status = move_status.unit_status[uid];

      if(unit_status.is_setup && vector_has(unit.deps, bid)) {
        int& cnt = unit_status.num_join_rem[loc];
        cnt -= 1;
        if(cnt < 0) {
          throw std::runtime_error("how can count go below zero: ec_join");
        } else if(cnt == 0) {
          for(int const& dst: unit_status.dsts()) {
            schedule_move(rid_t {gid, rid_bid}, uid, loc, dst);
          }
        }
      }
    }
  }
}

void forward_state_t::add_refi_dst(rid_t rid, jid_t jid, int dst) {
  auto const& [gid, bid] = rid;
  auto& ginfo = ginfos[gid];
  vector<move_status_t>& move_statuses = ginfo.move_status.value();
  move_status_t& move_status = move_statuses[bid];

  refinement_t const& refi = ginfo.refis.value()[bid];

  if(move_status.num_unit_rem.count(dst) == 0) {
    // this many units need to be completed at this new dst
    move_status.num_unit_rem[dst] = refi.units.size();

    // each unit needs to have done this many moves
    for(int uid = 0; uid != refi.units.size(); ++uid) {
      unit_status_t& unit_status = move_status.unit_status[uid];
      if(unit_status.is_setup) {
        int num_srcs = unit_status.num_join_rem.size();
        unit_status.num_move_rem[dst] = num_srcs;

        for(auto const& [src,num_rem]: unit_status.num_join_rem) {
          if(num_rem == 0) {
            schedule_move(rid, uid, src, dst);
          }
        }
      }
    }
  } else {
    // This dst was already scheduled. Is it already done?
    if(move_status.num_unit_rem.at(dst) == 0) {
      auto const& [out_gid, out_bid] = jid;
      int& cnt = ginfos[out_gid].compute_status.value()[out_bid];
      cnt -= 1;
      if(cnt == 0) {
        schedule_join(jid, dst);
      }
    }
  }
}

void forward_state_t::schedule_join(jid_t jid, int loc) {
  auto const& [gid, bid] = jid;
  if(!graph.nodes[gid].op.is_einsummable()) {
    // inputs and formations happen immediately
    ec_join(jid);
    return;
  }

  apply_workers[loc].add_to_pending(jid);
}

void forward_state_t::schedule_move(rid_t rid, int uid, int src, int dst) {
  auto const& [gid, bid] = rid;

  if(src == dst) {
    // this "move" is immediate and triggers the event completion
    ec_move(rid, uid, dst);
    return;
  }

  auto& worker = get_move_worker(src, dst);
  worker.add_to_pending({rid, uid});
}

worker_t<tuple<forward_state_t::rid_t, int>>&
forward_state_t::get_move_worker(int src, int dst) {
  return move_workers[to_move_worker.at({src,dst})];
}

void forward_state_t::print_twolayer_graphviz(std::ostream& out) const {
  using std::endl;

  auto xstr = [](string x, int gid, int bid) {
    return x + "_" + write_with_ss(gid) + "_" + write_with_ss(bid);
  };
  auto jstr = [&xstr](int gid, int bid) { return xstr("j", gid, bid); };
  auto rstr = [&xstr](int gid, int bid) { return xstr("r", gid, bid); };

  string tab = "  ";
  out << "digraph {" << endl;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& ginfo = ginfos[gid];
    if(ginfo.joins) {
      auto const& joins = ginfo.joins.value();
      for(int bid = 0; bid != joins.size(); ++bid) {
        out << tab << jstr(gid, bid) << endl;

        auto const& join = joins[bid];
        for(auto const& [inn_gid, inn_bid]: join.deps) {
          out << tab << rstr(inn_gid, inn_bid) << " -> "
            << jstr(gid, bid) << endl;
        }
      }
    }
    if(ginfo.refis) {
      auto const& refis = ginfo.refis.value();
      for(int bid = 0; bid != refis.size(); ++bid) {
        out << tab << rstr(gid, bid) << endl;

        auto const& refi = refis[bid];
        for(auto const& unit: refi.units) {
          for(auto const& inn_bid: unit.deps) {
            out << tab << jstr(gid, inn_bid) << " -> "
              << rstr(gid, bid) << endl;
          }
        }
      }
    }
  }
  out << "}" << endl;
}

bool operator==(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator< (forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
bool operator==(forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator< (forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}

std::ostream& operator<<(std::ostream& out, forward_state_t::jid_t const& jid) {
  auto const& [gid,bid] = jid;
  out << "jid{" << gid << "," << bid << "}";
  return out;
}
std::ostream& operator<<(std::ostream& out, forward_state_t::rid_t const& rid) {
  auto const& [gid,bid] = rid;
  out << "rid{" << gid << "," << bid << "}";
  return out;
}
