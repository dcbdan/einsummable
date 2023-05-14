#include "forwardsim.h"

forward_state_t::forward_state_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl)
  : cluster(c),
    twolayer(tl),
    joins(tl.joins),
    refis(tl.refinements),
    equal_compute_locations(ecl),
    apply_workers(c.devices.size()),
    move_workers(c.connections.size()),
    to_move_worker(cluster.to_connection),
    compute_locations(tl.joins.size(), -1),
    compute_status(tl.joins.size()),
    num_compute_remaining(tl.joins.size()),
    time(0.0)
{
  // set compute_status and add all inputs to
  // pending location choices
  for(int jid = 0; jid != joins.size(); ++jid) {
    auto const& join = joins[jid];
    int num_inns = join.deps.size();
    compute_status[jid] = num_inns;

    if(num_inns == 0) {
      pending_location_choices.push(jid);
    }
  }
}

tuple<double, double, forward_state_t::completed_t>
forward_state_t::step(decision_interface_t const& interface)
{
  while(pending_location_choices.size() != 0) {
    int jid_ = pending_location_choices.front();
    pending_location_choices.pop();

    if(compute_locations[jid_] >= 0) {
      continue;
    }

    vector<int> jids;
    if(equal_compute_locations.has(jid_)) {
      set<int> const& eq_jids = equal_compute_locations.get_at(jid_);
      jids = vector<int>(eq_jids.begin(), eq_jids.end());
      std::sort(jids.begin(), jids.end());
    } else {
      jids.push_back(jid_);
    }

    int chosen_loc = interface.choose_location(jid_);

    for(int const& jid: jids) {
      if(compute_locations[jid] != -1) {
        throw std::runtime_error(
          "this compute location has already been chosen somehow");
      }
      compute_locations[jid] = chosen_loc;

      process_assigned_location(jid);
    }
  }

  // Make sure all the workers with something to do
  // are doing something
  for(int loc = 0; loc != apply_workers.size(); ++loc) {
    auto& apply_worker = apply_workers[loc];
    if(!apply_worker.is_in_progress()) {
      vector<int> const& pending = apply_worker.get_pending();

      int which = interface.choose_apply(loc, pending);

      int const& jid = pending[which];
      uint64_t const& flops = joins[jid].flops;
      double work_time = cluster.compute(loc, flops);

      apply_worker.start_work(which, time, time + work_time);
    }
  }
  for(auto const& [key,idx]: to_move_worker) {
    auto const& [src,dst] = key;
    auto& move_worker = move_workers[idx];
    if(!move_worker.is_in_progress()) {
      vector<tl_move_t> const& pending = move_worker.get_pending();

      int which = interface.choose_move(src, dst, pending);

      auto const& [rid,uid] = pending[which];
      uint64_t const& bytes = refis[rid].units[uid].bytes;
      double work_time = cluster.move(src, dst, bytes);

      move_worker.start_work(which, time, time + work_time);
    }
  }

  // get the next thing to finish
  tuple<double, double, completed_t> ret = pop_work();
  auto const& [_, finished_at, completion] = ret;

  time = finished_at;

  if(completion.did_apply()) {
    auto const& [_, jid] = completion.get_apply_info();
    process_completed_join(jid);
  } else if(completion.did_move()) {
    auto const& [_,dst,rid,uid] = completion.get_move_info();
    process_completed_move(rid, uid, dst);
  } else {
    throw std::runtime_error("completion object cases: should not reach");
  }

  return ret;
}

bool forward_state_t::all_done() const {
  return num_compute_remaining == 0;
}

tuple<double, double, forward_state_t::completed_t>
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

  auto const& [_, is_apply, which] = *std::min_element(items.begin(), items.end());
  if(is_apply) {
    auto& apply_worker = apply_workers[which];

    int const& loc = which;

    auto [start,finish,join] = apply_worker.get_in_progress();
    apply_worker.finish_work();
    return {start, finish, completed_t(loc, join)};
  } else {
    auto& move_worker = move_workers[which];

    auto const& connection  = cluster.connections[which];
    int const& src = connection.src;
    int const& dst = connection.dst;

    auto [start,finish,move] = move_worker.get_in_progress();
    auto const& [rid, uid] = move;
    move_worker.finish_work();
    return {start, finish, completed_t(src, dst, rid, uid)};
  }
}

void forward_state_t::process_assigned_location(int jid)
{
  int const& num_deps = compute_status[jid];

  if(num_deps == 0) {
    int const& loc = compute_locations[jid];
    apply_workers[loc].add_to_pending(jid);
  } else if(compute_status[jid] < 0) {
    throw std::runtime_error("how come this jid was computed without a loc?");
  }

  auto const& join = joins[jid];
  for(auto const& out_rid: join.outs) {
    auto const& out_refi = refis[out_rid];
    for(int uid = 0; uid != out_refi.units.size(); ++uid) {
      if(is_avail_agg_unit(out_rid, uid)) {
        process_avail_agg_unit(out_rid, uid);
      }
    }
  }

  for(auto const& inn_rid: join.deps) {
    auto const& inn_refi = refis[inn_rid];
    for(int uid = 0; uid != inn_refi.units.size(); ++uid) {
      if(is_avail_agg_unit(inn_rid, uid)) {
        process_avail_agg_unit(inn_rid, uid);
      }
    }
  }
}

void forward_state_t::process_completed_join(int jid)
{
  num_compute_remaining -= 1;

  compute_status[jid] -= 1;
  if(compute_status[jid] != -1) {
    throw std::runtime_error("compute status does not have -1 value");
  }

  auto const& join = joins[jid];
  int const& loc = compute_locations[jid];

  for(auto const& rid: join.outs) {
    auto const& refi = refis[rid];
    for(int uid = 0; uid != refi.units.size(); ++uid) {
      auto const& unit = refis[rid].units[uid];
      if(vector_has(unit.deps, jid)) { // maybe this is always true (?)
        if(is_avail_agg_unit(rid, uid)) {
          notify_agg_moves_in_progress(rid, uid, loc);
        } else {
          // The agg move at rid,uid,loc would be notified
          // but it hasn't been setup yet. To get it setup,
          // add the necc things to pending_location_choices
          choose_agg_unit_locs(rid, uid);
        }
      }
    }
  }
}

// notify agg moves in progress that a corresponding join just completed
void forward_state_t::notify_agg_moves_in_progress(
  int rid, int uid, int src)
{
  auto iter = agg_moves_in_progress.find({rid,uid,src});
  if(iter == agg_moves_in_progress.end()) {
    throw std::runtime_error("this agg move is not in progress");
  }
  auto& [_,cnt] = *iter;
  cnt -= 1;
  if(cnt == 0) {
    agg_moves_in_progress.erase(iter);
    broadcast_agg_move(rid, uid, src);
  }
}

void forward_state_t::broadcast_agg_move(int rid, int uid, int src)
{
  auto maybe_dsts = get_refi_dsts(rid);
  if(!maybe_dsts) {
    throw std::runtime_error("this agg unit is not available");
  }
  for(auto const& dst: maybe_dsts.value()) {
    if(src == dst) {
      // the move doesn't actually happen, but the
      // bookkeeping does
      process_completed_move(rid, uid, dst);
    } else {
      int const& which_worker = to_move_worker.at({src,dst});
      auto& move_worker = move_workers[which_worker];
      move_worker.add_to_pending(tl_move_t{ rid, uid });
    }
  }
}

void forward_state_t::process_completed_move(int rid, int uid, int dst)
{
  int& cnt = agg_units_in_progress.at({rid,uid,dst});
  cnt -= 1;
  if(cnt == 0) {
    process_completed_agg_unit(rid, uid, dst);
  }
}

void forward_state_t::process_completed_agg_unit(int rid, int uid, int dst)
{
  auto iter = agg_units_in_progress.find({rid,uid,dst});
  if(iter == agg_units_in_progress.end()) {
    throw std::runtime_error("this agg unit dst is not available");
  }

  {
    auto const& [_, cnt] = *iter;
    if(cnt != 0) {
      throw std::runtime_error("this agg unit is not actually complete");
    }
  }

  agg_units_in_progress.erase(iter);

  int& cnt = refis_in_progress.at({rid,dst});
  cnt -= 1;
  if(cnt == 0) {
    process_completed_refi(rid, dst);
  }
}

void forward_state_t::process_completed_refi(int rid, int dst)
{
  auto iter = refis_in_progress.find({rid,dst});
  if(iter == refis_in_progress.end()) {
    throw std::runtime_error("this refi is not available");
  }

  {
    auto const& [_,cnt] = *iter;
    if(cnt != 0) {
      throw std::runtime_error("this refi is not actually complete");
    }
  }

  refis_in_progress.erase(iter);

  auto const& refi = refis[rid];
  for(auto const& out_jid: refi.outs) {
    if(compute_locations[out_jid] == dst) {
      int& cnt = compute_status[out_jid];
      cnt -= 1;
      if(cnt == 0) {
        // this apply should now be pushed to pending
        apply_workers[dst].add_to_pending(out_jid);
      }
    }
  }
}

void forward_state_t::process_avail_agg_unit(int rid, int uid)
{
  auto const& refi = refis[rid];
  auto const& unit = refi.units[uid];

  auto maybe_srcs = get_agg_unit_srcs(rid, uid);
  auto maybe_dsts = get_refi_dsts(rid);

  if(!maybe_srcs || !maybe_dsts) {
    throw std::runtime_error("this agg unit cannot be ready for setting up");
  }

  set<int> const& srcs = maybe_srcs.value();
  set<int> const& dsts = maybe_dsts.value();

  // If this is the first agg unit of this
  // refinement, setup refis_in_progress to start counting
  // for each dst
  {
    int const& a_dst = *dsts.begin();
    if(refis_in_progress.count({rid, a_dst}) == 0) {
      for(auto const& dst: dsts) {
        refis_in_progress.insert({ {rid,dst}, refi.units.size() });
      }
    }
  }

  // First setup agg_units_in_progress since agg_moves_in_progress
  // may propagate information into agg_units_in_progress
  for(auto const& dst: dsts) {
    agg_units_in_progress.insert({ {rid, uid, dst}, srcs.size() });

  }

  for(auto const& src: srcs) {
    int n_apply_here = 0;
    int n_apply = 0;
    for(auto const& inn_jid: unit.deps) {
      if(compute_locations[inn_jid] == src) {
        n_apply_here += 1;
        if(compute_status[inn_jid] != -1) {
          n_apply += 1;
        }
      }
    }

    if(n_apply_here == 0) {
      throw std::runtime_error("this src doesn't have a corr apply");
    }

    if(n_apply > 0) {
      agg_moves_in_progress.insert({ {rid, uid, src}, n_apply });
    } else {
      broadcast_agg_move(rid, uid, src);
    }
  }
}

bool forward_state_t::is_avail_agg_unit(int rid, int uid) const
{
  {
    auto maybe_srcs = get_agg_unit_srcs(rid, uid);
    if(!maybe_srcs) {
      return false;
    }
  }

  {
    auto maybe_dsts = get_refi_dsts(rid);
    if(!maybe_dsts) {
      return false;
    }
  }

  return true;
}

optional<set<int>> forward_state_t::get_agg_unit_srcs(
  int rid, int uid) const
{
  auto const& unit = refis[rid].units[uid];

  set<int> srcs;
  for(auto const& inn_jid: unit.deps) {
    int const& loc = compute_locations[inn_jid];
    if(loc == -1) {
      return {};
    }
    srcs.insert(loc);
  }

  return optional<set<int>>(srcs);
}

optional<set<int>> forward_state_t::get_refi_dsts(
  int rid) const
{
  auto const& refi = refis[rid];

  set<int> dsts;
  for(auto const& out_jid: refi.outs) {
    int const& loc = compute_locations[out_jid];
    if(loc == -1) {
      return {};
    }
    dsts.insert(loc);
  }

  return dsts;
}

void forward_state_t::choose_agg_unit_locs(int rid, int uid)
{
  auto const& refi = refis[rid];
  for(auto const& out_jid: refi.outs) {
    if(compute_locations[out_jid] == -1) {
      pending_location_choices.push(out_jid);
    }
  }

  auto const& unit = refi.units[uid];
  for(auto const& inn_jid: unit.deps) {
    if(compute_locations[inn_jid] == -1) {
      pending_location_choices.push(inn_jid);
    }
  }
}

bool operator==(tl_move_t const& lhs, tl_move_t const& rhs)
{
  return two_tuple_eq(lhs, rhs);
}

decision_interface_t decision_interface_t::random(int nloc)
{
  return decision_interface_t {
    .choose_apply = [](int loc, vector<int> const& pending) {
        return runif(pending.size());
      },
    .choose_move = [](int src, int dst, vector<tl_move_t> const& pending) {
        return runif(pending.size());
      },
    .choose_location = [nloc](int jid) {
        return runif(nloc);
      }
  };
}

decision_interface_t decision_interface_t::from_locs(vector<int> const& at_locs)
{
  return decision_interface_t {
    .choose_apply = [](int loc, vector<int> const& pending) {
        return runif(pending.size());
      },
    .choose_move = [](int src, int dst, vector<tl_move_t> const& pending) {
        return runif(pending.size());
      },
    .choose_location = [at_locs](int jid) {
        return at_locs[jid];
      }
  };
}

