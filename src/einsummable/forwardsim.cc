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
    started_move_from(tl.refinements.size()),
    refi_status_setup(tl.refinements.size(), 0),
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

  for(int rid = 0;  rid != refis.size(); ++rid) {
    auto const& refi = refis[rid];
    started_move_from[rid] = vector<set<int>>(refi.units.size());
  }
}

// What triggers what?
//   join can start:
//     location set
//     all input rids have completed at location
//   agg unit at src can be broadcast:
//     all input locations have been assigned
//     all input deps at src have been computed
//     all outputs have been given a location
//   rid at loc has completed:
//     all moves across all agg units to loc have completed
//   work can start:
//     corr worker is not busy
//     corr pending is not empty
// The invariant:
//   pending work should never not be in pending because a location
//   wasn't chosen
tuple<double, double, forward_state_t::completed_t>
forward_state_t::step(decision_interface_t const& interface)
{
  // Make sure all ids in pending location choices have
  // a location
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

      auto const& join = joins[jid];
      // all input joins can be computed immediately and
      // won't get triggered anywhere else so do so here
      if(join.deps.size() == 0) {
        apply_workers[chosen_loc].add_to_pending(jid);
      }

      // setting this location might trigger input rids
      for(int const& rid: join.deps) {
        if(refi_out_locs_set_and_refi_status_setup(rid)) {
          auto const& refi = refis[rid];
          for(int uid = 0; uid != refi.units.size(); ++uid) {
            for(auto const& src: get_avail_broadcast_srcs(rid, uid)) {
              add_broadcast_to_pending(rid, uid, src);
            }
          }
        }
      }
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
    auto const& [loc, jid] = completion.get_apply_info();

    num_compute_remaining -= 1;

    compute_status[jid] -= 1;
    if(compute_status[jid] != -1) {
      throw std::runtime_error("compute status does not have -1 value");
    }

    auto const& join = joins[jid];

    // For each output agg unit:
    //   notify the agg unit and see if it
    //
    // Case 1: The output agg unit cannot be moved from here. Do nothing else.
    // Case 2: The output agg unit can be moved from but some output locations of
    //         the refi are still not set. Then add all those output locations to
    //         be set later.
    // Case 3: The output agg unit can be moved from and all the output locations
    //         of this refi have been set. Then do the broadcast from this location.
    for(auto const& rid: join.outs) {
      auto const& refi = refis[rid];
      bool out_locs_set = refi_out_locs_set_and_refi_status_setup(rid);
      for(int uid = 0; uid != refi.units.size(); ++uid) {
        auto const& unit = refis[rid].units[uid];
        if(vector_has(unit.deps, jid)) { // maybe this is always true (?)
          bool unit_loc_is_ready = notify_agg_unit_at(rid, uid, loc);
          if(unit_loc_is_ready) {
            if(out_locs_set) {
              // Case 3.
              add_broadcast_to_pending(rid, uid, loc);
            } else {
              // Case 2.
              for(auto const& out_jid: refi.outs) {
                if(compute_locations[out_jid] < 0) {
                  pending_location_choices.push(out_jid);
                }
              }
            }
          } else {
            // Case 1.
          }
        }
      }
    }
  } else if(completion.did_move()) {
    auto const& [src,dst,rid,unit] = completion.get_move_info();

    // Did a refi complete with this agg unit being moved here?
    // If so, try to start subsequent joins.

    decrement_refi_status_and_maybe_dep_joins(rid, dst);

  } else {
    throw std::runtime_error("should not reach");
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

void forward_state_t::add_broadcast_to_pending(
  int rid, int uid, int src)
{
  {
    vector<int> can_move_srcs = get_avail_broadcast_srcs(rid, uid);
    if(!vector_has(can_move_srcs, src)) {
      throw std::runtime_error("trying to add broadcast that can't be added");
    }
  }

  auto const& refi = refis[rid];
  auto const& unit = refi.units[uid];

  set<int> dsts;
  for(auto const& out_jid: refi.outs) {
    dsts.insert(compute_locations[out_jid]);
  }
  for(auto const& dst: dsts) {
    if(src != dst) {
      int which_worker = to_move_worker.at({src,dst});
      auto& move_worker = move_workers[which_worker];
      move_worker.add_to_pending(tl_move_t{ rid, uid });
    }
  }

  started_move_from[rid][uid].insert(src);
}

bool forward_state_t::refi_out_locs_set_and_refi_status_setup(
  int rid)
{
  auto const& refi = refis[rid];

  bool all_out_locs_chosen = true;
  for(auto const& out_jid: refi.outs) {
    if(compute_locations[out_jid] < 0) {
      all_out_locs_chosen = false;
      break;
    }
  }

  if(all_out_locs_chosen && !refi_status_setup[rid]) {
    // At this point, refi_status for rid needs to be setup.
    //
    // A refi status contains for each dst of this rid,
    // the number of remaining agg units until the rid,dst
    // pair is complete.
    //
    // If an rid,dst pair has already completed, decrement
    // and possibly add to pending the dependent joins.
    // This may happen if none of the agg units actually
    // require moving data.

    set<int> dsts;
    for(auto const& out_jid: refi.outs) {
      dsts.insert(compute_locations[out_jid]);
    }

    // the number of expected moves for each dst
    map<int, int> cnts;
    for(auto const& unit: refi.units) {
      set<int> srcs;
      for(auto const& inn_jid: unit.deps) {
        srcs.insert(compute_locations[inn_jid]);
      }
      for(auto const& dst: dsts) {
        int n_srcs = srcs.size();
        if(srcs.count(dst) > 0) {
          n_srcs -= 1;
        }

        cnts[dst] += n_srcs;
      }
    }

    for(auto const& [dst,cnt]: cnts) {
      if(cnt == 0) {
        decrement_dep_joins(rid, dst);
      } else {
        refi_status.insert({ {rid,dst}, cnt});
      }
    }

    // flip the flag now that it has been setup
    refi_status_setup[rid] = 1;
  }

  return all_out_locs_chosen;
}

vector<int> forward_state_t::get_avail_broadcast_srcs(
  int rid, int uid) const
{
  auto const& refi = refis[rid];

  // Have all output locations been assigned?
  // If no, return {}
  for(auto const& out_jid: refi.outs) {
    if(compute_locations[out_jid] < 0) {
      return {};
    }
  }

  auto const& unit = refi.units[uid];

  // Have all input locations have been assigned?
  // If no, return {}
  // Also: collect all src locs to waiting jids
  map<int, vector<int>> srcs;
  for(auto const& inn_jid: unit.deps) {
    int const& inn_loc = compute_locations[inn_jid];
    if(inn_loc < 0) {
      return {};
    }
    srcs[inn_loc].push_back(inn_jid);
  }

  set<int> const& already_moved = started_move_from[rid][uid];

  vector<int> ret;
  ret.reserve(srcs.size());
  for(auto const& [src, inn_jids]: srcs) {
    // If this src has already been moved,
    // con't return this source
    if(already_moved.count(src)) {
      continue;
    }

    // If any of the inputs have not been computed,
    // don't return this source
    for(auto const& jid: inn_jids) {
      if(compute_status[jid] != -1) {
        continue;
      }
    }

    ret.push_back(src);
  }

  return ret;
}

bool forward_state_t::notify_agg_unit_at(int rid, int uid, int src)
{
  auto const& refi = refis[rid];
  auto const& unit = refi.units[uid];

  bool inputs_have_been_given_location = true;
  bool at_src_have_been_computed = true;
  for(auto const& inn_jid: unit.deps) {
    int const& maybe_loc = compute_locations[inn_jid];
    if(maybe_loc < 0) {
      inputs_have_been_given_location = false;
      pending_location_choices.push(inn_jid);
    } else if(maybe_loc == src) {
      if(compute_status[inn_jid] != -1) {
        at_src_have_been_computed = false;
      }
    }
  }

  return inputs_have_been_given_location && at_src_have_been_computed;
}

void forward_state_t::decrement_refi_status_and_maybe_dep_joins(
  int rid, int dst)
{
  int& refi_cnt = refi_status.at({rid, dst});
  refi_cnt -= 1;
  if(refi_cnt == 0) {
    refi_status.erase({rid, dst});

    decrement_dep_joins(rid, dst);
  }
}

void forward_state_t::decrement_dep_joins(int rid, int loc) {
  auto const& refi = refis[rid];
  for(auto const& jid: refi.outs) {
    int const& jid_loc = compute_locations[jid];
    if(loc == jid_loc) {
      int& join_cnt = compute_status[jid];
      join_cnt -= 1;
      if(join_cnt == 0) {
        apply_workers[loc].add_to_pending(jid);
      }
    }
  }
}

bool operator==(tl_move_t const& lhs, tl_move_t const& rhs)
{
  return two_tuple_eq(lhs, rhs);
}

