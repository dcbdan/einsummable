#include "forwardsim.h"

#include <thread>
#include <mutex>
#include <condition_variable>

forward_state_t::forward_state_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl)
  : forward_state_t(c, tl, ecl, vector<int>(tl.joins.size(), -1))
{}

vector<int> _setup_compute_locations(
  int nj,
  map<int,int> const& locs)
{
  vector<int> ret(nj, -1);
  for(auto const& [jid,loc]: locs) {
    if(jid < 0 || jid >= nj) {
      throw std::runtime_error("invalid compute locations map");
    }
    ret[jid] = loc;
  }
  return ret;
}

forward_state_t::forward_state_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl,
  map<int,int> const& compute_locations)
  : forward_state_t(c, tl, ecl,
      _setup_compute_locations(tl.joins.size(), compute_locations))
{}

forward_state_t::forward_state_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl,
  vector<int> const& cl)
  : cluster(c),
    twolayer(tl),
    joins(tl.joins),
    refis(tl.refinements),
    equal_compute_locations(ecl),
    apply_workers(c.devices.size()),
    move_workers(c.connections.size()),
    to_move_worker(cluster.to_connection),
    compute_locations(cl),
    compute_status(tl.joins.size()),
    num_compute_remaining(tl.joins.size()),
    time(0.0)
{
  if(compute_locations.size() != joins.size()) {
    throw std::runtime_error("invalid provided compute locations");
  }

  // See if any more compute locations can be deduced
  {
    vector<tuple<int, int>> init_locs;
    for(int jid = 0; jid != joins.size(); ++jid) {
      int const& loc = compute_locations[jid];
      if(loc != -1) {
        init_locs.emplace_back(jid, loc);
      }
    }
    for(auto const& [fixed_jid,loc]: init_locs) {
      if(equal_compute_locations.has(fixed_jid)) {
        for(int const& jid: equal_compute_locations.get_at(fixed_jid)) {
          if(compute_locations[jid] == -1) {
            compute_locations[jid] = loc;
          }
        }
      }
    }
  }

  // 1. set compute_status
  // 2. add all unassigned inputs to pending location choices
  // 3. process all initially assigned locations
  // 4. throw an error if compue_locations is not valid
  int n_locs = apply_workers.size();
  for(int jid = 0; jid != joins.size(); ++jid) {
    auto const& join = joins[jid];
    int num_inns = join.deps.size();
    compute_status[jid] = num_inns;

    int const& loc = compute_locations[jid];
    if(loc == -1) {
      if(num_inns == 0) {
        pending_location_choices.push(jid);
      }
    } else if(loc >= 0 && loc < n_locs) {
      // this location has already been assigned, so update this state
      process_assigned_location(jid);
    } else if(loc < -1) {
      throw std::runtime_error("invalid compute location: is < -1");
    } else if(loc >= n_locs) {
      throw std::runtime_error("invalid compute location: is > num locs in cluster");
    } else {
      throw std::runtime_error("that should've been all the cases");
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
      if(pending.size() == 0) {
        continue;
      }

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
      if(pending.size() == 0) {
        continue;
      }

      int which = interface.choose_move(src, dst, pending);

      auto const& [rid,uid] = pending[which];
      uint64_t const& size = refis[rid].units[uid].size;
      double work_time = cluster.move(src, dst, sizeof(float)*size);

      move_worker.start_work(which, time, time + work_time);
    }
  }

  // get the next thing to finish
  tuple<double, double, completed_t> ret = pop_work();
  auto const& [_, finished_at, completion] = ret;

  time = finished_at;

  if(completion.did_apply()) {
    auto const& [_0, jid,_1] = completion.get_apply_info();
    process_completed_join(jid);
  } else if(completion.did_move()) {
    auto const& [_0,dst,rid,uid,_1] = completion.get_move_info();
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
    uint64_t const& flops = joins[join].flops;
    apply_worker.finish_work();
    return {start, finish, completed_t(loc, join, flops)};
  } else {
    auto& move_worker = move_workers[which];

    auto const& connection  = cluster.connections[which];
    int const& src = connection.src;
    int const& dst = connection.dst;

    auto [start,finish,move] = move_worker.get_in_progress();
    auto const& [rid, uid] = move;
    uint64_t const& size = refis[rid].units[uid].size;
    move_worker.finish_work();
    return {start, finish, completed_t(src, dst, rid, uid, size)};
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

forward_node_t* forward_node_t::merge_line(forward_node_ptr_t && other) {
  if(children.size() != other->children.size()) {
    throw std::runtime_error("how come the choice sizes are not the same?");
  }

  if(children.size() == 0) {
    // this is a leaf node
    return this;
  }

  int choice = other->singleton_child();

  if(children[choice]) {
    // this choice has been taken; recurse
    forward_node_t& child = *children[choice];
    return child.merge_line(std::move(other->children[choice]));
  } else {
    // this choice has not been taken, merge the child into this tree
    children[choice] = std::move(other->children[choice]);
    // up and root need to be corrected on all the childre node,
    // so do that here
    return children[choice]->fix_merge_line_(root, this);
  }
}

forward_node_t* forward_node_t::fix_merge_line_(
  forward_node_t* root_, forward_node_t* up_)
{
  root = root_;
  up = up_;
  if(children.size() == 0) {
    return this;
  } else {
    int choice = singleton_child();
    return children[choice]->fix_merge_line_(root_, this);
  }
}

forward_node_t* forward_node_t::singleton_leaf() {
  if(children.size() == 0) {
    return this;
  }
  int choice = singleton_child();
  return children[choice]->singleton_leaf();
}

int forward_node_t::singleton_child() const {
  int choice = 0;
  for(; choice != children.size(); ++choice) {
    forward_node_ptr_t const& child = children[choice];
    if(child) {
      break;
    }
  }
  if(choice == children.size()) {
    throw std::runtime_error("all children in are nullptr");
  }
  for(int i = choice + 1; i != children.size(); ++i) {
    forward_node_ptr_t const& child = children[i];
    if(child) {
      throw std::runtime_error("multiple choices here!");
    }
  }
  return choice;
}

vector<tuple<decision_type_t, int>>
forward_node_t::get_decisions_to_here() const
{
  vector<tuple<decision_type_t, int>> ret;

  forward_node_t const* dwn = this;
  forward_node_t const* upp = up;
  while(dwn != root) {
    int d = 0;
    for(; d != upp->children.size(); ++d) {
      if(upp->children[d]) {
        forward_node_t const* child = upp->children[d].get();
        if(child == dwn) {
          break;
        }
      }
    }
    if(d == upp->children.size()) {
      throw std::runtime_error("could not find self");
    }
    ret.emplace_back(upp->decision, d);

    dwn = upp;
    upp = upp->up;
  }

  std::reverse(ret.begin(), ret.end());
  return ret;
}

void forward_node_t::increment_tau_to_here(double delta) {
  forward_node_t* dwn = this;
  forward_node_t* upp = up;
  while(dwn != root) {
    int d = 0;
    for(; d != upp->children.size(); ++d) {
      if(upp->children[d]) {
        forward_node_t* child = upp->children[d].get();
        if(child == dwn) {
          break;
        }
      }
    }
    if(d == upp->children.size()) {
      throw std::runtime_error("could not find self");
    }
    upp->taus[d] += delta;

    dwn = upp;
    upp = upp->up;
  }
}

void forward_node_t::shrink_all_tau(double shrink) {
  vector<forward_node_t*> ptrs;
  ptrs.push_back(this);
  while(ptrs.size() > 0) {
    forward_node_t* self = ptrs.back();
    ptrs.pop_back();
    for(double& tau: self->taus) {
      tau *= shrink;
    }
    for(auto& child: self->children) {
      if(child) {
        ptrs.push_back(child.get());
      }
    }
  }
}

// TODO: should be easy enough to just keep track
//       of the number of nodes
int forward_node_t::num_nodes() const {
  int cnt = 0;
  vector<forward_node_t const*> ptrs;
  ptrs.push_back(this);
  while(ptrs.size() > 0) {
    cnt += 1;
    forward_node_t const* self = ptrs.back();
    ptrs.pop_back();
    for(auto const& child: self->children) {
      if(child) {
        ptrs.push_back(child.get());
      }
    }
  }
  return cnt;
}

forward_manager_t::forward_manager_t::forward_manager_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl)
  : forward_manager_t(c, tl, ecl, vector<int>(tl.joins.size(), -1))
{}

forward_manager_t::forward_manager_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl,
  vector<int> const& cl)
  : cluster(c),
    twolayer(tl),
    equal_compute_locations(ecl),
    compute_locations(cl),
    best(nullptr)
{}

vector<int> _forward_manager_to_vector(
  map<int, int> const& cl,
  int n)
{
  vector<int> ret(n, -1);
  for(auto const& [i,l]: cl) {
    if(i < 0 || i >= n) {
      throw std::runtime_error("invalid mapping given to forwardmanager");
    }
    ret[i] = l;
  }
  return ret;
}

forward_manager_t::forward_manager_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl,
  map<int, int> const& cl)
  : forward_manager_t(c, tl, ecl,
      _forward_manager_to_vector(cl, tl.joins.size()))
{}

forward_node_t* forward_manager_t::merge_line(
  forward_node_ptr_t && new_root,
  forward_manager_t::stats_t const& new_stats)
{
  if(root) {
    forward_node_t* leaf = root->merge_line(std::move(new_root));
    if(new_stats.makespan < best_stats.makespan) {
      best = leaf;
      best_stats = new_stats;
    }
    return leaf;
  } else {
    root = std::move(new_root);
    forward_node_t* leaf = root->singleton_leaf();
    best = leaf;
    best_stats = new_stats;
    return leaf;
  }
}

forward_node_t* forward_manager_t::merge_line(
  tuple<forward_node_ptr_t, forward_manager_t::stats_t> && info)
{
  auto && [a,b] = info;
  return merge_line(std::move(a), b);
}

forward_state_t forward_manager_t::new_state() const {
  return forward_state_t(
    cluster,
    twolayer,
    equal_compute_locations,
    compute_locations);
}

tuple<forward_node_ptr_t, forward_manager_t::stats_t>
forward_manager_t::simulate_once()
{
  forward_state_t state = new_state();

  int nloc = cluster.devices.size();

  forward_node_ptr_t line = std::make_unique<forward_node_t>();
  forward_node_t* node = line.get();

  forward_node_t* base = root ? root.get() : nullptr;

  auto update_choice = [&](decision_type_t decision, int which, int num_options) {
    node->decision = decision;
    node->children = vector<forward_node_ptr_t>(num_options);
    node->taus     = vector<double>(num_options, 1.0);
    node->etas     = vector<double>(num_options, 1.0);
    node->children[which] = std::make_unique<forward_node_t>(node);

    if(decision.is_choose_location()) {
      int const& jid = decision.get_choose_location();
      vector<uint64_t> scores;
      for(int loc = 0; loc != num_options; ++loc) {
        scores.push_back(state.extra_elems_to(jid, loc));
      }
      for(int loc = 0; loc != num_options; ++loc) {
        if(scores[loc] == 0) {
          node->etas[loc] = 2.0;
        }
      }
    }

    if(base) {
      if(base->children[which]) {
        base = base->children[which].get();
      } else {
        base = nullptr;
      }
    }

    node = node->children[which].get();
  };

  decision_interface_t interface {
    .choose_apply = [&](int loc, vector<int> const& pending) {
      update_choice(
        decision_type_t::choose_apply(loc),
        0,
        pending.size());
      return 0;
    },
    .choose_move = [&](int src, int dst, vector<tl_move_t> const& pending) {
      update_choice(
        decision_type_t::choose_move(src, dst),
        0,
        pending.size());
      return 0;
    },
    .choose_location = [&](int id) {
      int loc;
      if(base) {
        vector<double> scores = base->taus;
        for(int i = 0; i != scores.size(); ++i) {
          scores[i] *= base->etas[i];
        }
        loc = runif(scores);
      } else {
        loc = runif(nloc);
      }
      update_choice(
        decision_type_t::choose_location(id),
        loc,
        nloc);
      return loc;
    }
  };

  stats_t stats { 0, 0, 0.0 };
  while(!state.all_done()) {
    auto [_, stop, completed] = state.step(interface);

    stats.makespan = stop;

    if(completed.did_move()) {
      stats.elems_total += completed.get_move_info().size;
    } else {
      stats.flops_total += completed.get_apply_info().flops;
    }
  }

  //DLINEOUT("Makespan: " << stats.makespan);

  return std::make_tuple(std::move(line), stats);
}

void forward_manager_t::step(int num_times, double shrink, double qq) {
  vector<tuple<forward_node_ptr_t, stats_t>> items;
  items.reserve(num_times);
  for(int i = 0; i != num_times; ++i) {
    items.push_back(simulate_once());
  }
  if(root) {
    root->shrink_all_tau(shrink);
  } else {
    throw std::runtime_error("root must be setup");
  }
  for(auto && [line,stats]: items) {
    auto save_line = merge_line(std::move(line), stats);
    save_line->increment_tau_to_here(qq/stats.makespan);
  }
}

void forward_manager_t::simulate(int num_times, int num_threads) {
  if(num_threads == 1) {
    for(int i = 0; i != num_times; ++i) {
      //DLINEOUT(i + 1 << " / " << num_times);
      auto [line, stats] = simulate_once();
      merge_line(std::move(line), stats);
      //DLINEOUT("  num nodes: " << root->num_nodes());
    }
    return;
  }

  vector<tuple<forward_node_ptr_t, stats_t>> sims(num_times);

  int counter = 0;
  std::mutex m;

  std::function<void()> runner = [this, &sims, &counter, &m, num_times] {
    int which;
    while(true) {
      {
        std::unique_lock lk(m);
        which = counter;
        counter += 1;
      }
      if(which >= num_times) {
        return;
      }
      sims[which] = simulate_once();
    }
  };

  vector<std::thread> runners;
  for(int i = 0; i != num_threads; ++i) {
    runners.emplace_back(runner);
  }
  for(auto& t: runners) {
    t.join();
  }

  for(auto && sim: sims) {
    merge_line(std::move(sim));
  }
}

// TODO: this would probably be better if it didn't have to rerun
//       the whole forward state.
vector<int> forward_manager_t::get_best_locations() const
{
  if(!best) {
    throw std::runtime_error("this tree must still be unset");
  }
  auto decisions = best->get_decisions_to_here();

  auto iter = decisions.begin();

  auto next = [&]() {
    auto const& [_, choice] = *iter;
    iter += 1;
    return choice;
  };

  decision_interface_t interface {
    .choose_apply = [&](int, vector<int> const&) {
      return next();
    },
    .choose_move = [&](int, int, vector<tl_move_t> const&) {
      return next();
    },
    .choose_location = [&](int) {
      return next();
    }
  };

  forward_state_t state = new_state();
  while(!state.all_done()) {
    state.step(interface);
  }

  return state.get_compute_locations();
}

uint64_t forward_state_t::extra_elems_to(int jid, int loc) const
{
  // A refi contains agg units.
  // An agg unit is broadcast to all locations it get used and
  // from all locations its inputs are from
  uint64_t total = 0;

  auto const& join = joins[jid];
  for(auto const& rid: join.deps) {
    auto const& refi = refis[rid];

    set<int> out_locs;
    for(auto const& out_jid: refi.outs) {
      int const& out_loc = compute_locations[out_jid];
      if(out_jid != jid && out_loc != -1) {
        out_locs.insert(out_loc);
      }
    }
    if(out_locs.count(loc) == 1) {
      // all agg units will be broadcast to this location
      // already so nothing to add
      continue;
    }

    for(auto const& [size, inn_jids]: refi.units) {
      set<int> inn_locs;
      for(auto const& inn_jid: inn_jids) {
        int const& inn_loc = compute_locations[inn_jid];
        // don't add inn_loc == loc since that move is free
        if(inn_loc != -1 && inn_loc != loc) {
          inn_locs.insert(inn_loc);
        }
      }
      int n = inn_locs.size();
      total += n*size;
    }
  }
  return total;
}

forward_mcts_tree_t::forward_mcts_tree_t(
  cluster_t const& c,
  twolayergraph_t const& tl,
  equal_items_t<int> const& ecl)
  : cluster(c), twolayer(tl), equal_compute_locations(ecl)
{
  forward_state_t state = new_state();

  int jid0 = -1;

  decision_interface_t interface {
    .choose_apply = [](int,      vector<int> const&      ) { return 0; },
    .choose_move  = [](int, int, vector<tl_move_t> const&) { return 0; },
    .choose_location = [&jid0](int jid) {
      if(jid0 == -1) {
        jid0 = jid;
      }
      return 0;
    }
  };

  while(jid0 == -1) {
    // note: this may call the interface functions many times
    state.step(interface);
  }

  nodes.push_back(node_t {
    .jid = jid0,
    .up = -1,
    .children = {},
    .cumul_makespan = 0.0,
    .num_sim = 0
  });
}

forward_state_t forward_mcts_tree_t::new_state() const {
  return forward_state_t(
    cluster,
    twolayer,
    equal_compute_locations);
}

double forward_mcts_tree_t::selection_score(double c, int id) const {
  auto const& node = nodes[id];
  double num_sim = double(node.num_sim);
  double up_num_sim = double(nodes[node.up].num_sim);

  double value = 0.0;
  double cases = node.jid == -1 ? 1.0 : 2.0 ;
  if(node.jid != -1) {
    auto const& [cumul,n] = eq_classes[node.eq_class];
    double avg_m = cumul / double(n);
    value += best.value().makespan / avg_m;
  }

  value += best.value().makespan / (node.cumul_makespan / num_sim);

  return value / cases + c * std::sqrt(
    std::log( up_num_sim ) / node.num_sim
  );
}

optional<int> forward_mcts_tree_t::selection(double c) {
  if(c == 0.0) {
    c = 1.4142135623730951;
  }
  int id = 0;
  while(true) {
    auto const& node = nodes[id];
    if(node.can_expand()) {
      return id;
    } else if(node.jid == -1) {
      return std::nullopt;
    } else {
      auto iter = max_element_transform(
        node.children.begin(),
        node.children.end(),
        [&, this](int const& id) {
          return this->selection_score(c, id);
        });
      id = *iter;
      if(id < 0 || id >= nodes.size()) {
        throw std::runtime_error("__________________");
      }
    }
  }
}

void forward_mcts_tree_t::expand_simulate_backprop(int top)
{
  // simulating each child node from this node
  // each child gets it's original time
  // then propagate the best makespan of these simulations

  if(!nodes[top].can_expand()) {
    throw std::runtime_error("cannot expand");
  }

  int n_locs = cluster.devices.size();
  for(int loc = 0; loc != n_locs; ++loc) {
    auto [info, next_jid, eq_class] = simulate(top, loc);

    if(!best || info.makespan < best.value().makespan) {
      best = info;
    }

    nodes[top].children.push_back(nodes.size());
    nodes.push_back(node_t {
      .jid = next_jid,
      .up = top,
      .children = {},
      .eq_class = eq_class,
      .cumul_makespan = info.makespan,
      .num_sim = 1
    });

    if(next_jid != -1) {
      auto& [v,n] = eq_classes[eq_class];
      v += info.makespan;
      n += 1;
    }
  }

  int const& best_child = *min_element_transform(
    nodes[top].children.begin(),
    nodes[top].children.end(),
    [this](int const& id) {
      return nodes[id].cumul_makespan;
    }
  );
  double const& best_child_makespan = nodes[best_child].cumul_makespan;

  int id = top;
  while(id != 0) {
    auto& node = nodes[id];
    node.cumul_makespan += best_child_makespan;

    if(node.jid == -1) {
      DOUT(id << " <id  it's jid> " << node.jid);
      throw std::runtime_error("this id has a -1 jid in backprop");
    }

    auto& [v,n] = eq_classes[node.eq_class];
    v += best_child_makespan;
    n += 1;

    id = node.up;
  }
}

tuple<forward_mcts_tree_t::sim_info_t, int, int>
forward_mcts_tree_t::simulate(
  int start_id,
  int start_loc)
{
  vector<tuple<int, int>> choices = locations_to(start_id);
  int cnt = choices.size();
  int num_locs = cluster.devices.size();
  int next_id = -1; // this will be -1 if start_id has the last decision
  int eq_class = -1;
  int const& start_jid = nodes[start_id].jid;

  forward_state_t state = new_state();

  decision_interface_t interface {
    .choose_apply = [](int,      vector<int> const&      ) { return 0; },
    .choose_move  = [](int, int, vector<tl_move_t> const&) { return 0; },
    .choose_location = [&, this](int jid) {
      if(cnt == -2 || cnt == -1) {
        int loc;
        if(runif(2) == 1) {
          loc = runif(num_locs);
        } else {
          vector<uint64_t> cnts;
          cnts.reserve(num_locs);
          for(int l = 0; l != num_locs; ++l) {
            cnts.push_back(state.extra_elems_to(jid, l));
          }
          loc = std::min_element(cnts.begin(), cnts.end()) - cnts.begin();
        }
        choices.emplace_back(jid, loc);
        if(cnt == -1) {
          next_id = jid;

          mcts_eq_t eq_item { .jid = jid, .ls = {} };
          {
            int center_cnt = 0;
            map<int, int> to_centered_loc;
            auto const& join = twolayer.joins[jid];
            for(auto const& rid: join.deps) {
              auto const& refi = twolayer.refinements[rid];
              for(auto const& [_, deps]: refi.units) {
                for(auto const& inn_jid: deps) {
                  int const& loc = state.get_compute_locations()[inn_jid];
                  if(to_centered_loc.count(loc) == 0) {
                    to_centered_loc[loc] = center_cnt;
                    center_cnt += 1;
                  }
                  eq_item.ls.push_back(to_centered_loc.at(loc));
                }
              }
            }
          }
          auto iter = eq_class_to_id.find(eq_item);
          if(iter == eq_class_to_id.end()) {
            eq_class = eq_classes.size();
            eq_class_to_id.insert({eq_item, eq_class});
            eq_classes.emplace_back(0.0, 0);
          } else {
            auto const& [_0, ii] = *iter;
            eq_class = ii;
          }

          cnt -= 1;
        }
        return loc;
      } else if(cnt == 0) {
        if(start_jid != jid) {
          throw std::runtime_error("invalid start id");
        }
        choices.emplace_back(jid, start_loc);
        cnt -= 1;
        return start_loc;
      } else {
        auto const& [jid_, loc] = choices[choices.size()-cnt];
        if(jid != jid_) {
          throw std::runtime_error("invliad jid path");
        }
        cnt -= 1;
        return loc;
      }
    }
  };

  double makespan = 0.0;
  while(!state.all_done()) {
    makespan = std::get<1>(state.step(interface));
  }

  return { sim_info_t { makespan, choices }, next_id, eq_class };
}

vector<int> forward_mcts_tree_t::path_to(int id) const {
  vector<int> ret;
  while(id != 0) {
    ret.push_back(id);
    id = nodes[id].up;
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

vector<tuple<int, int>> forward_mcts_tree_t::locations_to(int id) const {
  auto path = path_to(id);

  vector<tuple<int, int>> ret;
  ret.reserve(path.size());
  for(int const& id: path_to(id)) {
    auto const& node = nodes[id];
    auto const& upp = nodes[node.up];
    ret.emplace_back(upp.jid, upp.get_which(id));
  }

  return ret;
}

bool operator==(mcts_eq_t const& lhs, mcts_eq_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}

