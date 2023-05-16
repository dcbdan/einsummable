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

void forward_node_t::merge_line(forward_node_ptr_t && other) {
  if(children.size() != other->children.size()) {
    throw std::runtime_error("how come the choice sizes are not the same?");
  }

  if(children.size() == 0) {
    // this is a leaf node
    return;
  }

  int choice = 0;
  for(; choice != other->children.size(); ++choice) {
    forward_node_ptr_t& child = other->children[choice];
    if(child) {
      break;
    }
  }

  // Test the assumption that there is one and only one choice
  // taken in the line
  // (One could instead recurse on all the child nodes!)
  if(choice == other->children.size()) {
    throw std::runtime_error("all children in line tree are nullptr");
  }
  for(int i = choice + 1; i != other->children.size(); ++i) {
    forward_node_ptr_t& child = other->children[i];
    if(child) {
      throw std::runtime_error("this line took multiple choices here!");
    }
  }

  if(children[choice]) {
    // this choice has been taken; recurse
    forward_node_t& child = *children[choice];
    child.merge_line(std::move(other->children[choice]));
  } else {
    // this choice has not been taken, merge the child into this tree
    children[choice] = std::move(other->children[choice]);
  }
}

void forward_node_t::num_nodes_(int& ret) const {
  ret += 1;
  for(auto& child: children) {
    if(child) {
      child->num_nodes_(ret);
    }
  }
}
int forward_node_t::num_nodes() const {
  int ret = 0;
  num_nodes_(ret);
  return ret;
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
    compute_locations(cl)
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

void forward_manager_t::merge_line(forward_node_ptr_t && new_root) {
  if(root) {
    root->merge_line(std::move(new_root));
  } else {
    root = std::move(new_root);
  }
}

forward_state_t forward_manager_t::new_state() const {
  return forward_state_t(
    cluster,
    twolayer,
    equal_compute_locations,
    compute_locations);
}

forward_node_ptr_t forward_manager_t::simulate() {
  forward_state_t state = new_state();

  int nloc = cluster.devices.size();

  forward_node_ptr_t line = std::make_unique<forward_node_t>();
  forward_node_t* node = line.get();

  forward_node_t* base = root ? root.get() : nullptr;

  auto update_choice = [&](int which, int num_options) {
    node->children = vector<forward_node_ptr_t>(num_options);
    node->children[which] = std::make_unique<forward_node_t>(node);

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
      update_choice(0, pending.size());
      return 0;
    },
    .choose_move = [&](int src, int dst, vector<tl_move_t> const& pending) {
      update_choice(0, pending.size());
      return 0;
    },
    .choose_location = [&](int id) {
      int loc = runif(nloc);
      update_choice(loc, nloc);
      return loc;
    }
  };

  double finish;
  do {
    auto [_0, finish_, _1] = state.step(interface);
    finish = finish_;
  } while(!state.all_done());

  DOUT(finish);

  return line;
}

void forward_manager_t::run(int num_times, int num_threads) {
  if(num_threads == 1) {
    for(int i = 0; i != num_times; ++i) {
      merge_line(simulate());
      DOUT("  num nodes: " << root->num_nodes());
    }
    return;
  }

  vector<forward_node_ptr_t> sims(num_times);

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
      sims[which] = simulate();
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
