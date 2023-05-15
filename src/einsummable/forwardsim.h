#pragma once
#include "setup.h"

#include "cluster.h"
#include "twolayergraph.h"

template <typename T>
struct worker_t {
  worker_t() {}

  bool is_in_progress() const {
    return bool(in_progress);
  }

  tuple<double, double, T> const& get_in_progress() const {
    return in_progress.value();
  }

  vector<T> const& get_pending() const {
    return pending;
  }

  void finish_work() {
    in_progress.reset();
  }

  void add_to_pending(T const& new_work) {
    for(auto const& pending_work: pending) {
      if(new_work == pending_work) {
        return;
      }
    }
    pending.push_back(new_work);
  }

  void start_work(int which_pending, double time_now, double finish_time) {
    if(is_in_progress()) {
      throw std::runtime_error("cannot start work");
    }
    T const& work = pending[which_pending];
    in_progress = {time_now, finish_time, work};

    pending.erase(pending.begin() + which_pending);
  }

private:
  // this is in progress and will be done at this time
  optional<tuple<double, double, T>> in_progress;

  // these things can happen
  vector<T> pending;
};

// What is moved?
//   An agg unit is broadcast.
//   An agg unit has dependencies I and outputs O. All of these need
//     locations to be assigned. Once that has happend, collect
//     all the loacations from the dependencies L(I) and the outputs
//     L(O). There needs to be a move for each (src,dst) \in L(I) x L(J).
//
// Move data from tensor jid
// at the refinement rid's which unit.
struct tl_move_t {
  int rid;
  int uid;
};

bool operator==(tl_move_t const& lhs, tl_move_t const& rhs);

struct decision_interface_t {
  // loc, pending -> which apply
  std::function<int(
    int, vector<int> const&)> choose_apply;

  // src, dst, pending -> which pending
  std::function<int(
    int, int, vector<tl_move_t> const&)> choose_move;

  // compute node -> which location
  std::function<int(
    int)> choose_location;

  // randomly choose each choice
  static decision_interface_t random(int nloc);
};

struct forward_state_t {
  forward_state_t(
    cluster_t const& cluster,
    twolayergraph_t const& twolayer,
    equal_items_t<int> const& equal_compute_locations);
  forward_state_t(
    cluster_t const& cluster,
    twolayergraph_t const& twolayer,
    equal_items_t<int> const& equal_compute_locations,
    // for each jid, fix these location; -1 values will get chosen
    vector<int> const& compute_locations);
  forward_state_t(
    cluster_t const& cluster,
    twolayergraph_t const& twolayer,
    equal_items_t<int> const& equal_compute_locations,
    // for each jid, fix these location
    map<int,int> const& compute_locations);
  // Note: setting compute locations directly may override
  //       equal_compute_locations. For instance, 0==1,
  //       but compute_locations@0 = 9, @1 = 10,
  //       then the result will not have 0==1

  struct completed_t {
    completed_t(int src, int dst, int rid, int uid, uint64_t size)
      : c(done_move_t{ src, dst, rid, uid, size })
    {}
    completed_t(int loc, int jid, uint64_t flops)
      : c(done_apply_t{ loc, jid, flops })
    {}

    struct done_move_t {
      int src;
      int dst;
      int rid;
      int unit;
      uint64_t size;
    };
    struct done_apply_t {
      int loc;
      int jid;
      uint64_t flops;
    };

    bool did_move()  const { return std::holds_alternative<done_move_t>(c);  }
    bool did_apply() const { return std::holds_alternative<done_apply_t>(c); }

    done_move_t  const& get_move_info()  const { return std::get<done_move_t>(c);  }
    done_apply_t const& get_apply_info() const { return std::get<done_apply_t>(c); }

  private:
    std::variant<done_move_t, done_apply_t> c;
  };

  // return the start, finish op of what just completed
  tuple<double, double, completed_t>
  step(decision_interface_t const& interface);

  bool all_done() const;

private:
  // Find the earliest finish time of work currently
  // being processed, set that worker to no longer busy
  // and return what it was working on
  tuple<double, double, completed_t> pop_work();

  // Once a location has been assigned, it may
  // be the case that an agg unit is now available
  void process_assigned_location(int jid);
  // Note: if this guy has 0 dependencies, it must be
  //       added to pending once a location is chosen

  // Once a join completes, the outgoing agg units at
  // the computed location have one less dependent
  void process_completed_join(int jid);

  // A join at src has just completed, so propagate that information
  // in agg_moves_in_progress
  void notify_agg_moves_in_progress(int rid, int uid, int src);
  // When an agg move is completed, the actual moves can be
  // scheduled and when a move isn't required (src->src)
  // call process_completed_move directly
  void broadcast_agg_move(int rid, int uid, int src);

  // Once a move is completed, an agg unit at dst is that
  // move closer to being complete.
  // (This should be called even when the src location
  //  is dst and thus a physical move didn't actually happen)
  void process_completed_move(int rid, int uid, int dst);

  // Once an agg unit at some dst has completed,
  // a the corresponding refinement has one less dependent
  void process_completed_agg_unit(int rid, int uid, int dst);

  // Once a refinment at dst has completed, the outgoing joins at
  // at dst have one less dependency to wait for
  void process_completed_refi(int rid, int dst);

  // Once all input and output locations have been assigned
  // for an agg unit, where data needs to be moved is known.
  //
  // This creates agg units, dst pairs
  void process_avail_agg_unit(int rid, int uid);
  // Return whether or not all input and output locations
  // of this agg unit has been assigned
  bool is_avail_agg_unit(int rid, int uid) const;
  // Get the agg unit srcs/dsts. If the agg unit is not available,
  // this will return none. (For the dsts case, all agg units under
  // a single rid have the same dsts, so just the rid is needed)
  optional<set<int>> get_agg_unit_srcs(int rid, int uid) const;
  optional<set<int>> get_refi_dsts(int rid) const;

  // add all jid neccessary to make this agg unit avilable
  // that aren't yet given a location to pending location
  // choices
  void choose_agg_unit_locs(int rid, int uid);
private:
  cluster_t const& cluster;
  twolayergraph_t const& twolayer;

  // these are just aliases
  vector<twolayergraph_t::join_t>       const& joins;
  vector<twolayergraph_t::refinement_t> const& refis;

  equal_items_t<int> const& equal_compute_locations;

  vector<worker_t<int>   > apply_workers;
  vector<worker_t<tl_move_t>> move_workers;

  // map src,dst to an index
  map<tuple<int,int>, int> const& to_move_worker;

  // These tensors get computed here
  // (-1 == not yet assigned)
  vector<int> compute_locations;

  // -1 = has been comptued
  //  0 = can be computed or is being computed
  // >0 = this many tensor fractions must be moved or computed
  vector<int> compute_status;

  int num_compute_remaining;

  std::queue<int> pending_location_choices;

  float time;

  // rid,uid,src -> number of joins left to be completed
  //                at src until a broadcast can occur
  map<tuple<int,int,int>, int> agg_moves_in_progress;
  // rid,uid,dst -> number of moves from each src left
  map<tuple<int,int,int>, int> agg_units_in_progress;
  // rid,dst -> number of agg units left
  map<tuple<int,int>, int> refis_in_progress;
};

