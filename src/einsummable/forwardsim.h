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
};

struct forward_state_t {
  forward_state_t(
    cluster_t const& cluster,
    twolayergraph_t const& twolayer,
    equal_items_t<int> const& equal_compute_locations);

  struct completed_t {
    completed_t(int src, int dst, int rid, int uid)
      : c(done_move_t{ src, dst, rid, uid })
    {}
    completed_t(int loc, int jid)
      : c(done_apply_t{ loc, jid })
    {}

    struct done_move_t {
      int src;
      int dst;
      int rid;
      int unit;
    };
    struct done_apply_t {
      int loc;
      int jid;
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
  tuple<double, double, completed_t> pop_work();

  void add_broadcast_to_pending(int rid, int uid, int src);

  // There is an invariant: whenever all of a refinements outputs become set,
  // it should be the case that the correponding rid refi_status items
  // are initialised. To maintain this invariant, also setup refi status
  // if it should be setup
  bool refi_out_locs_set_and_refi_status_setup(int rid);

  vector<int> get_avail_broadcast_srcs(int rid, int uid) const;

  // Notify that agg unit at (rid, uid) that an input
  // has been computed at src.
  // The agg unit at loc can be moved from when
  // 1. all inputs have been assigned a compute location
  // 2. all inputs at loc have been computed.
  // If (1) does not hold, add the input joins without a location to
  //        pending_location_choices.
  // Return whether or not (1) and (2) hold.
  bool notify_agg_unit_at(int rid, int uid, int src);
  // Only call this directly after a dependent join at src happened

  void decrement_refi_status_and_maybe_dep_joins(int rid, int dst);

  void decrement_dep_joins(int rid, int loc);

private:
  cluster_t const& cluster;
  twolayergraph_t const& twolayer;

  // these are just aliases
  vector<twolayergraph_t::join_t>       const& joins;
  vector<twolayergraph_t::refinement_t> const& refis;

  equal_items_t<int> const& equal_compute_locations;

  // TODO implement fixed compute locations
  //   map<int, int> const& fixed_compute_locations

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

  // for each rid, uid pair, collect the source locations
  // for which a broadcast has already occurred
  vector<vector<set<int>>> started_move_from;

  // This rid,dst pair needs this many more moves to complete
  map<tuple<int, int>, int> refi_status;
  // refi status is not always setup, so have a vector of flags
  // to indicate if the refi has been setup
  vector<char> refi_status_setup;

  int num_compute_remaining;

  std::queue<int> pending_location_choices;

  float time;
};


