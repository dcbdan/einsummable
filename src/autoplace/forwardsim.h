#pragma once
#include "../base/setup.h"

#include "../base/hrect.h"

#include "../einsummable/graph.h"
#include "cluster.h"

// Assumption: all time >= 0; all util >= 1
struct capacity_scheduler_t {
  capacity_scheduler_t(int c);

  double schedule(int util, double min_start, double compute_time);

  void complete(int util, double start, double finish);

  int const capacity;

private:
  struct block_t {
    double beg;
    double end;
    int cnt;
  };

  // Invariant: holds interval [0,max double)
  vector<block_t> blocks;

  using iter_t = vector<block_t>::iterator;

  tuple<iter_t, iter_t>
  find_available(int util, double min_start, double time);

  tuple<int, iter_t> get_avail(iter_t iter, double end);

  iter_t get_exact_start(double t);
  iter_t get_exact_finish(double t);

  void merge_zeros();
};

template <typename T>
struct in_progress_t {
  int util;
  double beg;
  double end;
  T payload;
};

template <typename T>
bool operator>(
  in_progress_t<T> const& lhs,
  in_progress_t<T> const& rhs)
{
  return lhs.end > rhs.end;
}

template <typename T>
struct capacity_worker_t {
  capacity_worker_t(int capacity)
    : scheduler(capacity)
  {}

  bool is_in_progress() const {
    return in_progress.size() > 0;
  }

  in_progress_t<T> const& get_in_progress() const {
    return in_progress.top();
  }

  vector<T> const& get_pending() const {
    return pending;
  }

  T const& get_pending(int which) const {
    return pending[which];
  }

  void finish_work() {
    auto const& p = get_in_progress();
    scheduler.complete(p.util, p.beg, p.end);
    in_progress.pop();
  }

  void add_to_pending(T const& new_work) {
    for(auto const& pending_work: pending) {
      if(new_work == pending_work) {
        return;
      }
    }
    pending.push_back(new_work);
  }

  void start_work(
    int which_pending,
    int util,
    double time_now,
    double total_work_time)
  {
    double start_time = scheduler.schedule(
      util, time_now, total_work_time);

    T const& work = pending[which_pending];

    in_progress.push(in_progress_t<T> {
      .util = util,
      .beg = start_time,
      .end = start_time + total_work_time,
      .payload = work
    });

    pending.erase(pending.begin() + which_pending);
  }

private:
  capacity_scheduler_t scheduler;

  priority_queue_least<in_progress_t<T>> in_progress;

  // these things can happen
  vector<T> pending;
};

template <typename T>
struct worker_t {
  worker_t() {}

  bool is_in_progress() const {
    return in_progress.size() > 0;
  }

  tuple<double, double, T> const& get_in_progress() const {
    return in_progress.front();
  }

  vector<T> const& get_pending() const {
    return pending;
  }

  T const& get_pending(int which) const {
    return pending[which];
  }

  void finish_work() {
    in_progress.pop();
  }

  void add_to_pending(T const& new_work) {
    for(auto const& pending_work: pending) {
      if(new_work == pending_work) {
        return;
      }
    }
    pending.push_back(new_work);
  }

  void start_work(int which_pending, double time_now, double total_work_time) {
    double start_time;
    if(is_in_progress()) {
      double const& last_finish = std::get<1>(in_progress.back());
      start_time = std::max(time_now, last_finish);
    } else {
      start_time = time_now;
    }

    T const& work = pending[which_pending];
    in_progress.push({start_time, start_time + total_work_time, work});

    pending.erase(pending.begin() + which_pending);
  }

private:
  // all of these things will happen in fifo order
  std::queue<tuple<double, double, T>> in_progress;

  // these things can happen
  vector<T> pending;
};

// Actions:
//   partition
//   enqueue worker
//   assign_jid_loc
//   pop_work
// Events:
//   assign partition (gid)
//   setup joins (gid)
//   assign location (jid)
//   completed move (rid,uid,dst)
//   completed agg unit (rid,uid,dst)
//   completed refinement (rid,dst)
//   completed join (jid)
struct forward_state_t {
  struct jid_t {
    int gid;
    int bid;
  };
  struct rid_t {
    int gid;
    int bid;
  };

  struct completed_t {
    completed_t(double b, double e, int src, int dst, int gid, int bid, int uid, uint64_t size)
      : start(b), finish(e), c(done_move_t{ src, dst, gid, bid, uid, size })
    {}
    completed_t(double b, double e, int loc, int gid, int bid, uint64_t flops)
      : start(b), finish(e), c(done_apply_t{ loc, gid, bid, flops })
    {}

    struct done_move_t {
      int src;
      int dst;
      int gid;
      int bid;
      int unit;
      uint64_t size;
    };
    struct done_apply_t {
      int loc;
      int gid;
      int bid;
      uint64_t flops;
    };

    bool did_move()  const { return std::holds_alternative<done_move_t>(c);  }
    bool did_apply() const { return std::holds_alternative<done_apply_t>(c); }

    done_move_t  const& get_move_info()  const { return std::get<done_move_t>(c);  }
    done_apply_t const& get_apply_info() const { return std::get<done_apply_t>(c); }

    double start;
    double finish;

  private:
    std::variant<done_move_t, done_apply_t> c;
  };

  // An agg unit is something that will get summed.
  // So if Y = X1 + X2 + X3 + X4 at locations
  //       0    0   1    1    2
  // then X1 is not moved,
  //      X2 and X3 are summed at location 1 and moved
  //      X4 is moved.
  // An agg unit depends on a set of join ids (X1,X2,X3,X4)
  // but not neccessarily on all elements of the join ids.
  // The size variable how much of each input it takes.
  struct agg_unit_t {
    uint64_t size;
    vector<int> deps; // these are the join bids of the graph node that
                      // this agg unit belongs to
  };

  // A refinement is called such because when constructing from
  // a graph object, this is one block of the refinement partition
  // of all usages of some graph node.
  //
  // Consider Y = UnaryElementwiseOp(X) where X is partitioned into 2x2 blocks
  // and Y is a formation and the only usage, partitioned into 3x3 blocks.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   just Unary(X[0,0]) and the size is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit has one of the
  //   Unary(X[i,j]) blocks for i,j in [0,1]x[0,1]. The size of each agg unit block
  //   is roughly 1/4 the size of the Y[1,1] block.
  // Since this is an elementwise op, each agg unit is just a copy and does no
  //   actual summation.
  //
  // Consider instead Y = (ijk->ij, X) where X is partition into 2x2x3 blocks
  // and Y is partitioned again into 3x3 blocks. Everything holds the same
  // as before except each agg unit has 3 inputs.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   (ijk->ij, X[0,0,k]) for k=0,1,2 and the size is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit represents some i,j
  //   and that agg unit has blocks (ijk->ij, X[i,j,k]) for k=0,1,2.
  //   The size of each agg unit block is roughly 1/4 the size of the Y[1,1] block.
  struct refinement_t {
    vector<agg_unit_t> units;
    set<jid_t> outs;
  };

  struct join_t {
    optional<einsummable_t> einsummable;
    vector<rid_t> deps;
    set<int> outs; // get all refinement bids that depend on this join
  };

  struct unit_status_t {
    unit_status_t();

    // this should be true once all input
    // joins have been assigned a location
    bool is_setup;

    // src -> # of joins left until move from
    //        src can occur
    map<int, int> num_join_rem;
    // if is_setup, num_join_rem.keys() contains all src
    // this relies on

    // dst -> # of moves from each src left
    map<int, int> num_move_rem;
    // num_move_rem.keys() contains all dst that this should
    // end up at. It should be updated whenever a new dst is added
    // (if is_setup is true)

    set<int> dsts() const;
  };

  struct move_status_t {
    move_status_t(int n);

    vector<unit_status_t> unit_status;

    // dst -> number of agg units remaining
    map<int, int> num_unit_rem;
    // num_unit_rem.keys() contains all dst that this
    // should eventually end up at. It should be updated
    // whenever a new dst is added.
  };

  struct graph_node_info_t {
    graph_node_info_t();

    optional<partition_t> partition;
    optional<vector<int>> locs;

    // -1 = has been comptued
    //  0 = can be computed or is being computed
    // >0 = this many tensor fractions must be moved or computed
    optional<vector<int>> compute_status;

    optional<vector<join_t>> joins;

    optional<partition_t> refinement_partition;
    optional<vector<refinement_t>> refis;
    optional<vector<move_status_t>> move_status;
  };

  forward_state_t(cluster_t const& cl, graph_t const& g);

  bool all_done() const;

  // get all gids that can currently be given a partition
  set<int> const& can_assign_partition() const;

  void assign_partition(int gid, partition_t const& part);

  void assign_location(jid_t jid, int loc);

  void enqueue_apply_worker(int loc, int which);

  void enqueue_move_worker(int src, int dst, int which);

  completed_t pop_work();

  ////////////
  void enqueue_all();

  void print_twolayer_graphviz(std::ostream&) const;

  struct random_settings_t {
    std::function<partition_t(int)> get_part;
    std::function<int(jid_t)> get_loc;
    bool always_enqueue_all;
    bool priority_assign_partition;
    bool priority_assign_location;
    double assign_partition;
    double assign_location;
    double enqueue_apply;
    double enqueue_move;
    double pop_work;
  };
  static random_settings_t random_step_settings(
    std::function<partition_t(int)> get_part,
    std::function<int(jid_t)> get_loc);

  optional<completed_t> random_step(random_settings_t const& settings);

  optional<int> num_join_bid(int gid) const;

  graph_node_info_t const& get_ginfo(int gid) const;

  // Count the number of elements moved if jid is
  // set to have location loc. Only jids dependent
  // on id will be accessed in get_loc.
  //
  // The join must be set at jid and any set locations
  // in this object are ignored.
  uint64_t count_elements_to(
    std::function<int(jid_t)> get_loc,
    jid_t jid,
    int loc) const;

private:
  // ec = Event Completed

  // Once a partition is assigned, it may be the case that
  // a refinement partition of one of the inputs can be setup.
  // Once a partition is assigned, those locations can also
  // be assigned
  void ec_assign_partition(int gid);

  // Once refis (and refinement_partition) is setup, it may be the
  // case that an output graph node can have the the joins setup
  void ec_setup_refis(int gid);

  // Once the joins are setup, it may be possible to add
  // them to the things that can be computed
  // Once the joins are setup, the refis may be setup
  void ec_setup_joins(int gid);

  // Once a location has been assigned, it may
  // be the case that an agg unit is now available
  // or if this is an input, the computation can be completed
  void ec_assign_location(jid_t jid);

  // Once a move is completed, an agg unit at dst is that
  // move closer to being complete.
  // (This should be called even when the src location
  //  is dst and thus a physical move didn't actually happen)
  void ec_move(rid_t rid, int uid, int dst);

  // Once an agg unit at some dst has completed,
  // the corresponding refinement has one less dependent
  void ec_agg_unit(rid_t rid, int dst);

  // Once a refinment at dst has completed, the outgoing joins at
  // dst have one less dependency to wait for
  void ec_refinement(rid_t rid, int dst);

  // Once a join completes, the outgoing agg units at
  // the computed location have one less dependent
  void ec_join(jid_t jid);

private:
  cluster_t const& cluster;
  graph_t const& graph;

  vector<graph_node_info_t> ginfos;

  set<int> can_partition;

  vector<capacity_worker_t<jid_t>> apply_workers;

  // each worker processes rid,uid pairs
  vector<worker_t<tuple<rid_t,int>>> move_workers;

  // map src,dst to an index
  map<tuple<int,int>, int> const& to_move_worker;

  int num_join_remaining;

  float time;

  // TODO: go through and add access methods to refis given rid
  //       and joins given jid and so on
private:
  bool can_setup_joins(int gid) const;
  void setup_joins(int gid);

  bool can_setup_refinement_partition(int gid) const;
  void setup_refinement_partition(int gid);

  bool can_setup_refis(int gid) const;
  void setup_refis(int gid);

  bool can_setup_unit_status(rid_t rid, int uid) const;
  void setup_unit_status(rid_t rid, int uid);

  // Add to the refinement a new dst. If the dst is complete,
  // update the out join.
  void add_refi_dst(rid_t rid, jid_t jid, int dst);

  void schedule_move(rid_t rid, int uid, int src, int dst);
  worker_t<tuple<rid_t, int>>& get_move_worker(int src, int dst);

  void schedule_join(jid_t jid, int loc);

  void insert_refi_out(rid_t rid, jid_t jid);
};

bool operator==(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator!=(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator< (forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator==(forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs);
bool operator!=(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs);
bool operator< (forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs);

std::ostream& operator<<(std::ostream&, forward_state_t::jid_t const&);
std::ostream& operator<<(std::ostream&, forward_state_t::rid_t const&);
