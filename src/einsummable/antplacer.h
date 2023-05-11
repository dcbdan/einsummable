#pragma once
#include "setup.h"

#include "coster.h"

struct ant_graph_t {
  // use the graph placements to create an ant graph
  static
  tuple<
    map<int, tensor_t<int>> // map grom graph id to tensor of ant graph ids
    equal_items_t<int>,     // these ant_graph ids are suggested to have
                            //   the same location assignment
    ant_graph_t>
  make(graph_t const& graph);

  void insert_edge(int inn, int out, uint64_t cost, int ident);

  int insert_node(uint64_t cost);

  void print_graphviz(std::ostream& out) const;

  // The idea is that node id produces a tensor. Outgoing
  // nodes from id may require arbitrary portions of node id's produced
  // tensor. Each one of these portions is given a unique ident to disambiguate
  // the different portions.
  struct edge_t {
    int id;
    int ident;
  };

  // This node can depend on different portions of the same input id.
  struct node_t {
    uint64_t cost;
    set<edge_t> inns;
    set<int> outs;

    set<int> get_inputs() const;
  };

  vector<edge_t> outgoing_edges(int id) const;

  vector<node_t> nodes;

  map<edge_t, uint64_t> edge_costs;
};

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

struct ant_interface_t {
  using move_t = ant_graph_t::edge_t;

  // loc, pending -> which apply
  std::function<int(
    int, vector<int> const&)> choose_apply;

  // src, dst, pending -> which pending
  std::function<int(
    int, int, vector<move_t> const&)> choose_move;

  // compute node -> which location
  std::function<int(
    int)> choose_location;
};

struct ant_state_t {
  ant_state_t(
    cluster_t const& cluster,
    ant_graph_t const& ant_graph,
    equal_items_t<int> const& equal_compute_locations);

  using move_t = ant_graph_t::edge_t;

  struct completed_t {
    completed_t(int src, int dst, int id, int ident)
      : c(done_move_t{ src, dst, id, ident })
    {}
    completed_t(int loc, int id)
      : c(done_apply_t{ loc, id })
    {}

    struct done_move_t {
      int src;
      int dst;
      int id;
      int ident;
    };
    struct done_apply_t {
      int loc;
      int id;
    };

    bool did_move()  const { return std::holds_alternative<done_move_t>(c);  }
    bool did_apply() const { return std::holds_alternative<done_apply_t>(c); }

    done_move_t  const& get_move_info()  { return std::get<done_move_t>(c);  }
    done_apply_t const& get_apply_info() { return std::get<done_apply_t>(c); }

  private:
    std::variant<done_move_t, done_apply_t> c;
  };

  // return the start, finish op of what just completed
  tuple<double, double, completed_t>
  step(ant_interface_t const& interface);

  bool all_done() const;

  bool can_be_moved(int id) const;

  void add_move_to_pending(int id);

  // is_apply,which
  tuple<bool,int> get_next_finish() const;

private:
  cluster_t const& cluster;
  ant_graph_t const& ant_graph;
  equal_items_t<int> const& equal_compute_locations;

  // TODO implement fixed compute locations
  //   map<int, int> const& fixed_compute_locations

  vector<worker_t<int>   > apply_workers;
  vector<worker_t<move_t>> move_workers;

  // map src,dst to an index
  map<tuple<int,int>, int> const& to_move_worker;

  // This tensor-fraction has ended up at these locations
  map<move_t, set<int>> fraction_at;

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
};

bool operator==(ant_graph_t::edge_t const& lhs, ant_graph_t::edge_t const& rhs);
bool operator< (ant_graph_t::edge_t const& lhs, ant_graph_t::edge_t const& rhs);

