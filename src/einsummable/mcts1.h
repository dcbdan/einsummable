#pragma once
#include "setup.h"

#include "forwardsim.h"

namespace mcts1_ns {

struct locset_t {
  int min_loc;
  int max_loc;
  // all locs [min_loc,max_loc)
};

struct place_choice_t {
  locset_t locset;
  bool by_agg_group;
  bool with_load_balance;
};

struct leaf_t {};

using choice_t = std::variant<
  vector<partition_t>,
  vector<place_choice_t>,
  leaf_t
>;

struct node_t {
  int up;

  // map from choice to child id
  map<int, int> children;

  int which;

  choice_t choice;

  bool is_leaf() const {
    return std::holds_alternative<leaf_t>(choice);
  }

  vector<partition_t>& child_parts() {
    return std::get<vector<partition_t>>(choice);
  }
  vector<partition_t> const& child_parts() const {
    return std::get<vector<partition_t>>(choice);
  }
  vector<place_choice_t> const& child_places() const {
    return std::get<vector<place_choice_t>>(choice);
  }

  int num_possible_children() const {
    if(is_leaf()) {
      return 0;
    }
    if(std::holds_alternative<vector<partition_t>>(choice)) {
      return child_parts().size();
    }
    if(std::holds_alternative<vector<place_choice_t>>(choice)) {
      return child_places().size();
    }
    throw std::runtime_error("invalid num possible children");
  }

  double best_makespan;
};

struct tree_t {
  tree_t(
    graph_t const& graph,
    cluster_t const& cluster);

  // simulate and return true if nodes
  // were added to the tree
  bool step();

  double get_best_makespan() const;

  bool is_leaf(int id) const;

  double simulate(int leaf_id) const;

private:
  graph_t const& graph;
  cluster_t const& cluster;

  vector<int> const ordered_gids;

  vector<locset_t> const locsets;

  double best_makespan;
  int best_leaf;

  vector<node_t> nodes;

private:
  tuple<double, int> _step();
  int _step_select_which_choice(int id);

  tuple<double, int> _step_init();

  void update_from_leaf(int leaf_id, double found_makespan);

  int get_part_child(
    int id,
    int choice);
  int get_place_child(
    int id,
    int choice,
    // gid -> partition
    map<int, partition_t> const& parts_before);

  // list of (node, choice taken at node)
  vector<tuple<int,int>> get_path_from_leaf(int leaf) const;

  void assign_locations(forward_state_t& state, int gid, place_choice_t const&) const;

  vector<partition_t>
  partition_choices(
    int gid,
    vector<partition_t> const& inn_parts) const;
};

}

bool operator==(mcts1_ns::locset_t const& lhs, mcts1_ns::locset_t const& rhs);

