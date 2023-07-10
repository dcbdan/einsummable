#pragma once
#include "../base/setup.h"

#include "relationwise.h"

struct relationwise_mcmc_t {
  relationwise_mcmc_t(
    graph_t const& graph,
    int nlocs,
    int max_blocks,
    double scale_compute,
    double scale_move,
    equal_items_t<int> equal_gids);
  // TODO: incorporate equal_gids
  // TODO: make sure that obvious elementwise ops are added to equal_gids

  // take a step and return whether or not it was
  // accepted
  bool step(double beta);

  vector<placement_t> const& get_best_placements() const {
    return best_placements;
  }

private:
  struct op_t {
    struct greedy_t {
      int gid;
    };
    struct set_directly_t {
      vector<tuple<int, placement_t>> items;
    };
    struct crement_t {
      int gid;
      int dim;
      bool do_increase;
    };

    op_t(greedy_t const& g):       op(g) {}
    op_t(set_directly_t const& s): op(s) {}
    op_t(crement_t const& c):      op(c) {}

    std::variant<greedy_t, set_directly_t, crement_t> op;

    bool is_greedy()       const { return std::holds_alternative<greedy_t>(      op); }
    bool is_set_directly() const { return std::holds_alternative<set_directly_t>(op); }
    bool is_crement()      const { return std::holds_alternative<crement_t>(     op); }

    greedy_t       const& get_greedy()       const { return std::get<greedy_t>(      op); }
    set_directly_t const& get_set_directly() const { return std::get<set_directly_t>(op); }
    crement_t      const& get_crement()      const { return std::get<crement_t>(     op); }

    int candidate_gid() const;
  };

  void change(op_t const& op);
  op_t random_op()             const;
  op_t reverse(op_t const& op) const;

  // For each block in order,
  //   find the location with the best cost and choose that
  void greedy_solve(int gid);

  void update_cost(int64_t compute_delta, int64_t move_delta);
  void update_cost(tuple<int64_t, int64_t> delta);

  double cost_from_scores(int64_t compute, int64_t move) const;
  double cost() const { return cost_from_scores(current_compute, current_move); }

  partition_t crement_partition(op_t::crement_t const& crement) const;

  relationwise_t gwise;

  int max_blocks;
  double scale_compute;
  double scale_move;

  int64_t current_compute;
  int64_t current_move;

  equal_items_t<int> equal_gids;

  double best_cost;
  vector<placement_t> best_placements;
};
