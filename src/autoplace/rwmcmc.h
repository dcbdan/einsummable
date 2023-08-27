#pragma once
#include "../base/setup.h"

#include "relationwise.h"

struct relationwise_mcmc_t {
  relationwise_mcmc_t(
    graph_t const& graph,
    kernel_coster_t const& kernel_coster,
    int nlocs,
    int n_threads_per_loc,
    int max_blocks,
    equal_items_t<int> equal_gids);
  // TODO: incorporate equal_gids
  // TODO: make sure that obvious elementwise ops are added to equal_gids

  // take a step and return whether or not it was
  // accepted
  bool step(double beta);

  vector<placement_t> const& get_best_placements() const {
    return best_placements;
  }

  double const& get_best_cost() const {
    return best_cost;
  }

  void set_placements(vector<placement_t> const& pls);

  equal_items_t<int> get_equal_gids() const { return equal_gids; }

  double const& cost() const { return current_cost; }

  relationwise_stat_t make_stat() const { return gwise.make_stat(); }

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
    struct set_notouch_t {
      int gid;
    };

    op_t(greedy_t const& g):       op(g) {}
    op_t(set_directly_t const& s): op(s) {}
    op_t(crement_t const& c):      op(c) {}
    op_t(set_notouch_t const& t): op(t) {}

    std::variant<greedy_t, set_directly_t, crement_t, set_notouch_t> op;

    bool is_greedy()       const { return std::holds_alternative<greedy_t>(      op); }
    bool is_set_directly() const { return std::holds_alternative<set_directly_t>(op); }
    bool is_crement()      const { return std::holds_alternative<crement_t>(     op); }
    bool is_set_notouch()  const { return std::holds_alternative<set_notouch_t>( op); }

    greedy_t       const& get_greedy()       const { return std::get<greedy_t>(      op); }
    set_directly_t const& get_set_directly() const { return std::get<set_directly_t>(op); }
    crement_t      const& get_crement()      const { return std::get<crement_t>(     op); }
    set_notouch_t  const& get_set_notouch()  const { return std::get<set_notouch_t>( op); }

    int candidate_gid() const;
  };

  void change(op_t const& op);
  op_t random_op()             const;
  op_t reverse(op_t const& op) const;

  // For each block in order,
  //   find the location with the best cost and choose that
  void greedy_solve(int gid);

  partition_t crement_partition(op_t::crement_t const& crement) const;

  optional<partition_t> notouch_partition(int gid) const;

  relationwise_t gwise;

  int max_blocks;

  double current_cost;

  equal_items_t<int> equal_gids;

  double best_cost;
  vector<placement_t> best_placements;
};
