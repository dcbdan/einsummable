#include "alocate.h"
#include "buildtrees.h"
#include "loadbalanceplace.h"

#include "../base/permute.h"

using ap_graph_t     = typename build_trees_ns::bgraph_t<unit_t>;
using ap_tree_t      = typename build_trees_ns::btree_t<unit_t>;
using ap_graphtree_t = typename build_trees_ns::bgraph_t<ap_tree_t>;

struct structured_placement_t {
  structured_placement_t(
    int nlocs,
    vector<uint64_t> const& size,
    vector<int> const& splits);

  structured_placement_t(
    int nlocs,
    vector<uint64_t> const& size,
    vector<int> const& splits,
    vector<int> const& priority);

  struct dim_t {
    uint64_t size;
    int split;
    int priority;
  };

  int nlocs;
  vector<dim_t> dims;

  vector<int> get_priorities() const {
    return vector_from_each_member(dims, int, priority);
  }
  vector<int> get_block_shape() const {
    return vector_from_each_member(dims, int, split);
  }
  vector<uint64_t> get_total_shape()  const {
    return vector_from_each_member(dims, uint64_t, size);
  }

  // Suppose the priorities are {1,2,0}.
  //                           di dj dk
  // Then the last dk moves the fastest, followed by di, then dj.
  // So we create a "tensor" of shape (dk,di,dj)
  // with locations 0,1,2,...,nlocs and then we permute
  // the locations kij->ijk.
  // Thats how to build the location tensor
  placement_t as_placement() const {
    vector<partdim_t> pds;
    for(int rank = 0; rank != dims.size(); ++rank) {
      pds.push_back(partdim_t::split(dims[rank].size, dims[rank].split));
    }

    auto out_block_shape = get_block_shape();
    int n_blocks = product(out_block_shape);

    vector<int> locs(n_blocks);
    {
      // get the permutation "kij->ijk"
      vector<int> out_perm;
      {
        vector<int> iota_(dims.size());
        std::iota(iota_.begin(), iota_.end(), 0);
        out_perm = as_out_perm(
          get_priorities(),
          iota_);
      }

      // get the input shape
      vector<uint64_t> inn_block_shape;
      {
        // convert out_block_shape to uint64_t
        vector<uint64_t> out_block_shape_;
        out_block_shape_.reserve(out_block_shape.size());
        for(int const& x: out_block_shape) {
          out_block_shape_.push_back(x);
        }

        inn_block_shape = backward_permute(out_perm, out_block_shape_);
      }

      // the input locations
      vector<int> inn_locs(n_blocks);
      std::iota(inn_locs.begin(), inn_locs.end(), 0);
      for(int& loc: inn_locs) {
        loc = loc % nlocs;
      }

      // now permute directly into locs
      permute_t permuter(1024);
      permuter(inn_block_shape, out_perm, locs.data(), inn_locs.data());
    }

    return placement_t(
      partition_t(pds),
      vtensor_t<int>(out_block_shape, locs));
  }

  // TODO: is this needed?
  placement_t as_placement_chopped(int new_rank) const {
    int full_rank = dims.size();

    if(new_rank > full_rank) {
      throw std::runtime_error("invalid rank to chop to");
    }

    if(new_rank == full_rank) {
      return as_placement();
    }

    placement_t full_pl = as_placement();

    partition_t new_partition = [&] {
      auto const& full_pds = full_pl.partition.partdims;
      vector<partdim_t> pds(
        full_pds.begin(),
        full_pds.begin() + new_rank);
      return partition_t(pds);
    }();

    vector<int> new_locs;
    {
      auto full_block_shape = full_pl.block_shape();

      vector<tuple<int,int>> region;
      region.reserve(full_rank);
      for(int r = 0; r != new_rank; ++r) {
        region.emplace_back(0, full_block_shape[r]);
      }
      for(int r = new_rank; r != full_rank; ++r) {
        region.emplace_back(0, 1);
      }

      new_locs = full_pl.locations.subset(region).get();
    }

    return placement_t(
      new_partition,
      vtensor_t<int>(new_partition.block_shape(), new_locs));
  }
};

vector<int> _default_priority(int n) {
  std::vector<int> ret(n);
  std::iota(ret.begin(), ret.end(), 0);
  return ret;
}

structured_placement_t::structured_placement_t(
  int nlocs,
  vector<uint64_t> const& size,
  vector<int> const& splits)
  : structured_placement_t(nlocs, size, splits, _default_priority(splits.size()))
{}

structured_placement_t::structured_placement_t(
  int nlocs,
  vector<uint64_t> const& sizes,
  vector<int> const& splits,
  vector<int> const& priority)
  : nlocs(nlocs)
{
  int rank = sizes.size();
  if(rank != splits.size() || rank != priority.size()) {
    throw std::runtime_error("different input ranks");
  }
  if(nlocs <= 0) {
    throw std::runtime_error("invalid nlocs");
  }
  dims.reserve(rank);
  for(int r = 0; r != rank; ++r) {
    dims.push_back(dim_t {
      .size = sizes[r],
      .split = splits[r],
      .priority = priority[r]
    });
  }
}

struct _lplan_t {
  placement_t placement;
  vector<int> which;
  uint64_t cost;
};

bool operator<(_lplan_t const& lhs, _lplan_t const& rhs) {
  return lhs.cost < rhs.cost;
}

void _add_to_placements(
  vector<placement_t>& placements,
  map<int, vector<_lplan_t>>& solved,
  ap_tree_t const& tree,
  int start_id)
{
  auto get_which_best_plan = [](vector<_lplan_t> const& plans)
  {
    auto iter = std::min_element(plans.begin(), plans.end());
    return std::distance(
      plans.begin(),
      iter);
  };

  vector<tuple<int, int>> to_process;
  to_process.emplace_back(start_id, get_which_best_plan(solved.at(start_id)));

  while(to_process.size() > 0) {
    auto [id, which] = to_process.back();
    to_process.pop_back();

    auto const& plan = solved.at(id)[which];
    placements[id] = plan.placement;

    // Note: it may be the case that
    //   plan.which.size() < tree.get_inns_as_vec(id).size()
    auto const& next_which = plan.which;

    vector<int> next_inns = tree.get_inns_as_vec(id);
    next_inns.resize(next_which.size());

    vector_concatenate_into(
      to_process,
      vector_zip(next_inns, next_which));

    solved.erase(id);
  }
}

vector<structured_placement_t>
make_structured_placements(
  int nlocs,
  vector<uint64_t> const& size,
  vector<int> const& splits)
{
  int rank = splits.size();

  auto make_priority = [&](vector<int> const& top_ranks) {
    vector<int> ret(rank);
    int c = 0;
    for(int const& which: top_ranks) {
      ret[which] = c++;
    }
    for(int r = 0; r != rank; ++r) {
      auto iter = std::find(top_ranks.begin(), top_ranks.end(), r);
      if(iter == top_ranks.end()) {
        ret[r] = c++;
      }
    }
    return ret;
  };

  vector<int> top_ranks;
  {
    auto iters = select_topn(splits.begin(), splits.end(), 3);
    for(auto iter: iters) {
      if(*iter > 1) {
        top_ranks.push_back(std::distance(splits.begin(), iter));
      }
    }
  }

  vector<structured_placement_t> ret;
  if(top_ranks.size() <= 1) {
    ret.emplace_back(nlocs, size, splits);
  } else if(top_ranks.size() == 2) {
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[0], top_ranks[1]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[1], top_ranks[0]}));
  } else if(top_ranks.size() == 3) {
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[0], top_ranks[1], top_ranks[2]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[0], top_ranks[2], top_ranks[1]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[1], top_ranks[0], top_ranks[2]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[1], top_ranks[2], top_ranks[0]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[2], top_ranks[0], top_ranks[1]}));
    ret.emplace_back(nlocs, size, splits, make_priority({top_ranks[2], top_ranks[1], top_ranks[0]}));
  } else {
    throw std::runtime_error("should not reach: make_structured_placements");
  }

  return ret;
}

vector<placement_t> _get_possible_placements(
  int nloc,
  partition_t const& part)
{
  vector<uint64_t> total_shape = part.total_shape();
  int rank = total_shape.size();

  vector<int> splits;
  splits.reserve(rank);
  for(int r = 0; r != rank; ++r) {
    splits.push_back(part.partdims[r].num_parts());
  }

  auto structured_placements = make_structured_placements(nloc, total_shape, splits);

  auto ret = vector_from_each_method(structured_placements, placement_t, as_placement);
  for(auto const& pl: ret) {
    if(pl.partition != part) {
      throw std::runtime_error(
        "the partitions are not evenly split! _get_possible_placements");
    }
  }

  return ret;
}

void _solve_locate_tree(
  vector<placement_t>& placements,
  int nloc,
  graph_t const& graph,
  ap_tree_t const& tree,
  int max_branching)
{
  map<int, vector<_lplan_t>> solved;
  auto all_gids = tree.dfs_from_root();
  std::reverse(all_gids.begin(), all_gids.end());
  for(int const& gid: all_gids) {
    vector<int> tree_inns = tree.get_inns_as_vec(gid);
    if(tree_inns.size() > max_branching) {
      for(int i = max_branching; i != tree_inns.size(); ++i) {
        int const& inn_gid = tree_inns[i];
        _add_to_placements(placements, solved, tree, inn_gid);
      }
    }

    int num_tree_inns = std::min(max_branching, int(tree_inns.size()));
    tree_inns.resize(num_tree_inns);
    bool has_tree_inns = num_tree_inns > 0;

    vector<int> inn_num_plans;
    for(int i = 0; i != num_tree_inns; ++i) {
      int const& inn_gid = tree_inns[i];
      inn_num_plans.push_back(solved.at(inn_gid).size());
    }

    vector<_lplan_t>& plans = solved[gid];

    // the placements already contain all the partitions
    auto const& part = placements[gid].partition;
    for(auto const& placement: _get_possible_placements(nloc, part)) {
      vector<int> best_which;
      optional<uint64_t> best_cost = std::nullopt;

      vector<int> which(num_tree_inns, 0);
      do {
        uint64_t cost_from_children = 0;
        for(int i = 0; i != num_tree_inns; ++i) {
          int const& w = which[i];
          int const& g = tree_inns[i];
          cost_from_children += solved.at(g)[w].cost;
        }

        auto get_placement = [&](int other_gid) -> placement_t const& {
          if(gid == other_gid) {
            return placement;
          }
          for(int i = 0; i != which.size(); ++i) {
            int const& w = which[i];
            int const& g = tree_inns[i];
            if(g == other_gid) {
              return solved.at(g)[w].placement;
            }
          }
          return placements[other_gid];
        };

        uint64_t cost_from_node = compute_tensor_move_cost(graph, get_placement, gid);

        uint64_t cost = cost_from_children + cost_from_node;

        if(!bool(best_cost) || cost < best_cost.value()) {
          best_cost = cost;
          best_which = which;
        }
      } while(has_tree_inns && increment_idxs(inn_num_plans, which));

      plans.push_back(_lplan_t {
        .placement = placement,
        .which = best_which,
        .cost = best_cost.value()
      });
    }
  }

  _add_to_placements(placements, solved, tree, tree.get_root_id());
}

vector<placement_t> autolocate(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs)
{
  // 0. set up the outputs
  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    ret.emplace_back(part);
  }

  // 1. build a graph of trees
  ap_graphtree_t btrees;
  {
    ap_graph_t bgraph;
    for(int const& gid: graph.get_reverse_order()) {
      auto const& node = graph.nodes[gid];

      // Going from graph outs to bgraph inns since we'll be
      // costing edges from the graph-outs
      set<int> const& inns = node.outs;

      bgraph.insert(gid, unit_t{}, inns);
    }
    btrees = build_trees_ns::build_trees(bgraph);
  }

  // 2. solve each tree, in the right order
  for(int const& root_id: btrees.dag_order_inns_to_outs()) {
    auto const& tree = btrees[root_id];
    _solve_locate_tree(ret, nlocs, graph, tree, 2);
  }

  return ret;
}
