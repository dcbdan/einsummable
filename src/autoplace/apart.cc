#include "apart.h"

#include "twolayer.h" // for twolayer_construct_refinement_partition
#include "../base/copyregion.h"

#include "buildtrees.h"

//#include <fstream>

using ap_graph_t     = typename build_trees_ns::bgraph_t<unit_t>;
using ap_tree_t      = typename build_trees_ns::btree_t<unit_t>;
using ap_graphtree_t = typename build_trees_ns::bgraph_t<ap_tree_t>;

struct _plan_t {
  partition_t partition;
  vector<int> which;
  double cost; // TODO: why not uint64_t cost?
};

bool operator<(_plan_t const& lhs, _plan_t const& rhs) {
  return lhs.cost < rhs.cost;
}

double _add_to_ret(
  map<int, partition_t>& ret,
  map<int, vector<_plan_t>>& solved,
  ap_tree_t const& tree,
  int start_id)
{
  auto get_which_best_plan = [](vector<_plan_t> const& plans)
  {
    auto iter = std::min_element(plans.begin(), plans.end());
    return std::distance(
      plans.begin(),
      iter);
  };

  vector<tuple<int, int>> to_process;
  to_process.emplace_back(start_id, get_which_best_plan(solved.at(start_id)));

  double start_cost;
  {
    auto const& [_, which] = to_process[0];
    start_cost = solved.at(start_id)[which].cost;
  }

  while(to_process.size() > 0) {
    auto [id, which] = to_process.back();
    to_process.pop_back();

    auto const& plan = solved.at(id)[which];
    ret.insert({id, plan.partition});

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

  return start_cost;
}

vector<vector<int>> compute_equal_sums(int n, int d);

struct tuple_pair_int_hash_t {
  std::size_t operator()(tuple<int,int> const& lr) const noexcept
  {
    std::hash<int> h;
    auto const& [l,r] = lr;
    std::size_t ret = h(l);
    hash_combine_impl(ret, h(r));
    return ret;
  }
};

vector<vector<int>> compute_equal_sums_(int n, int d) {
  auto vector_sum = [](vector<int> const& xs) {
    int ret = 0;
    for(auto const& x: xs) {
      ret += x;
    }
    return ret;
  };

  if(d == 1) {
    return {vector<int>{n}};
  }

  vector<vector<int>> ret;
  ret.reserve(n);
  for(int i = 0; i != n+1; ++i) {
    for(auto d: compute_equal_sums(n-i, d-1)) {
      d.push_back(i);
      ret.push_back(d);
    }
  }

  return ret;
}

vector<vector<int>> compute_equal_sums(int n, int d) {
  static std::unordered_map<
    tuple<int,int>,
    vector<vector<int>>,
    tuple_pair_int_hash_t
  > cache;

  tuple<int,int> key(n,d);

  auto iter = cache.find(key);
  if(iter != cache.end()) {
    return iter->second;
  }

  auto ret = compute_equal_sums_(n,d);

  cache.insert({key, ret});
  return ret;
}

struct get_parts_t {
  graph_t const& graph;
  int log2_n;

  vector<partition_t> operator()(int gid) const {
    auto const& op = graph.nodes[gid].op;

    if(op.is_einsummable() && op.get_einsummable().is_contraction())
    {
      vector<uint64_t> shape = op.shape();
      vector<vector<int>> pps = compute_equal_sums(log2_n, shape.size());
      return fix(shape, pps);
    } else {
      vector<uint64_t> shape = op.shape();
      vector<vector<int>> pps = compute_equal_sums(log2_n, shape.size() + 1);
      return fix(shape, pps);
    }
  }

  static optional<partition_t>
  get_part(
    vector<uint64_t> const& shape,
    vector<int> const& ps)
  {
    vector<partdim_t> pd;
    pd.reserve(ps.size());
    for(int rank = 0; rank != shape.size(); ++rank) {
      uint64_t const& d = shape[rank];
      int s = exp2(ps[rank]);
      if(s > d) {
        return std::nullopt;
      }
      pd.push_back(partdim_t::split(d, s));
    }
    return partition_t(pd);
  }

  static vector<partition_t>
  fix(
    vector<uint64_t> const& shape,
    vector<vector<int>> const& pps)
  {
    vector<partition_t> ret;
    ret.reserve(pps.size());
    for(auto const& ps: pps) {
      auto maybe_part = get_part(shape, ps);
      if(maybe_part) {
        ret.push_back(maybe_part.value());
      }
    }

    if(ret.size() == 0) {
      throw std::runtime_error("should not be empty number of partitions");
    }
    return ret;
  }
};

struct compute_cost_t {
  graph_t const& graph;
  double time_per_elem;
  std::unordered_map<string, uint64_t> cache;

  double operator()(
    int gid,
    partition_t const& join_part,
    optional<partition_t> const& maybe_usage_part)
  {
    auto const& node = graph.nodes[gid];

    uint64_t compute_cost = 0;
    if(node.op.is_einsummable()) {
      einsummable_t const& e = node.op.get_einsummable();
      if(e.is_contraction()) {
        vector<int> op_block_shape = join_part.block_shape();
        int op_n_blocks = product(op_block_shape);

        // for each input, compute the cost to broadcast it
        // across the remaining portion of the grid
        {
          for(int i = 0; i != e.inns.size(); ++i) {
            vector<int> inn_block_shape = e.get_input_from_join(op_block_shape, i);
            int inn_n_blocks = product(inn_block_shape);
            int multiplier = op_n_blocks / inn_n_blocks;

            vector<uint64_t> inn_shape = e.get_input_from_join(e.join_shape, i);
            compute_cost += product(inn_shape) * multiplier;
          }
        }

        // Way 0:
        {
          vector<int> out_block_shape(
            op_block_shape.begin(),
            op_block_shape.begin() + e.out_rank);
          int out_n_blocks = product(out_block_shape);
          compute_cost += product(e.out_shape()) * (op_n_blocks - out_n_blocks);
        }

        // Way 1:
        //compute_cost += product(e.out_shape()) * op_n_blocks;

        // Way 2:
        //{
        //  // add the amount to broadcast the output tensor across the
        //  // aggregation blocks. No agg == no work.
        //  vector<int> agg_block_shape(
        //    op_block_shape.begin() + e.out_rank,
        //    op_block_shape.end());

        //  int multiplier = product(agg_block_shape) - 1;
        //  compute_cost += product(e.out_shape()) * multiplier;
        //}
      } else {
        // ignoring einsummable that is not contraction
      }
    } else {
      // ignoring the cost for non-einsummable nodes
    }

    uint64_t repart_cost = 0;
    if(maybe_usage_part) {
      auto const& usage_part = maybe_usage_part.value();
      int out_rank = usage_part.partdims.size();
      bool is_complex_join_part = dtype_is_complex(node.op.out_dtype());

      partition_t out_part = [&] {
        vector<partdim_t> pds(
          join_part.partdims.begin(),
          join_part.partdims.begin() + out_rank);
        if(is_complex_join_part) {
          partdim_t& pd = pds.back();
          pd = partdim_t::from_sizes(vector_double(pd.sizes()));
        }
        return partition_t(pds);
      }();

      repart_cost = compute_repart_cost(out_part, usage_part);
    }

    return time_per_elem * (compute_cost + repart_cost);
  }

  uint64_t compute_repart_cost(
    partition_t const& out_part,
    partition_t const& refi_part)
  {
    if(out_part == refi_part) {
      return 0;
    }

    string lhs = write_with_ss(out_part);
    string rhs = write_with_ss(refi_part);

    string key1 = lhs + rhs;
    string key2 = rhs + lhs;

    auto iter = cache.find(key1);
    if(iter != cache.end()) {
      return iter->second;
    }
    iter = cache.find(key2);
    if(iter != cache.end()) {
      return iter->second;
    }

    // TODO: This code is a hotspot; speed it up.
    // {{{
    uint64_t ret = 0;
    {
      vector<uint64_t> block_sizes_aa = out_part.all_block_sizes().get();
      vector<uint64_t> block_sizes_bb = refi_part.all_block_sizes().get();

      copyregion_full_t copyregion(out_part, refi_part);

      do {
        ret += block_sizes_aa[copyregion.idx_aa];
        ret += block_sizes_aa[copyregion.idx_bb];
      } while(copyregion.increment());
    }
    // }}}

    cache.insert({key1, ret});
    return ret;
  }
};

double _solve_tree(
  map<int, partition_t>& ret,
  graph_t const& graph,
  ap_tree_t const& tree,
  int max_branching,
  get_parts_t& get_possible_partitions,
  compute_cost_t& compute_cost)
{
  // Assumption: the tree is not empty

  map<int, vector<_plan_t>> solved;

  auto all_gids = tree.dfs_from_root();
  std::reverse(all_gids.begin(), all_gids.end());
  for(int const& gid: all_gids) {
    vector<int> tree_inns = tree.get_inns_as_vec(gid);

    // Note: cost_from_fixed is a bit goofy in that it won't be
    //       changing so isn't needed in the solve
    double cost_from_fixed = 0.0;

    // solve any trees past max_branching; the partitions
    // therein will be grabbed from ret
    if(tree_inns.size() > max_branching) {
      for(int i = max_branching; i != tree_inns.size(); ++i) {
        int const& inn_gid = tree_inns[i];
        cost_from_fixed += _add_to_ret(ret, solved, tree, inn_gid);
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

    vector<_plan_t>& plans = solved[gid];

    for(auto const& part: get_possible_partitions(gid)) {
      vector<int> best_which;
      double best_cost = -1.0;

      vector<int> which(num_tree_inns, 0);
      do {
        auto get_partition = [&](int gid) -> partition_t const& {
          for(int i = 0; i != which.size(); ++i) {
            int const& w = which[i];
            int const& g = tree_inns[i];
            if(g == gid) {
              return solved.at(g)[w].partition;
            }
          }
          auto iter = ret.find(gid);
          if(iter == ret.end()) {
            throw std::runtime_error("get_partition lambda: did not find");
          }
          return iter->second;
        };

        double cost_from_children = 0.0;
        for(int i = 0; i != num_tree_inns; ++i) {
          int const& w = which[i];
          int const& g = tree_inns[i];
          cost_from_children += solved.at(g)[w].cost;
        }

        optional<partition_t> refi_partition;
        {
          int gid_fix = gid;

          auto const& node = graph.nodes[gid];
          int out_gid = *node.outs.begin();
          if(graph.nodes[out_gid].op.is_formation()) {
            gid_fix = out_gid;
          }

          if(graph.nodes[gid_fix].outs.size() > 0) {
            refi_partition = twolayer_construct_refinement_partition(
              graph, gid_fix, get_partition);
          }
        }

        double cost_from_node = compute_cost(gid, part, refi_partition);

        double cost = cost_from_fixed + cost_from_children + cost_from_node;

        if(best_cost < 0.0 || cost < best_cost) {
          best_cost = cost;
          best_which = which;
        }
      } while(has_tree_inns && increment_idxs(inn_num_plans, which));

      plans.push_back(_plan_t {
        .partition = part,
        .which = best_which,
        .cost = best_cost
      });
    }
  }

  return _add_to_ret(ret, solved, tree, tree.get_root_id());
}

set<int> _get_graph_outs_past_formation(graph_t const& graph, int gid)
{
  auto const& node = graph.nodes[gid];
  if(node.outs.size() == 0) {
    return node.outs;
  }
  if(node.outs.size() == 1) {
    auto const& out_gid = *node.outs.begin();
    auto const& out_node = graph.nodes[out_gid];
    if(out_node.op.is_formation()) {
      return _get_graph_outs_past_formation(graph, out_gid);
    }
  }
  return node.outs;
}

vector<partition_t> _build_vector(
  graph_t const& graph,
  map<int, partition_t> const& parts)
{
  vector<partition_t> ret;
  ret.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_formation()) {
      auto const& inn_gid = node.inns[0];

      auto const& inn_part = parts.at(inn_gid);

      int rank = node.op.out_rank();

      vector<partdim_t> pds(
        inn_part.partdims.begin(),
        inn_part.partdims.begin() + rank);

      ret.emplace_back(pds);
    } else {
      ret.push_back(parts.at(gid));
    }
  }
  return ret;
}

vector<partition_t> autopartition_for_bytes(
  graph_t const& graph,
  int n_compute)
{
  //{
  //  std::ofstream f("g.gv");
  //  graph.print_graphviz(f);
  //  DOUT("printed g.gv");
  //}

  int max_branching = 2;
  int log2_n = log2(n_compute);
  // TODO: ^ use this for the costing

  // TODO: ^ fill this out

  // Note: every formation node has the same partition as the input
  //       and isn't solved as part of the graph
  // Assumption: there are no formation->formation edges
  // Assumption: if a node has an out formation node, that is it's
  //             only out node.

  // 1. build a ap_graph_t with gids we care about
  //    finding a partition for (so skip over formation nodes)
  // 2. build a graph of trees
  // 3. for each tree in graph order,
  //      solve for the best partition for that tree
  // 4. build the vector of partitions

  // 1.
  ap_graph_t bgraph;
  vector<int> graph_order = graph.get_order();
  std::reverse(graph_order.begin(), graph_order.end());
  for(int const& gid: graph_order) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_formation()) {
      continue;
    }

    // Going from graph outs to bgraph inns since we'll be
    // costing edges from the graph-outs
    set<int> inns = _get_graph_outs_past_formation(graph, gid);

    bgraph.insert(gid, unit_t{}, inns);
  }
  //{
  //  std::ofstream f("bg.gv");
  //  bgraph.print_graphviz(f);
  //  DOUT("printed bg.gv");
  //}

  // 2.
  ap_graphtree_t btrees = build_trees_ns::build_trees(bgraph);
  //{
  //  std::ofstream f("bt.gv");
  //  btrees.print_graphviz(f);
  //  DOUT("printed bt.gv");
  //}

  //{
  //  vector<string> color_list{
  //    "#61B292",
  //    "#AED09E",
  //    "#F1E8A7",
  //    "#A8896C",
  //    "#A8D8EA",
  //    "#AA96DA",
  //    "#FCBAD3",
  //    "#FFFFD2",
  //    "#F0FFFF",
  //    "#DEB887",
  //    "yellow",
  //    "red",
  //    "orange"
  //  };
  //  int i = 0;

  //  map<int, string> colors;

  //  for(auto const& [_, node]: btrees.get()) {
  //    string color = color_list[i];
  //    i = (i + 1) % color_list.size();

  //    auto const& tree = node.item;
  //    for(int const& id: tree.dfs_from_root()) {
  //      colors.insert({id, color});
  //    }
  //  }

  //  std::ofstream f("bg_color.gv");
  //  bgraph.print_graphviz(f, colors);
  //  DOUT("printed bg_color.gv");
  //}

  // 3.
  get_parts_t get_possible_partitions {
    .graph = graph,
    .log2_n = log2_n
  };
  compute_cost_t compute_cost {
    .graph = graph,
    .time_per_elem = 1e-9,
    .cache = {}
  };
  map<int, partition_t> partitions_so_far;
  for(int const& root_id: btrees.dag_order_inns_to_outs()) {
    auto const& tree = btrees[root_id];
    double cost_tree = _solve_tree(
      partitions_so_far, graph, tree, max_branching,
      get_possible_partitions,
      compute_cost);
  }

  // 4.
  return _build_vector(graph, partitions_so_far);
}


