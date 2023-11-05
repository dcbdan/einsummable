#include "loadbalanceplace.h"
#include "locsetter.h"
#include "twolayer.h"

#include "../einsummable/taskgraph.h"
#include "../base/copyregion.h"

struct _lbp_count_t {
  uint64_t size;
  int loc;
  int block_id;
};

inline bool operator<(
  _lbp_count_t const& lhs,
  _lbp_count_t const& rhs)
{
  return lhs.size < rhs.size;
}

vector<_lbp_count_t> compute_loc_scores(
  vector<int> const& avail_locs,
  // bid,loc->score
  std::function<uint64_t(int,int)> get_score,
  vector<int> const& bids)
{
  vector<_lbp_count_t> ret;
  ret.reserve(bids.size());

  for(auto const& bid: bids) {
    vector<uint64_t> scores;
    scores.reserve(avail_locs.size());
    for(auto const& loc: avail_locs) {
      scores.push_back(get_score(bid,loc));
    }

    int which = std::min_element(scores.begin(), scores.end()) - scores.begin();
    ret.push_back(_lbp_count_t {
      .size = scores[which],
      .loc = avail_locs[which],
      .block_id = bid
    });
  }

  return ret;
}

vector<placement_t> load_balanced_placement(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool random_input)
{
  // the forward_state_t object is just acting as a twolayer graph
  // here, so give it a dummy cluster that won't actually end up
  // being used
  cluster_t dummy_cluster = cluster_t::make(
    { cluster_t::device_t(1000000) },
    {});
  forward_state_t twolayer(dummy_cluster, graph);

  // Assign all the partitions in twolayer and
  // initialize pls
  vector<placement_t> pls;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& part = parts[gid];

    twolayer.assign_partition(gid, part);

    vtensor_t<int> locations(part.block_shape(), -1);
    pls.emplace_back(part, locations);
  }

  auto get_loc = [&pls, &nlocs](jid_t const& jid) {
    auto const& [gid,bid] = jid;
    int ret = pls[gid].locations.get()[bid];
    if(ret < 0 || ret > nlocs) {
      throw std::runtime_error("get_loc: invalid return value");
    }
    return ret;
  };

  for(int gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];

    auto& pl = pls[gid];
    vtensor_t<int>& locations = pl.locations;
    vector<int>& locs = locations.get();

    auto const& part = parts[gid];

    if(node.op.is_input()) {
      if(random_input) {
        loc_setter_t setter(locs.size(), nlocs);
        for(int bid = 0; bid != locs.size(); ++bid) {
          int loc = setter.runif();
          locs[bid] = loc;
          setter.decrement(loc);
        }
      } else {
        // Just round robin assign input nodes
        int l = 0;
        for(int bid = 0; bid != locs.size(); ++bid) {
          locs[bid] = l;
          l = (l + 1) % nlocs;
        }
      }

      if(!is_balanced(locs)) {
        throw std::runtime_error("random assignment is not balanced");
      }

      continue;
    }

    if(node.op.is_formation()) {
      int const& id_inn = node.inns[0];
      if(part == parts[id_inn]) {
        // This is a formation node with an equivalently placed input,
        // so there is no other reasonable location than this one
        locations = pls[id_inn].locations;
        continue;
      }
    }

    if(node.op.is_einsummable() &&
       node.inns.size() == 1 &&
       !node.op.has_aggregation())
    {
      auto const& e = node.op.get_einsummable();

      int const& id_inn = node.inns[0];
      auto const& part_inn = parts[id_inn];

      auto partdims_with_respect_to_inn =
        e.get_input_from_join(part.partdims, 0);

      if(part_inn.partdims == partdims_with_respect_to_inn) {
        // now we have to reach past the permutation
        auto join_shape = part.block_shape();
        vector<int> join_index(join_shape.size(), 0);
        vtensor_t<int> const& inn_locations = pls[id_inn].locations;
        do {
          vector<int> inn_index = e.get_input_from_join(join_index, 0);
          locations.at(join_index) = inn_locations.at(inn_index);
        } while(increment_idxs(join_shape, join_index));

        continue;
      }
    }

    // TODO: this is not taking into account the dtype
    auto get_score = [&](int bid, int loc) {
      return twolayer.count_elements_to(
        get_loc,
        jid_t { gid, bid },
        loc);
    };

    vector<int> remaining(locs.size());
    std::iota(remaining.begin(), remaining.end(), 0);

    loc_setter_t setter(locs.size(), nlocs);

    while(remaining.size() > 0) {
      vector<_lbp_count_t> loc_scores = compute_loc_scores(
        setter.get_avail_locs(),
        get_score,
        remaining);
      std::sort(loc_scores.begin(), loc_scores.end());

      vector<int> new_remaining;
      bool ran_out = false;
      for(auto const& [_, loc, bid]: loc_scores) {
        if(ran_out) {
          new_remaining.push_back(bid);
        } else {
          locs[bid] = loc;
          bool loc_still_avail = setter.decrement(loc);
          ran_out = !loc_still_avail;
        }
      }

      remaining = new_remaining;
    }

    if(!is_balanced(locs)) {
      throw std::runtime_error("is not balanced!");
    }
  }

  // do some more checks
  for(auto const& pl: pls) {
    vector<int> const& locs = pl.locations.get();
    int mn = *std::min_element(locs.begin(), locs.end());
    int mx = *std::max_element(locs.begin(), locs.end());
    if(mn < 0 || mn >= nlocs || mx < 0 || mx >= nlocs) {
      throw std::runtime_error("invalid locs");
    }
    if(locs.size() != product(pl.partition.block_shape())) {
      throw std::runtime_error("invalid num locs");
    }
  }

  return pls;
}

vector<int> graph_reverse_order(graph_t const& graph) {
  vector<int> ret;
  // reserve to not invalidate iterators
  ret.reserve(graph.nodes.size());

  vector<int> deps;
  deps.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    int ndep = node.outs.size();
    if(ndep == 0) {
      ret.push_back(gid);
    }
    deps[gid] = ndep;
  }

  for(auto iter = ret.begin(); iter != ret.end(); ++iter) {
    int gid = *iter;
    auto const& node = graph.nodes[gid];
    set<int> inns(node.inns.begin(), node.inns.end());
    for(auto const& out_gid: inns) {
      int& cnt = deps[out_gid];
      cnt--;
      if(cnt == 0) {
        ret.push_back(out_gid);
      }
    }
  }

  return ret;
}

uint64_t _compute_move_cost_with_dst_sizes(
  set<int> const& srcs,
  vector<uint64_t> const& dst_sizes)
{
  uint64_t ret = 0;
  for(int const& src: srcs) {
    for(int dst = 0; dst != dst_sizes.size(); ++dst) {
      if(src != dst) {
        ret += dst_sizes[dst];
      }
    }
  }
  return ret;
}

struct _count_block_parts_t {
  _count_block_parts_t(
    partition_t const& join_part, // wrt real
    multiple_placement_t const& usage)
    : out_part(vector<partdim_t>{ partdim_t::singleton(1) })
  {
    out_rank = usage.partition.partdims.size();
    out_part = partition_t(vector<partdim_t>(
      join_part.partdims.begin(),
      join_part.partdims.begin() + out_rank));

    join_part_num_parts = join_part.num_parts();
    out_part_num_parts = out_part.num_parts();
    num_agg = join_part_num_parts / out_part_num_parts;
  }

  int out_rank;
  partition_t out_part;
  int join_part_num_parts;
  int out_part_num_parts;
  int num_agg;
};

vector<vector<uint64_t>> _build_dst_sizes(
  partition_t const& join_part, // wrt real
  multiple_placement_t const& usage,
  _count_block_parts_t const& c)
{
  vector<vector<uint64_t>> dst_sizes(c.out_part_num_parts);
  {
    copyregion_full_t copyregion(c.out_part, usage.partition);
    vector<set<int>> const& usage_locs = usage.locations.get();
    do {
      int const& out_idx = copyregion.idx_aa;
      int const& usage_idx = copyregion.idx_bb;

      uint64_t block_size = product(copyregion.size);
      set<int> const& dsts = usage_locs[usage_idx];
      vector<uint64_t>& dsizes = dst_sizes[out_idx];
      for(int const& dst: dsts) {
        if(dsizes.size() <= dst) {
          dsizes.resize(dst+1);
        }
        dsizes[dst] += block_size;
      }
    } while(copyregion.increment());
  }
  return dst_sizes;
}

vector<vector<uint64_t>> _build_dst_sizes(
  partition_t const& join_part, // wrt real
  multiple_placement_t const& usage)
{
  return _build_dst_sizes(join_part, usage, _count_block_parts_t(join_part, usage));
}

vtensor_t<int> greedy_solve_placement(
  partition_t const& part,
  multiple_placement_t const& usage,
  int nlocs)
{
  auto compute_cost = [](
    int src,
    set<int> srcs,
    vector<uint64_t> const& dst_sizes)
  {
    srcs.insert(src);
    return _compute_move_cost_with_dst_sizes(srcs, dst_sizes);
  };

  auto pick_loc = [&compute_cost](
    set<int> const& srcs,
    vector<int> const& avail_locs,
    vector<uint64_t> const& dst_sizes)
  {
    for(int const& loc: avail_locs) {
      if(srcs.count(loc) > 0) {
        return loc;
      }
    }
    int ret = avail_locs[0];
    uint64_t best_cost = compute_cost(avail_locs[0], srcs, dst_sizes);
    for(int idx = 1; idx != avail_locs.size(); ++idx) {
      uint64_t cost = compute_cost(avail_locs[idx], srcs, dst_sizes);
      if(cost < best_cost) {
        ret = avail_locs[idx];
        best_cost = cost;
      }
    }
    return ret;
  };

  _count_block_parts_t c(part, usage);
  vector<vector<uint64_t>> dst_sizes = _build_dst_sizes(part, usage, c);

  vtensor_t<int> ret(part.block_shape());
  vector<int>& ret_locs = ret.get();
  loc_setter_t setter(c.join_part_num_parts, nlocs);
  for(int out_bid = 0; out_bid != c.out_part_num_parts; ++out_bid) {
    set<int> srcs;
    int bid = out_bid * c.num_agg;
    int stop_bid = bid + c.num_agg;
    for(; bid != stop_bid; ++bid) {
      int& loc = ret_locs[bid];
      loc = pick_loc(srcs, setter.get_avail_locs(), dst_sizes[out_bid]);
      srcs.insert(loc);
      setter.decrement(loc);
    }
  }

  return ret;
}

partition_t _get_part_wrt_real(
  graph_t const& graph,
  partition_t part,
  int gid)
{
  auto const& node = graph.nodes[gid];
  if(dtype_is_complex(node.op.out_dtype())) {
    int out_rank = node.op.out_rank();
    auto& pd = part.partdims[out_rank-1];
    pd = partdim_t::from_sizes(vector_double(pd.sizes()));
  }

  return part;
}

vector<placement_t> load_balanced_placement_from_outs(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool random_output)
{
  if(parts.size() != graph.nodes.size()) {
    throw std::runtime_error("invalid lbp");
  }

  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    ret.emplace_back(part);
  }

  auto get_placement = [&ret](int other_gid)
    -> placement_t const&
  {
    return ret[other_gid];
  };
  auto get_part_wrt_real = [&](int gid) {
    partition_t const& part = parts[gid];
    return _get_part_wrt_real(graph, part, gid);
  };

  for(int const& gid: graph_reverse_order(graph)) {
    auto const& node = graph.nodes[gid];
    if(node.outs.size() == 0) {
      vector<int>& locs = ret[gid].locations.get();
      if(random_output) {
        loc_setter_t setter(locs.size(), nlocs);
        for(int& loc: locs) {
          loc = setter.runif();
          setter.decrement(loc);
        }
      } else {
        int l = 0;
        for(int& loc: locs) {
          loc = l;
          l = (l + 1) % nlocs;
        }
      }
    } else {
      multiple_placement_t usage_pl = construct_refinement_placement(
        graph, gid, get_placement);

      ret[gid].locations = greedy_solve_placement(
        get_part_wrt_real(gid), usage_pl, nlocs);
    }
  }

  return ret;
}

vector<uint64_t> compute_tensor_move_costs(
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto get_placement = [&placements](int other_gid)
    -> placement_t const&
  {
    return placements[other_gid];
  };

  vector<uint64_t> ret(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.outs.size() == 0) {
      continue;
    }

    multiple_placement_t usage = construct_refinement_placement(
      graph, gid, get_placement);

    auto const& join_placement = placements[gid];

    vector<int> const& locs = join_placement.locations.get();

    auto part = _get_part_wrt_real(graph, join_placement.partition, gid);

    _count_block_parts_t c(part, usage);
    vector<vector<uint64_t>> dst_sizes = _build_dst_sizes(part, usage, c);

    uint64_t& total = ret[gid];
    for(int out_bid = 0; out_bid != c.out_part_num_parts; ++out_bid) {
      vector<uint64_t> const& dsizes = dst_sizes[out_bid];
      set<int> srcs(
        locs.begin()  + ( out_bid      * c.num_agg),
        locs.begin()  + ((out_bid + 1) * c.num_agg));
      total += _compute_move_cost_with_dst_sizes(srcs, dsizes);
    }

    // from units of (real) elems to bytes
    auto dtype = node.op.out_dtype();
    if(dtype_is_complex(dtype)) {
      total *= (dtype_size(dtype) / 2);
    } else {
      total *= dtype_size(dtype);
    }
  }

  return ret;
}

