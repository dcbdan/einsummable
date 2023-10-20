#include "apart.h"

#include "twolayer.h" // for twolayer_construct_refinement_partition
#include "../base/copyregion.h" // for union_pair_partitions
#include "../einsummable/taskgraph.h" // for double_last_dim

struct apart_state_t {
  apart_state_t(graph_t const& g, int l);

  struct plan_t {
    plan_t(int gid, partition_t const&  p, double cost, int next);
    int gid;

    int ds[10];
    static int _ds_size() { return 10; }

    double cost;

    int next;

    partition_t get_partition(vector<uint64_t> const& shape) const;
  };

  void solve(int gid);

  plan_t solve_(int gid, partition_t const& part);

  vector<partition_t> get_possible_partitions_(int gid) const;

  vector<partition_t> get_partition() const;

  // returned with respect to the real dtype
  partition_t get_refi_part(int gid, int which_plan) const;

  // Note: refi_part is always with respect to the real dtype
  // The cost is equal to
  //   [a*bytes-touched + b for bytes in each output-touch]
  virtual double compute_cost(
    int gid,
    partition_t const& join_part,
    partition_t const& refi_part) const = 0;

  graph_t const& graph;
  int log2_n;

  double time_per_elem;
  double time_per_write;

  vector<vector<plan_t>> mapping;
};

bool operator<(
  apart_state_t::plan_t const& lhs,
  apart_state_t::plan_t const& rhs)
{
  return lhs.cost < rhs.cost;
}

struct apart_repart_t : apart_state_t {
  apart_repart_t(
    graph_t const& g,
    int l,
    double t_per_write,
    double t_per_elem)
  : apart_state_t(g, l),
    time_per_write(t_per_write), time_per_elem(t_per_elem)
  {}

  double compute_cost(
    int gid,
    partition_t const& join_part,
    partition_t const& refi_part) const;

  double time_per_write;
  double time_per_elem;
};

vector<partition_t> autopartition_for_repart(
  graph_t const& graph,
  int n_compute,
  double time_per_write,
  double time_per_elem)
{
  int log2_n = log2(n_compute);
  apart_repart_t state(graph, log2_n, time_per_write, time_per_elem);

  for(int gid = graph.nodes.size() - 1; gid >= 0; --gid) {
    state.solve(gid);
  }

  return state.get_partition();
}

struct apart_bytes_t : apart_state_t {
  apart_bytes_t(
    graph_t const& g,
    int l)
  : apart_state_t(g, l), time_per_elem(1e-9), compute_mult(1)
  {}

  double compute_cost(
    int gid,
    partition_t const& join_part,
    partition_t const& refi_part) const;

  // these should both be with respect to the real dtype
  uint64_t compute_repart_cost(
    partition_t const& out_part,
    partition_t const& refi_part) const;

  double time_per_elem;
  uint64_t compute_mult;
};

vector<partition_t> autopartition_for_bytes(
  graph_t const& graph,
  int n_compute)
{
  int log2_n = log2(n_compute);
  apart_bytes_t state(graph, log2_n);

  for(int gid = graph.nodes.size() - 1; gid >= 0; --gid) {
    state.solve(gid);
  }

  return state.get_partition();
}

apart_state_t::apart_state_t(graph_t const& g, int l)
  : graph(g), log2_n(l), mapping(graph.nodes.size())
{}

void apart_state_t::solve(int gid)
{
  // if this node has already been solved, return
  if(mapping[gid].size() > 0) {
    return;
  }

  for(partition_t part: get_possible_partitions_(gid)) {
    mapping[gid].push_back(solve_(gid, part));
  }
}

apart_state_t::plan_t
apart_state_t::solve_(int gid, partition_t const& join_part)
{
  // Assumption: any out node is not an agg node

  if(gid + 1 == graph.nodes.size()) {
    // Note: the last node must b e an out node
    return plan_t(gid, join_part, 0.0, -1);
  }

  auto const& node = graph.nodes[gid];

  int best_plan = -1;
  double best_cost;

  auto const& next_mapping = mapping[gid+1];
  for(int which_plan = 0; which_plan != next_mapping.size(); ++which_plan)
  {
    auto const& next_plan = next_mapping[which_plan];

    // TODO: maybe we want to cache get_refi_part or store the
    //       refi part of the previous node in the next plan
    double just_this_cost = 0.0;
    if(node.outs.size() > 0) {
      auto refi_part = get_refi_part(gid, which_plan);

      just_this_cost = compute_cost(gid, join_part, refi_part);
    }

    double cost = just_this_cost + next_plan.cost;
    if(best_plan == -1 || cost < best_cost) {
      best_plan = which_plan;
      best_cost = cost;
    }
  }

  if(best_plan < 0) {
    throw std::runtime_error("the next node has not been solved!");
  }

  return plan_t(gid, join_part, best_cost, best_plan);
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

vector<partition_t> apart_state_t::get_possible_partitions_(int gid) const
{
  auto get_part = [](
    vector<uint64_t> const& shape,
    vector<int> const& ps)
  -> optional<partition_t>
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
  };

  auto fix = [&get_part](
    vector<uint64_t> const& shape,
    vector<vector<int>> const& pps)
  -> vector<partition_t>
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
  };

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

vector<partition_t> apart_state_t::get_partition() const
{
  vector<partition_t> ret;
  ret.reserve(graph.nodes.size());

  if(mapping[0].size() == 0) {
    throw std::runtime_error("root gid not solved yet");
  }

  int which = std::distance(
    mapping[0].begin(),
    std::min_element(mapping[0].begin(), mapping[0].end()));
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& op = graph.nodes[gid].op;
    auto const& plan = mapping[gid][which];
    if(gid == 0) {
      DOUT("cost is " << plan.cost);
    }
    ret.push_back(plan.get_partition(op.shape()));
    which = plan.next;
  };

  return ret;
}

apart_state_t::plan_t::plan_t(int gid, partition_t const& part, double cost, int next)
  : gid(gid), cost(cost), next(next)
{
  vector<partdim_t> const& pds = part.partdims;
  if(pds.size() > _ds_size()) {
    throw std::runtime_error("ds not big enough!");
  }
  for(int r = 0; r != pds.size(); ++r) {
    auto const& pd = pds[r];
    ds[r] = pd.num_parts();
  }
}

partition_t
apart_state_t::plan_t::get_partition(
  vector<uint64_t> const& shape) const
{
  if(shape.size() > _ds_size()) {
    throw std::runtime_error("ds not big enough");
  }

  vector<partdim_t> ret;
  ret.reserve(shape.size());
  for(int r = 0; r != shape.size(); ++r) {
    uint64_t const& dim = shape[r];
    int const& split = ds[r];
    ret.push_back(partdim_t::split(dim, split));
  }

  return partition_t(ret);
}

partition_t
apart_state_t::get_refi_part(
  int gid,
  int which_plan) const
{
  // get the partition for each output node, by traversing
  // the linked list

  auto const& outs = graph.nodes[gid].outs;
  map<int, partition_t> out_parts;
  for(int up_gid = gid+1; up_gid != graph.nodes.size(); ++up_gid) {
    auto const& plan = mapping[up_gid][which_plan];
    which_plan = plan.next;

    // if this is an out node
    if(outs.count(up_gid) > 0) {
      auto shape = graph.nodes[up_gid].op.shape();
      out_parts.insert({up_gid, plan.get_partition(shape)});
      if(out_parts.size() == outs.size()) {
        break;
      }
    }
  }

  if(out_parts.size() != outs.size()) {
    throw std::runtime_error("could not get all outs partition");
  }

  return twolayer_construct_refinement_partition(
    graph,
    gid,
    [&out_parts](int i) -> partition_t const& { return out_parts.at(i); }
  );
}

double
apart_repart_t::compute_cost(
  int gid,
  partition_t const& join_part_,
  partition_t const& refi_part) const
{
  int out_rank   = refi_part.partdims.size();
  int join_rank  = join_part_.partdims.size();
  int agg_rank   = join_rank - out_rank;

  bool is_complex_join_part =
    join_part_.partdims[out_rank-1].total() !=
    refi_part.partdims[out_rank-1].total() ;

  partition_t join_part = [&] {
    if(is_complex_join_part) {
      auto pds = join_part_.partdims;
      partdim_t& pd = pds[out_rank-1];
      pd = partdim_t::from_sizes(vector_double(pd.sizes()));
      return partition_t(pds);
    } else {
      return join_part_;
    }
  }();

  auto join_shape = join_part.block_shape();
  auto const& _join_partdims = join_part.partdims;

  int num_agg_blocks = 1;
  if(agg_rank > 0) {
    partition_t agg_part(vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end()));
    num_agg_blocks = agg_part.num_parts();
  } else if(join_part == refi_part) {
    // when the join and refi are equal, there is no repartitioning work
    // to do
    return 0.0;
  }

  partition_t out_part(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  partition_t touch_part = union_pair_partitions(refi_part, out_part);
  vector<uint64_t> touch_block_sizes = touch_part.all_block_sizes().get();

  double ret = touch_block_sizes.size() * time_per_write;
  for(uint64_t const& touch_size: touch_block_sizes) {
    ret += time_per_elem * touch_size;
  }

  ret *= num_agg_blocks;

  //DOUT(refi_part);
  //DOUT(out_part);
  //DOUT(num_
  return ret;
}

double
apart_bytes_t::compute_cost(
  int gid,
  partition_t const& join_part,
  partition_t const& usage_part) const
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

      // add the amount to broadcast the output tensor across the
      // aggregation blocks. No agg == no work.
      if(e.has_aggregation()) {
        vector<int> agg_block_shape(
          op_block_shape.begin() + e.out_rank,
          op_block_shape.end());

        int multiplier = product(agg_block_shape);
        compute_cost += product(e.out_shape()) * multiplier;
      }
    } else {
      // ignoring einsummable that is not contraction
    }
  } else {
    // ignoring the cost for non-einsummable nodes
  }

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

  uint64_t repart_cost = compute_repart_cost(out_part, usage_part);

  return time_per_elem * (compute_mult*compute_cost + repart_cost);
}

uint64_t
apart_bytes_t::compute_repart_cost(
  partition_t const& out_part,
  partition_t const& refi_part) const
{
  if(out_part == refi_part) {
    return 0;
  }

  vector<uint64_t> block_sizes_aa = out_part.all_block_sizes().get();
  vector<uint64_t> block_sizes_bb = refi_part.all_block_sizes().get();

  copyregion_full_t copyregion(out_part, refi_part);

  uint64_t ret = 0;
  do {
    ret += block_sizes_aa[copyregion.idx_aa];
    ret += block_sizes_aa[copyregion.idx_bb];
  } while(copyregion.increment());

  return ret;
}
