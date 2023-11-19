#include "twolayer.h"
#include "../base/hrect.h"
#include "../base/copyregion.h"
#include "../einsummable/taskgraph.h"

vector<join_t> twolayer_construct_joins(
  graph_t const& graph,
  int gid,
  partition_t const& join_partition)
{
  auto join_block_shape = join_partition.block_shape();

  vector<join_t> join_infos(product(join_block_shape));

  auto const& node = graph.nodes[gid];
  if(node.op.is_einsummable()) {
    einsummable_t const& base_einsummable = node.op.get_einsummable();
    vector<int> join_index(join_block_shape.size(), 0);
    do {
      int join_bid = idxs_to_index(join_block_shape, join_index);
      join_t& join_info = join_infos[idxs_to_index(join_block_shape, join_index)];
      join_info.einsummable = einsummable_t::with_new_shape(
        base_einsummable,
        join_partition.tensor_shape_at(join_index));
    } while(increment_idxs(join_block_shape, join_index));
  }

  return join_infos;
}

partition_t _double_dim(partition_t const& p, int idx) {
  partition_t ret = p;
  partdim_t& pd = ret.partdims[idx];
  pd = partdim_t::from_sizes(vector_double(pd.sizes()));
  return ret;
}

void twolayer_insert_join_deps(
  graph_t const& graph,
  int gid,
  vector<join_t>& join_infos,
  partition_t const& join_partition,
  std::function<partition_t const&(int)> get_refinement_partition)
{
  auto const& node = graph.nodes[gid];

  dtype_t join_dtype = node.op.out_dtype();
  bool join_is_complex = dtype_is_complex(join_dtype);

  if(node.op.is_input()) {
    // input joins don't have deps, so nothing to do
  } else if(node.op.is_concat()) {
    auto join_block_shape = join_partition.block_shape();
    vector<int> join_index(join_block_shape.size(), 0);
    do {
      int join_bid = idxs_to_index(join_block_shape, join_index);
      join_t& join_info = join_infos[idxs_to_index(join_block_shape, join_index)];

      using hrect_t = vector<tuple<uint64_t, uint64_t>>;
      hrect_t join_hrect = join_partition.get_hrect(join_index);

      auto const& concat = node.op.get_concat();
      int n_inns = concat.num_inns();
      for(int which_inn = 0; which_inn != n_inns; ++which_inn) {
        hrect_t inn_hrect = concat.get_hrect(which_inn);
        if(interval_intersect(join_hrect[concat.dim], inn_hrect[concat.dim])) {
          // get the copy_hrect with respect to the input relation
          hrect_t copy_hrect = hrect_center(
            inn_hrect,
            hrect_intersect(join_hrect, inn_hrect));

          // If this concat is complex, then the refinement_partitition
          // is still real. So the corresponding copy_rect is actually doubled
          if(join_is_complex) {
            auto& [b,e] = copy_hrect.back();
            b *= 2;
            e *= 2;
          }

          int const& inn = node.inns[which_inn];
          partition_t const& inn_partition = get_refinement_partition(inn);

          auto inn_region = inn_partition.get_region(copy_hrect);
          auto inn_shape = inn_partition.block_shape();
          auto inn_index = vector_mapfst(inn_region);
          do {
            int inn_refi_bid = idxs_to_index(inn_shape, inn_index);
            rid_t dep_rid { inn, inn_refi_bid };
            join_info.deps.insert(dep_rid);
          } while(increment_idxs_region(inn_region, inn_index));
        }
      }
    } while(increment_idxs(join_block_shape, join_index));
  } else if(node.op.is_subset()) {
    auto join_block_shape = join_partition.block_shape();
    vector<int> join_index(join_block_shape.size(), 0);
    do {
      int join_bid = idxs_to_index(join_block_shape, join_index);
      join_t& join_info = join_infos[idxs_to_index(join_block_shape, join_index)];
      auto const& subset_ = node.op.get_subset();
      vector<int> unsqueezed_join_index = subset_.unsqueeze_vec(join_index, 0);

      int const& inn = node.inns[0];
      partition_t const& inn_partition = get_refinement_partition(inn);

      auto [subset, out_partition] = unsqueeze_subset_partition(
        subset_, join_partition);

      using hrect_t = vector<tuple<uint64_t, uint64_t>>;

      hrect_t inn_hrect = subset.get_hrect();

      // get the copy_hrect (with respect to the inn_hrect)
      hrect_t copy_hrect = out_partition.get_hrect(unsqueezed_join_index);
      for(int i = 0; i != inn_hrect.size(); ++i) {
        auto&       [jb,je] = copy_hrect[i];
        auto const& [ib,_ ] = inn_hrect[i];
        jb += ib;
        je += ib;
      }

      // If this subset is complex, then the refinement_partitition
      // is still real. So the corresponding copy_rect is actually doubled
      if(join_is_complex) {
        auto& [b,e] = copy_hrect.back();
        b *= 2;
        e *= 2;
      }

      auto inn_region = inn_partition.get_region(copy_hrect);
      auto inn_shape = inn_partition.block_shape();
      auto inn_index = vector_mapfst(inn_region);
      do {
        int inn_refi_bid = idxs_to_index(inn_shape, inn_index);
        rid_t dep_rid { inn, inn_refi_bid };
        join_info.deps.insert(dep_rid);
      } while(increment_idxs_region(inn_region, inn_index));
    } while(increment_idxs(join_block_shape, join_index));
  } else if(node.op.is_formation()) {
    int const& inn = node.inns[0];
    partition_t const& inn_partition = get_refinement_partition(inn);

    vector<int> inn_idxs(inn_partition.partdims.size());
    std::iota(inn_idxs.begin(), inn_idxs.end(), 0);

    // If this formation is complex, double the last dim of the
    // join partition so that it is with respect to real, like
    // the inn_partition (which is a refinement and therefore wrt real)
    partition_t join_partition_real =
      join_is_complex                 ?
      double_last_dim(join_partition) :
      join_partition                  ;

    copyregion_join_inn_t cr(
      join_partition_real,
      inn_partition,
      inn_idxs);

    do {
      join_infos[cr.idx_join()].deps.insert(rid_t{ inn, cr.idx_inn() });
    } while(cr.increment());
  } else if(node.op.is_complexer()) {
    int const& inn = node.inns[0];
    partition_t const& inn_partition = get_refinement_partition(inn);

    vector<int> inn_idxs(inn_partition.partdims.size());
    std::iota(inn_idxs.begin(), inn_idxs.end(), 0);

    // if(join_is_complex) {
    //   then this is real -> complex
    // } else {
    //   then this is complex -> real
    // }
    // But the inn partition is already with respect to real
    partition_t join_partition_real =
      join_is_complex                 ?
      double_last_dim(join_partition) :
      join_partition                  ;

    copyregion_join_inn_t cr(
      join_partition_real,
      inn_partition,
      inn_idxs);

    do {
      join_infos[cr.idx_join()].deps.insert(rid_t{ inn, cr.idx_inn() });
    } while(cr.increment());
  } else if(node.op.is_einsummable()) {
    auto const& einsummable = node.op.get_einsummable();
    for(int which_inn = 0; which_inn != einsummable.inns.size(); ++which_inn) {
      int const& inn = node.inns[which_inn];
      partition_t const& inn_partition = get_refinement_partition(inn);

      vector<int> const& inn_idxs = einsummable.inns[which_inn];

      // The inn -> out can be
      //    complex -> real
      //    real    -> complex
      //    real    -> real
      //    complex -> complex
      // However, the inn partition is always with respect to real,
      // which means if the inn dtype is complex, the last dimension
      // ___on the input___ is actually doubled. The correction for
      // when the input is complex is thus to double the corresponding
      // join dimension regardless of whether or not the join results
      // in a complex or real value.
      dtype_t inn_dtype = einsummable.inn_dtype(which_inn);
      partition_t join_partition_fix =
        dtype_is_complex(inn_dtype)                  ?
        _double_dim(join_partition, inn_idxs.back()) :
        join_partition                               ;

      copyregion_join_inn_t cr(
        join_partition_fix,
        inn_partition,
        inn_idxs);
      do {
        join_infos[cr.idx_join()].deps.insert(rid_t{ inn, cr.idx_inn() });
      } while(cr.increment());
    }
  } else {
    throw std::runtime_error("should not reach: twolayer");
  }
}

void twolayer_insert_refi_outs_from_join_deps(
  graph_t const& graph,
  int join_gid,
  vector<join_t> const& joins,
  std::function<vector<refinement_t>&(int)> get_refis)
{
  for(int join_bid = 0; join_bid != joins.size(); ++join_bid) {
    join_t const& join = joins[join_bid];
    for(auto const& [dep_gid, dep_bid]: join.deps) {
      refinement_t& refi = get_refis(dep_gid)[dep_bid];
      refi.outs.insert(jid_t { join_gid, join_bid });
    }
  }
}

inline partdim_t _fast_unions(vector<vector<uint64_t>> const& _ps) {
  vector<uint64_t> spans = vector_sorted_merges(_ps);
  vector_remove_duplicates(spans);

  return partdim_t { .spans = spans };
}

// Note: Almost a copy of union_partition_holders in src/taskgraph.cc
partition_t union_partitions(vector<partition_t> const& ps)
{
  if(ps.size() == 0) {
    throw std::runtime_error("union partitions: input is empty");
    return ps[0];
  }
  if(ps.size() == 1) {
    return ps[0];
  }

  vector<partdim_t> partdims;
  int rank = ps[0].block_shape().size();
  partdims.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    vector<vector<uint64_t>> xs;
    xs.reserve(ps.size());
    for(auto const& p: ps) {
      xs.push_back(p.partdims[i].spans);
    }
    partdims.push_back(_fast_unions(xs));
  }
  return partition_t(partdims);
}

partition_t twolayer_construct_refinement_partition(
  graph_t const& graph,
  int join_gid,
  std::function<partition_t const&(int)> get_partition)
{
  // Note that the refinement partition is with respect to the real dtype
  auto const& join_node = graph.nodes[join_gid];

  if(join_node.outs.size() == 0) {
    throw std::runtime_error(
      "this node has no outs so it can't have a refinement partition");
  }

  dtype_t join_dtype = join_node.op.out_dtype();
  bool join_is_complex = dtype_is_complex(join_dtype);

  vector<partition_t> usage_partitions;
  usage_partitions.reserve(2*join_node.outs.size());
  auto insert_usage = [&](partition_t p) {
    if(join_is_complex) {
      double_last_dim_inplace(p);
    }
    usage_partitions.push_back(p);
  };

  for(auto const& out_gid: join_node.outs) {
    auto const& out_node = graph.nodes[out_gid];
    auto const& out_part = get_partition(out_gid);
    if(out_node.op.is_formation()) {
      insert_usage(out_part);
    } else if(out_node.op.is_complexer()) {
      if(join_is_complex) {
        // complex -> real
        usage_partitions.push_back(out_part);
      } else {
        // real -> complex
        usage_partitions.push_back(double_last_dim(out_part));
      }
    } else if(out_node.op.is_einsummable()) {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage partitions to collect
      auto const& einsummable = out_node.op.get_einsummable();
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          insert_usage(partition_t(einsummable.get_input_from_join(
            out_part.partdims,
            which_input)));
        }
      }
    } else if(out_node.op.is_concat()) {
      auto const& concat = out_node.op.get_concat();
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          insert_usage(concat_get_input_partition(
            out_part, concat, which_input));
        }
      }
    } else if(out_node.op.is_subset()) {
      insert_usage(make_subset_input_partition(
        out_node.op.get_subset(),
        out_part));
    } else {
      throw std::runtime_error("setup refinement part: should not reach");
    }
  }

  return union_partitions(usage_partitions);
}

vector<refinement_t> twolayer_construct_refis_and_connect_joins(
  graph_t const& graph,
  int gid,
  vector<join_t>& joins,
  partition_t const& join_partition,
  partition_t const& refi_partition)
{
  auto refi_shape = refi_partition.block_shape();
  vector<refinement_t> refis(product(refi_shape));
  twolayer_connect_join_to_refi(
    graph, gid, joins, join_partition, refis, refi_partition);
  return refis;
}

void twolayer_connect_join_to_refi(
  graph_t const& graph,
  int gid,
  vector<join_t>& joins,
  partition_t const& join_partition_,
  vector<refinement_t>& refis,
  partition_t const& refi_partition)
{
  auto const& node = graph.nodes[gid];

  partition_t join_partition = join_partition_;

  int join_rank = node.op.rank();
  int out_rank  = node.op.out_rank();
  int agg_rank  = join_rank - out_rank;

  // fix join_partition and dtype_sz so that they are
  // with respect to reals
  dtype_t _dtype = node.op.out_dtype();
  uint64_t dtype_sz = dtype_size(_dtype);
  if(dtype_is_complex(_dtype)) {
    partdim_t& pd = join_partition.partdims[out_rank-1];
    pd = partdim_t::from_sizes(vector_double(pd.sizes()));

    dtype_sz /= 2;
  }

  auto join_shape = join_partition.block_shape();
  auto const& _join_partdims = join_partition.partdims;

  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  std::optional<vector<int>> maybe_agg_shape;
  if(agg_rank > 0) {
    partition_t agg_partition(vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end()));

    maybe_agg_shape = agg_partition.block_shape();
  }

  copyregion_full_t copyregion(refi_partition, out_partition);
  do {
    int const& refi_bid = copyregion.idx_aa;

    auto& refi = refis[refi_bid];
    refi.units.push_back(agg_unit_t {
      .size = dtype_sz * product(copyregion.size),
      .deps = {}
    });

    vector<int>& deps = refi.units.back().deps;

    if(maybe_agg_shape) {
      vector<int> const& out_index = copyregion.index_bb;

      vector<int> agg_index(agg_rank, 0);
      auto const& agg_shape = maybe_agg_shape.value();
      deps.reserve(product(agg_shape));
      do {
        vector<int> join_index = vector_concatenate(out_index, agg_index);
        int join_bid = idxs_to_index(join_shape, join_index);
        deps.push_back(join_bid);
        joins[join_bid].outs.insert(refi_bid);
      } while(increment_idxs(agg_shape, agg_index));
    } else {
      // the join index is the out index if there is no agg
      // and there is only one input
      int const& out_bid = copyregion.idx_bb;
      int const& join_bid = out_bid;
      deps.push_back(join_bid);
      joins[join_bid].outs.insert(refi_bid);
    }
  } while(copyregion.increment());
}

void twolayer_erase_refi_deps(vector<refinement_t>& refis)
{
  for(auto& refi: refis) {
    for(auto& unit: refi.units) {
      unit.deps = vector<int>();
    }
  }
}

void twolayer_erase_join_outs(vector<join_t>& joins)
{
  for(auto& join: joins) {
    join.outs = set<int>();
  }
}

void twolayer_erase_join_deps(vector<join_t>& joins)
{
  for(auto& join: joins) {
    join.deps = set<rid_t>();
  }
}

bool operator==(jid_t const& lhs, jid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator!=(jid_t const& lhs, jid_t const& rhs) {
  return !(lhs == rhs);
}
bool operator< (jid_t const& lhs, jid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
bool operator==(rid_t const& lhs, rid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator!=(rid_t const& lhs, rid_t const& rhs) {
  return !(lhs == rhs);
}
bool operator< (rid_t const& lhs, rid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}

std::ostream& operator<<(std::ostream& out, jid_t const& jid) {
  auto const& [gid,bid] = jid;
  out << "jid{" << gid << "," << bid << "}";
  return out;
}
std::ostream& operator<<(std::ostream& out, rid_t const& rid) {
  auto const& [gid,bid] = rid;
  out << "rid{" << gid << "," << bid << "}";
  return out;
}

std::ostream& operator<<(std::ostream& out, join_t const& join) {
  out << "join[deps=" << vector<rid_t>(join.deps.begin(), join.deps.end())
      << ",outs=" << vector<int>(join.outs.begin(), join.outs.end()) << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, refinement_t const& refi) {
  set<int> deps;
  for(auto const& unit: refi.units) {
    for(auto const& dep: unit.deps) {
      deps.insert(dep);
    }
  }

  out << "refi[deps=" << vector<int>(deps.begin(), deps.end())
      << ",outs=" << vector<jid_t>(refi.outs.begin(), refi.outs.end()) << "]";
  return out;
}

