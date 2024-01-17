#include "copyregion.h"

copyregion_t::copyregion_t(
  partition_t const& out,
  partition_t const& inn,
  vector<int> const& inn_index)
  : out(out), inn(inn), inn_index(inn_index)
{
  if(!vector_equal(out.total_shape(), inn.total_shape())) {
    throw std::runtime_error("total shapes do not match");
  }

  inn_hrect = inn.get_hrect(inn_index);
  region = out.get_region(inn_hrect);
  index = vector_mapfst(region);

  info = vector<diminfo_t>(index.size());

  set_info();
}

bool copyregion_t::increment() {
  if(increment_idxs_region(region, index)) {
    set_info();
    return true;
  }
  return false;
}

void copyregion_t::set_info() {
  out_hrect = out.get_hrect(index);

  for(int i = 0; i != index.size(); ++i) {
    auto const& [o_beg, o_end] = out_hrect[i];
    auto const& [i_beg, i_end] = inn_hrect[i];

    uint64_t beg = std::max(o_beg, i_beg);
    uint64_t end = std::min(o_end, i_end);

    info[i].idx = index[i];
    info[i].offset_inn = beg - i_beg;
    info[i].offset_out = beg - o_beg;
    info[i].size = end - beg;
  }
}

partition_t
union_pair_partitions(partition_t const& aa, partition_t const& bb)
{
  if(!vector_equal(aa.total_shape(), bb.total_shape())) {
    throw std::runtime_error("total shapes do not match");
  }

  vector<partdim_t> pds;
  pds.reserve(aa.partdims.size());
  for(int i = 0; i != aa.partdims.size(); ++i) {
    pds.push_back(
      partdim_t::unions({aa.partdims[i], bb.partdims[i]}));
  }

  return partition_t(pds);
}

copyregion_full_t::copyregion_full_t(
  partition_t const& a,
  partition_t const& b)
  : aa(a), bb(b), rr(union_pair_partitions(a,b)),
    idx_aa(0), index_aa(a.partdims.size(), 0), offset_aa(a.partdims.size(), 0),
    idx_bb(0), index_bb(a.partdims.size(), 0), offset_bb(a.partdims.size(), 0),
    idx_rr(0), index_rr(a.partdims.size(), 0),
    rem_idx_aa(a.partdims.size(), 0),
    rem_idx_bb(a.partdims.size(), 0)
{
  block_shape_rr = rr.block_shape();

  breaks_aa.reserve(block_shape_rr.size());
  rem_aa.reserve(block_shape_rr.size());

  breaks_bb.reserve(block_shape_rr.size());
  rem_bb.reserve(block_shape_rr.size());

  size.reserve(block_shape_rr.size());
  for(int i = 0; i != block_shape_rr.size(); ++i) {
    breaks_aa.push_back(rr.partdims[i].refine_counts(aa.partdims[i]));
    rem_aa.push_back(breaks_aa.back()[0]);

    breaks_bb.push_back(rr.partdims[i].refine_counts(bb.partdims[i]));
    rem_bb.push_back(breaks_bb.back()[0]);

    size.push_back(rr.partdims[i].size_at(0));
  }

  strides_aa = vector<int>(block_shape_rr.size());
  strides_bb = vector<int>(block_shape_rr.size());
  int sa = 1;
  int sb = 1;
  for(int i = block_shape_rr.size() - 1; i >= 0; --i) {
    strides_aa[i] = sa;
    sa *= aa.partdims[i].spans.size();

    strides_bb[i] = sb;
    sb *= bb.partdims[i].spans.size();
  }
}

// This is a copy of increment_idxs, but it updates all
// the extra data with the increments
bool copyregion_full_t::increment() {
  bool could_increment = false;

  int r = index_rr.size();
  do {
    r -= 1;
    if(index_rr[r] + 1 == block_shape_rr[r]) {
      index_rr[r] = 0;

      idx_aa -= index_aa[r]*strides_aa[r];
      index_aa[r] = 0;
      offset_aa[r] = 0;
      rem_idx_aa[r] = 0;
      rem_aa[r] = breaks_aa[r][0];

      idx_bb -= index_bb[r]*strides_bb[r];
      index_bb[r] = 0;
      offset_bb[r] = 0;
      rem_idx_bb[r] = 0;
      rem_bb[r] = breaks_bb[r][0];

      size[r] = rr.partdims[r].size_at(0);
    } else {
      idx_rr++;
      index_rr[r]++;
      size[r] = rr.partdims[r].size_at(index_rr[r]);

      rem_aa[r]--;
      if(rem_aa[r] == 0) {
        rem_idx_aa[r]++;
        rem_aa[r] = breaks_aa[r][rem_idx_aa[r]];

        idx_aa += strides_aa[r];
        index_aa[r]++;
        offset_aa[r] = 0;
      } else {
        offset_aa[r] += size[r];
      }

      rem_bb[r]--;
      if(rem_bb[r] == 0) {
        rem_idx_bb[r]++;
        rem_bb[r] = breaks_bb[r][rem_idx_bb[r]];

        idx_bb += strides_bb[r];
        index_bb[r]++;
        offset_bb[r] = 0;
      } else {
        offset_bb[r] += size[r];
      }

      could_increment = true;
      break;
    }
  } while(r > 0);

  return could_increment;
}

partition_t
broadcast_inn_partition(
  partition_t const& join_partition,
  partition_t const& inn_partition,
  vector<int> const& inns)
{
  vector<partdim_t> pds;
  pds.reserve(join_partition.partdims.size());
  for(auto const& pd: join_partition.partdims){
    pds.push_back(partdim_t::singleton(pd.total()));
  }
  for(int w = 0; w != inns.size(); ++w) {
    int const& i = inns[w];
    pds[i] = partdim_t::unions({pds[i], inn_partition.partdims[w]});
  }
  return partition_t(pds);
}

copyregion_join_inn_t::copyregion_join_inn_t(
  partition_t const& partition_join,
  partition_t const& partition_inn_orig,
  vector<int> const& inns):
    partition_inn(broadcast_inn_partition(partition_join, partition_inn_orig, inns))
{
  cr = std::make_shared<copyregion_full_t>(partition_join, partition_inn);

  // now we change the strides of the inn component.
  int rank = partition_join.partdims.size();
  int inn_rank = partition_inn_orig.partdims.size();
  vector<int> strides_inn(rank, 0);
  int sinn = 1;
  for(int i = inn_rank-1; i >= 0; --i) {
    int const& w = inns[i];
    strides_inn[w] = sinn;
    sinn *= partition_inn_orig.partdims[i].spans.size();
  }
  cr->reset_strides_bb(strides_inn);
}

