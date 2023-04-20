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

