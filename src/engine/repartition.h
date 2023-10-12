#pragma once
#include "../base/setup.h"

#include "../base/buffer.h"
#include "communicator.h"
#include "../einsummable/relation.h"

void repartition(
  communicator_t& comm,
  remap_relations_t const& remap,
  map<int, buffer_t>& data);

template <typename T>
void _update_map_with_new_tg_inns(
  map<int, T>& data,
  vector<tuple<int,int>> const& remap_gid,
  map<int, vtensor_t<int>> const& gid_to_inn,
  remap_relations_t const& _remap,
  optional<int> this_rank)
{
  auto const& remap = _remap.remap;

  map<int, T> tmp;
  for(int i = 0; i != remap_gid.size(); ++i) {
    auto const& gid  = std::get<0>(remap_gid[i]);
    auto const& info = std::get<0>(remap[i]);
    vector<int> const& locs = info.placement.locations.get();
    vector<int> const& inn_tids = info.tids.get();
    vector<int> const& mid_tids = gid_to_inn.at(gid).get();
    if(inn_tids.size() != mid_tids.size()) {
      throw std::runtime_error("!");
    }
    for(int j = 0; j != inn_tids.size(); ++j) {
      if(!this_rank || locs[j] == this_rank.value()) {
        int const& inn_tid = inn_tids[j];
        int const& mid_tid = mid_tids[j];
        tmp.insert({mid_tid, data.at(inn_tid)});
      }
    }
  }
  data = tmp;
}

template <typename T>
void _update_map_with_new_tg_outs(
  map<int, T>& data,
  vector<tuple<int, int>> const& remap_gid,
  map<int, vtensor_t<int>> const& gid_to_out,
  remap_relations_t const& _remap,
  optional<int> this_rank)
{
  auto const& remap = _remap.remap;

  map<int, T> tmp;
  for(int i = 0; i != remap_gid.size(); ++i) {
    auto const& gid  = std::get<1>(remap_gid[i]);
    auto const& info = std::get<1>(remap[i]);
    vector<int> const& locs = info.placement.locations.get();
    vector<int> const& out_tids = info.tids.get();
    vector<int> const& mid_tids = gid_to_out.at(gid).get();
    for(int j = 0; j != out_tids.size(); ++j) {
      if(!this_rank || locs[j] == this_rank.value()) {
        int const& out_tid = out_tids[j];
        int const& mid_tid = mid_tids[j];
        tmp.insert({out_tid, data.at(mid_tid)});
      }
    }
  }
  data = tmp;
}


