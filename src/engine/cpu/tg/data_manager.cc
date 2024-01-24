#include "data_manager.h"

data_manager_desc_t::data_manager_desc_t(
  vector<int> const& ws,
  vector<int> const& rs)
  : write_tids(ws), read_tids(rs)
{
  int num_unique = set<int>(ws.begin(), ws.end()).size();
  if(num_unique != ws.size()) {
    throw std::runtime_error("data_manager_desc_t: can't have duplicate write tids");
  }
}


data_manager_t::data_manager_t(
  map<int, buffer_t>& data,
  map<int, info_t> const& infos,
  uint64_t max_memory_usage)
  : data(data), infos(infos), max_memory_usage(max_memory_usage)
{
  // Assumption: all buffers in data are unique and there is no overlap
  current_memory_usage = 0;
  for(auto const& [_, d]: data) {
    current_memory_usage += d->size;
  }
  if(current_memory_usage > max_memory_usage) {
    throw std::runtime_error("too much init memory given to data_manager_t");
  }
}

optional<data_manager_resource_t>
data_manager_t::try_to_acquire_impl(data_manager_desc_t const& desc)
{
  auto const& [out_tids, inn_tids] = desc;

  data_manager_resource_t ret(
    out_tids, inn_tids,
    vector<void*>(out_tids.size(), nullptr),
    vector<void const*>(inn_tids.size(), nullptr));

  // All the inn tids should already be here
  for(int which = 0; which != inn_tids.size(); ++which) {
    int const& tid = inn_tids[which];
    ret.inns[which] = data.at(tid)->raw();
  }

  // Assumption: out_tids does not contain any duplicates.

  vector<int> which_missing;
  uint64_t mem_required = 0;
  map<int, vector<int>> missing;
  for(int which = 0; which != out_tids.size(); ++which) {
    int const& tid = out_tids[which];
    auto iter = data.find(tid);
    if(iter == data.end()) {
      which_missing.push_back(which);
      mem_required += get_size(tid);
    } else {
      ret.outs[which] = iter->second->raw();
    }
  }

  if(current_memory_usage + mem_required > max_memory_usage) {
    return std::nullopt;
  }

  for(int const& which: which_missing) {
    int const& tid = out_tids[which];
    uint64_t const& size = get_size(tid);
    buffer_t d = make_buffer(size);
    current_memory_usage += size;
    data.insert({tid, d});
    ret.outs[which] = d->raw();
  }

  return ret;
}

void data_manager_t::release_impl(data_manager_resource_t const& resource) {
  if(!resource.extracted) {
    return;
  }

  vector<int> used_tids =
    vector_concatenate(resource.out_tids, resource.inn_tids);
  for(auto const& tid: used_tids)
  {
    auto& [usage_rem, is_save, size] = infos.at(tid);
    usage_rem--;
    if(!is_save && usage_rem == 0) {
      data.erase(tid);
      current_memory_usage -= size;
    }
  }
}

