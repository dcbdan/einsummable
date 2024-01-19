#include "data_manager.h"

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
data_manager_t::try_to_acquire_impl(vector<int> const& tids)
{
  data_manager_resource_t ret(tids, vector<buffer_t>(tids.size(), nullptr));

  // Note that tids can be all the same value.. So make sure not
  // to create duplicates.

  // a map from tid to which are missing
  map<int, vector<int>> missing;
  for(int which = 0; which != tids.size(); ++which) {
    int const& tid = tids[which];
    auto iter = data.find(tid);
    if(iter == data.end()) {
      missing[tid].push_back(which);
    } else {
      ret.buffers[which] = iter->second;
    }
  }

  uint64_t mem_required = 0;
  for(auto const& [tid, _]: missing) {
    mem_required += get_size(tid);
  }

  if(current_memory_usage + mem_required > max_memory_usage) {
    return std::nullopt;
  }

  for(auto const& [tid, whiches]: missing) {
    uint64_t const& size = get_size(tid);
    buffer_t d = make_buffer(size);
    current_memory_usage += size;
    data.insert({tid, d});
    for(int const& which: whiches) {
      ret.buffers[which] = d;
    }
  }

  return ret;
}

void data_manager_t::release_impl(data_manager_resource_t const& resource) {
  if(!resource.extracted) {
    return;
  }

  for(auto const& tid: resource.used_tids) {
    auto& [usage_rem, is_save, size] = infos.at(tid);
    usage_rem--;
    if(!is_save && usage_rem == 0) {
      data.erase(tid);
      current_memory_usage -= size;
    }
  }
}

