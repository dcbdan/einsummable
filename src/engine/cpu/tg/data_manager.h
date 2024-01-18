#pragma once
#include "../../../base/setup.h"

#include "../../../base/buffer.h"
#include "../../resource_manager.h"

struct data_manager_resource_t {
  vector<int> used_tids;
  vector<buffer_t> buffers;
};

struct data_manager_t
  : rm_template_t< vector<int>, data_manager_resource_t >
{
  struct info_t {
    int usage_rem;
    bool is_save;
    uint64_t size;
  };

  data_manager_t(
    map<int, buffer_t>& data,
    map<int, info_t> const& infos,
    uint64_t max_memory_usage);

  optional<data_manager_resource_t>
  try_to_acquire_impl(vector<int> const& tids);

  void release_impl(data_manager_resource_t const& resource);

private:
  map<int, buffer_t>& data;

  map<int, info_t> infos;

  uint64_t current_memory_usage;
  uint64_t max_memory_usage;

private:
  int& get_usage_rem(int tid) { return infos.at(tid).usage_rem; }
  uint64_t const& get_size(int tid) const { return infos.at(tid).size; }
};
