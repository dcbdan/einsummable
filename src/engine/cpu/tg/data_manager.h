#pragma once
#include "../../../base/setup.h"

#include "../../../base/buffer.h"
#include "../../resource_manager.h"

struct data_manager_t;

struct data_manager_desc_t {
  data_manager_desc_t() {}

  data_manager_desc_t(
    int write_tid,
    vector<int> const& read_tids)
    : data_manager_desc_t(vector<int>{write_tid}, read_tids)
  {}

  data_manager_desc_t(
    vector<int> const& ws,
    vector<int> const& rs);

private:
  vector<int> write_tids;
  vector<int> read_tids;

  friend class data_manager_t;
};

struct data_manager_resource_t {
  tuple<vector<void*>, vector<void const*>>
  extract() const {
    // Ugh. This is hacky, modifying extracted under const.
    // The idea is that only once the extraction has occurred
    // will the corresponding tids have a usage.
    extracted = true;
    return {outs, inns};
  }

private:
  data_manager_resource_t(
    vector<int> const& out_tids_,
    vector<int> const& inn_tids_,
    vector<void*> const& out_mems_,
    vector<void const*> const& inn_mems_)
    : out_tids(out_tids_), inn_tids(inn_tids_),
      outs(out_mems_), inns(inn_mems_),
      extracted(false)
  {}

private:
  friend class data_manager_t;

  vector<int> out_tids;
  vector<int> inn_tids;

  vector<void*> outs;
  vector<void const*> inns;

  mutable bool extracted;
};

struct data_manager_t
  : rm_template_t< data_manager_desc_t, data_manager_resource_t >
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
  try_to_acquire_impl(data_manager_desc_t const& desc);

  void release_impl(data_manager_resource_t const& resource);

  static desc_ptr_t make_desc(int out_tid, vector<int> const& inn_tids) {
    return rm_template_t::make_desc(data_manager_desc_t(out_tid, inn_tids));
  }
  static desc_ptr_t make_desc(vector<int> const& out_tids, vector<int> const& inn_tids) {
    return rm_template_t::make_desc(data_manager_desc_t(out_tids, inn_tids));
  }

private:
  map<int, buffer_t>& data;

  map<int, info_t> infos;

  uint64_t current_memory_usage;
  uint64_t max_memory_usage;

private:
  int& get_usage_rem(int tid) { return infos.at(tid).usage_rem; }
  uint64_t const& get_size(int tid) const { return infos.at(tid).size; }
};
