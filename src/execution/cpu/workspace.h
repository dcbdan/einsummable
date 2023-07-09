#pragma once
#include "../../base/setup.h"

#include "../../base/buffer.h"

#include "kernels.h"

struct workspace_manager_t {
  workspace_manager_t(
    kernel_manager_t const& km)
    : kernel_manager(km)
  {}

  tuple<
    optional<tuple<void*, uint64_t>>,
    optional<int>>
  get(einsummable_t const&);

  void release(int which);

  void release(optional<int> w) {
    if(w) { return release(w.value()); }
  }

  kernel_manager_t const& kernel_manager;

  std::mutex m;

  vector<tuple<bool,buffer_t>> workspace;
};


