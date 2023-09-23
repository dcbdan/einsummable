#pragma once
#include "../base/setup.h"

#include "resource_manager.h"

#include "../einsummable/memgraph.h"

#ifdef CPU_EXEC
#include "cpu/kernel_executor.h"
#include "cpu/workspace_manager.h"
#endif

struct exec_graph_t {
#ifdef CPU_EXEC
  static exec_graph_t make_cpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    cpu_kernel_executor_t& cpu_executor);
#endif

  using desc_t = resource_manager_t::desc_t;
  using desc_unit_t = resource_manager_t::desc_unit_t;

  using rsrc_t = resource_manager_t::resource_t;
  using rsrc_unit_t = resource_manager_t::resource_unit_t;

  struct dummy_t {
    void launch(rsrc_t resource, std::function<void()> callback) const {
      callback();
    }
    desc_t resource_description() const {
      return vector<desc_unit_t>();
    }
  };

#ifdef CPU_EXEC
  struct cpu_einsummable_t {
    cpu_kernel_executor_t& cpu_executor;
    einsummable_t einsummable;
    vector<mem_t> mems;
    uint64_t workspace_size;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };

  struct cpu_touch_t {
    cpu_kernel_executor_t& cpu_executor;
    touch_t touch;
    int group_id;
    vector<mem_t> mems;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };
#endif

  using op_t = std::variant<
#ifdef CPU_EXEC
    cpu_touch_t,
    cpu_einsummable_t,
#endif
    dummy_t>;


  struct node_t {
    desc_t resource_description() const;

    void launch(rsrc_t resource, std::function<void()> callback) const;

    op_t op;

    vector<int> inns;

    vector<int> outs;
  };

  vector<node_t> nodes;
#ifdef CPU_EXEC
  cpu_kernel_executor_t& cpu_executor;
#endif
};
