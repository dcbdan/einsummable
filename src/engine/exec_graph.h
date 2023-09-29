#pragma once
#include "../base/setup.h"

#include "resource_manager.h"

#include "../einsummable/memgraph.h"

#define CPU_EXEC
#define GPU_EXEC

#ifdef CPU_EXEC
#include "cpu/kernel_executor.h"
#include "cpu/workspace_manager.h"
#include "cpu/storage.h"
#endif

#include "gpu/gpu_kernel_manager.h"
#include "gpu/workspace.h"

struct exec_graph_t {
#ifdef CPU_EXEC
  exec_graph_t(cpu_kernel_executor_t& e)
    : cpu_executor(e)
  {}
#endif

#ifdef CPU_EXEC
  static exec_graph_t make_cpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    cpu_kernel_executor_t& cpu_executor);
#endif

#ifdef GPU_EXEC
  exec_graph_t(kernel_manager_t& km)
    : gpu_km(km)
  {}
#endif

#ifdef GPU_EXEC
  static exec_graph_t make_gpu_exec_graph(
    memgraph_t const& memgraph,
    int this_rank,
    kernel_manager_t& gpu_km);
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

  struct cpu_evict_t {
    int id;
    mem_t mem;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };

  struct cpu_load_t {
    int id;
    mem_t mem;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };
#endif

#ifdef GPU_EXEC
  struct gpu_einsummable_t {
    kernel_manager_t& gpu_km;
    einsummable_t einsummable;
    vector<mem_t> mems;
    workspace_info_t worksize;
    int device;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };

  struct gpu_touch_t {
    kernel_manager_t& gpu_km;
    touch_t touch;
    int group_id;
    vector<mem_t> mems;
    int device;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };

  struct gpu_copy_t {
    memgraph_t::move_t move;

    void launch(rsrc_t resource, std::function<void()> callback) const;
    desc_t resource_description() const;
  };

#endif

  using op_t = std::variant<
#ifdef CPU_EXEC
    cpu_touch_t,
    cpu_einsummable_t,
    cpu_evict_t,
    cpu_load_t,
#endif
#ifdef GPU_EXEC
    gpu_touch_t,
    gpu_einsummable_t,
    gpu_copy_t,
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
  // When compiling exec graphs, einsummable nodes are built into
  // this particular executor. An executor could be considered a "resource"
  // that gets passed in to the einsummable nodes, but the same
  // cpu executor would have to get passed in. Also the cpu executor's state
  // does not change during execution, only during compilation.
  cpu_kernel_executor_t& cpu_executor;
#endif

#ifdef GPU_EXEC
  kernel_manager_t& gpu_km;
#endif

private:
  int insert(op_t const& op, vector<int> const& inns);
};
