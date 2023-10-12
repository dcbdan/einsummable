#include "../exec_graph.h"
#include "../resource_manager.h"

struct gpu_einsummable_t : exec_graph_t::op_base_t {
  gpu_einsummable_t(
    kernel_manager_t& a,
    einsummable_t const& b,
    vector<mem_t> const& c,
    int d)
    : gpu_km(a), einsummable(b), mems(c), device(d)
  {}

  kernel_manager_t& gpu_km;
  einsummable_t einsummable;
  vector<mem_t> mems;
  int device;
  uint64_t workspace_size;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "gpu_einsummable"; }
};

struct gpu_touch_t : exec_graph_t::op_base_t {
  gpu_touch_t(
    kernel_manager_t& a,
    touch_t const& b,
    int c,
    vector<mem_t> d,
    int e)
    : gpu_km(a), touch(b), group_id(c), mems(d), device(e)
  {}

  kernel_manager_t& gpu_km;
  touch_t touch;
  int group_id;
  vector<mem_t> mems;
  int device;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "gpu_touch"; }
};

struct gpu_copy_t : exec_graph_t::op_base_t {
  gpu_copy_t(memgraph_t::move_t const& m, gpu_comm_t gc)
    : move(m), gpu_comm(gc)
  {}

  memgraph_t::move_t move;
  gpu_comm_t gpu_comm;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "gpu_copy"; }
};
