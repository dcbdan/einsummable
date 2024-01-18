#include "../../exec_graph.h"

struct cpu_einsummable_t : exec_graph_t::op_base_t {
  cpu_einsummable_t(
    cpu_kernel_executor_t& a,
    einsummable_t const& b,
    vector<mem_t> const& c,
    uint64_t d)
    : cpu_executor(a), einsummable(b), mems(c),
      workspace_size(d)
  {}

  cpu_kernel_executor_t& cpu_executor;
  einsummable_t einsummable;
  vector<mem_t> mems;
  uint64_t workspace_size;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_einsummable"; }

  int get_priority() const { return 200; }
};

struct cpu_touch_t : exec_graph_t::op_base_t {
  cpu_touch_t(
    cpu_kernel_executor_t& a,
    touch_t const& b,
    int c,
    vector<mem_t> const& d)
    : cpu_executor(a), touch(b), group_id(c), mems(d)
  {}

  cpu_kernel_executor_t& cpu_executor;
  touch_t touch;
  int group_id;
  vector<mem_t> mems;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_touch"; }
};

struct cpu_evict_t : exec_graph_t::op_base_t {
  cpu_evict_t(int a, mem_t const& b)
    : id(a), mem(b)
  {}

  int id;
  mem_t mem;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_evict"; }
};

struct cpu_load_t : exec_graph_t::op_base_t {
  cpu_load_t(int a, mem_t const& b)
    : id(a), mem(b)
  {}

  int id;
  mem_t mem;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_load"; }
};
