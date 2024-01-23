#include "../../exec_graph.h"

struct cpu_tg_einsummable_t : exec_graph_t::op_base_t {
  cpu_tg_einsummable_t(
    map<int, data_manager_t::info_t>& dinfos,
    cpu_kernel_executor_t& a,
    einsummable_t const& b,
    int c,
    vector<int> const& d,
    uint64_t e)
    : cpu_executor(a), einsummable(b), out(c), inns(d),
      workspace_size(e)
  {
    dinfos.at(out).usage_rem++;
    for(int const& inn: inns) {
      dinfos.at(inn).usage_rem++;
    }
  }

  cpu_kernel_executor_t& cpu_executor;
  einsummable_t einsummable;
  int out;
  vector<int> inns;
  uint64_t workspace_size;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_tg_einsummable"; }

  int get_priority() const { return 200; }
};

// tg_send_t and rp_recv_t are exec_graph_t::send_t and exec_graph_t::recv_t
// but with the datamap instead of global buffers

struct tg_send_t : exec_graph_t::op_base_t {
  tg_send_t(
    map<int, data_manager_t::info_t>& dinfos,
    int a, int b, int c)
    : src_tid(a), dst_tid(b), dst(c)
  {
    dinfos.at(src_tid).usage_rem++;
  }

  int src_tid;
  int dst_tid;
  int dst;

  void launch(resource_ptr_t rsrc, std::function<void()> callback) const;

  desc_ptr_t resource_description() const;

  void print(std::ostream& out) const {
    out << "tg_send_t {id = " << dst_tid << "}";
  }
};

struct tg_recv_t : exec_graph_t::op_base_t {
  tg_recv_t(
    map<int, data_manager_t::info_t>& dinfos,
    int a, uint64_t b, int c)
    : dst_tid(a), size(b), src(c)
  {
    dinfos.at(dst_tid).usage_rem++;
  }

  int dst_tid;
  uint64_t size;
  int src;

  void launch(resource_ptr_t rsrc, std::function<void()> callback) const;

  desc_ptr_t resource_description() const;

  void print(std::ostream& out) const {
    out << "tg_recv_t {id = " << dst_tid << "}";
  }
};

struct tg_touch_t : exec_graph_t::op_base_t {
  tg_touch_t(
    map<int, data_manager_t::info_t>& dinfos,
    touch_t const& a,
    int b,
    int c,
    int d)
    : touch(a), out(b), inn(c), group_id(d)
  {
    dinfos.at(out).usage_rem++;
    dinfos.at(inn).usage_rem++;
  }

  touch_t touch;
  int out;
  int inn;
  int group_id;

  void launch(resource_ptr_t resource, std::function<void()> callback) const;
  desc_ptr_t resource_description() const;
  void print(std::ostream& out) const { out << "cpu_touch"; }
};