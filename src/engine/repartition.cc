#include "repartition.h"
#include "resource_manager.h"
#include "channel_manager.h"
#include "notifier.h"
#include "exec_graph.h"
#include "exec_state.h"
#include "touch.h"

struct datamap_manager_t
  : rm_template_t< vector<tuple<int, uint64_t>>, vector<buffer_t> >
{
  datamap_manager_t(map<int, buffer_t>& data)
    : data(data)
  {}

  static desc_ptr_t make_desc(vector<tuple<int, uint64_t>> const& xs) {
    return rm_template_t::make_desc(xs);
  }
  static desc_ptr_t make_desc(tuple<int, uint64_t> const& d) {
    return make_desc(vector<tuple<int, uint64_t>>{d});
  }
  static desc_ptr_t make_desc(int tid, uint64_t size) {
    return make_desc({tid,size});
  }
  static desc_ptr_t make_desc(int tid) {
    return make_desc({tid,0});
  }

private:
  optional<vector<buffer_t>>
  try_to_acquire_impl(vector<tuple<int, uint64_t>> const& info)
  {
    std::unique_lock lk(m);

    vector<buffer_t> ret;
    ret.reserve(info.size());

    for(auto const& [tid, size]: info) {
      auto iter = data.find(tid);
      if(iter == data.end()) {
        if(size == 0) {
          throw std::runtime_error("size is zero for datamap allocate");
        }
        // assumption: allocating doesn't take that long
        ret.push_back(make_buffer(size));
        data.insert({tid, ret.back()});
      } else {
        ret.push_back(iter->second);
      }
    }

    return ret;
  }

  void release_impl(vector<buffer_t> const&) {}

  std::mutex m;
  map<int, buffer_t>& data;
};


struct rp_touch_t : exec_graph_t::op_base_t {
  rp_touch_t(touch_t const& t, int out_tid, int b)
    : touch(t), inn_tid(b)
  {
    if(touch.castable) {
      throw std::runtime_error("not supporting aggregation in repartition graph");
    }
    uint64_t out_size =
      dtype_size(touch.dtype) *
      product(vector_from_each_member(touch.selection, uint64_t, d_out));
    out = tuple<int, uint64_t>(out_tid, out_size);
  }

  touch_t touch;

  tuple<int, uint64_t> out;

  int inn_tid;

  void launch(resource_ptr_t rsrc, std::function<void()> callback) const {
    vector<resource_ptr_t> const& resources =
      resource_manager_t::get_resource(rsrc);

    auto buffers = datamap_manager_t::get_resource(resources[0]);
    buffer_t out_buffer = buffers[0];
    buffer_t inn_buffer = buffers[1];

    std::thread t([this, callback, out_buffer, inn_buffer] {
      execute_touch(touch, out_buffer->raw(), inn_buffer->raw());
      callback();
    });

    t.detach();
  }

  desc_ptr_t resource_description() const {
    vector<tuple<int, uint64_t>> mems;
    mems.push_back(out);
    mems.emplace_back(inn_tid, 0);
    desc_ptr_t ptr = datamap_manager_t::make_desc(mems);
    return resource_manager_t::make_desc({ ptr });
  }

  void print(std::ostream& out) const {
    out << "rp_touch";
  }
};

// rp_send_t and rp_recv_t are exec_graph_t::send_t and exec_graph_t::recv_t
// but with the datamap instead of global buffers

struct rp_send_t : exec_graph_t::op_base_t {
  rp_send_t(int a, int b, int c)
    : src_tid(a), dst_tid(b), dst(c)
  {}

  int src_tid;
  int dst_tid;
  int dst;

  void launch(resource_ptr_t rsrc, std::function<void()> callback) const {
    vector<resource_ptr_t> const& resources =
      resource_manager_t::get_resource(rsrc);

    notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

    auto const& wire = channel_manager_t::get_resource(resources[1]);

    buffer_t buffer = datamap_manager_t::get_resource(resources[2])[0];

    std::thread t([this, notifier, wire, buffer, callback] {
      notifier->notify_send_ready(this->dst, this->dst_tid, wire.get_channel());

      wire.send(buffer->raw(), buffer->size);

      callback();
    });

    t.detach();
  }

  desc_ptr_t resource_description() const {
    return resource_manager_t::make_desc(
      vector<desc_ptr_t> {
        notifier_t::make_desc(unit_t{}),
        channel_manager_t::make_desc({ true, dst }),
        datamap_manager_t::make_desc(src_tid)
      }
    );
  }

  void print(std::ostream& out) const {
    out << "rp_send_t {id = " << dst_tid << "}";
  }
};

struct rp_recv_t : exec_graph_t::op_base_t {
  rp_recv_t(int a, uint64_t b, int c)
    : dst_tid(a), size(b), src(c)
  {}

  int dst_tid;
  uint64_t size;
  int src;

  void launch(resource_ptr_t rsrc, std::function<void()> callback) const {
    vector<resource_ptr_t> const& resources =
      resource_manager_t::get_resource(rsrc);

    notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

    buffer_t buffer = datamap_manager_t::get_resource(resources[1])[0];

    auto const& wire = channel_manager_t::get_resource(resources[2]);

    if(buffer->size != size) {
      throw std::runtime_error("how come buffer has incorrect size?");
    }

    std::thread t([this, notifier, wire, buffer, callback] {
      int channel = notifier->get_channel(this->dst_tid);

      wire.recv(buffer->raw(), this->size, channel);

      callback();
    });

    t.detach();
  }

  desc_ptr_t resource_description() const {
    return resource_manager_t::make_desc(
      vector<desc_ptr_t> {
        notifier_t::make_desc(unit_t{}),
        datamap_manager_t::make_desc(dst_tid, size),
        channel_manager_t::make_desc({ false, src })
      }
    );
  }

  void print(std::ostream& out) const {
    out << "rp_recv_t {id = " << dst_tid << "}";
  }
};

exec_graph_t create_repartition_execgraph(
  int this_rank,
  taskgraph_t const& taskgraph)
{
  using op_ptr_t            = exec_graph_t::op_ptr_t;
  using dummy_t             = exec_graph_t::dummy_t;
  using notify_recv_ready_t = exec_graph_t::notify_recv_ready_t;
  using wait_recv_ready_t   = exec_graph_t::wait_recv_ready_t;

  exec_graph_t graph;

  map<int, int> tid_to_eid;

  for(auto const& tid: taskgraph.get_order()) {
    auto const& node = taskgraph.nodes[tid];

    if(!node.op.is_local_to(this_rank)) {
      continue;
    }

    if(node.op.is_input()) {
      op_ptr_t op(new dummy_t());
      int eid = graph.insert(op, {});
      tid_to_eid.insert({tid, eid});
    } else if(node.op.is_move()) {
      auto const& [src,dst,inn_tid,size] = node.op.get_move();
      if(src == this_rank) {
        int inn_eid = tid_to_eid.at(inn_tid);

        int recv_ready_eid = graph.insert(
          op_ptr_t(new exec_graph_t::wait_recv_ready_t(tid, dst)),
          { inn_eid });

        int send_eid = graph.insert(
          op_ptr_t(new rp_send_t(inn_tid, tid, dst)),
          { recv_ready_eid });
      } else if(dst == this_rank) {
        int recv_ready_eid = graph.insert(
          op_ptr_t(new notify_recv_ready_t(tid, src)),
          {} // no deps: always ready to recv
        );

        int recv_eid = graph.insert(
          op_ptr_t(new rp_recv_t(tid, size, src)),
          { recv_ready_eid });

        tid_to_eid.insert({tid, recv_eid});
      } else {
        throw std::runtime_error("should not reach: move not local");
      }
    } else if(node.op.is_partialize()) {
      auto const& touches_from = node.op.get_partialize().as_touches_from_flat();

      vector<int> tmp_eids;
      for(auto const& [inn_tid, touch]: touches_from) {
        op_ptr_t op(new rp_touch_t(touch, tid, inn_tid));

        int inn_eid = tid_to_eid.at(inn_tid);
        int tmp_eid = graph.insert(op, { inn_eid });

        tmp_eids.push_back(tmp_eid);
      }

      int eid = tmp_eids[0];
      if(tmp_eids.size() > 1) {
        op_ptr_t dummy_op(new dummy_t());
        eid = graph.insert(dummy_op, tmp_eids);
      }

      tid_to_eid.insert({tid, eid});
    } else {
      throw std::runtime_error(
        "repartition exec graph only supports input, move and partialize"
        "nodes.");
    }
  }

  return graph;
}

rm_ptr_t create_repartition_resource_manager(
  communicator_t& communicator,
  map<int, buffer_t>& data)
{
  vector<rm_ptr_t> managers;
  managers.reserve(3);
  managers.emplace_back(new datamap_manager_t(data));
  managers.emplace_back(new notifier_t(communicator));
  managers.emplace_back(new channel_manager_t(communicator));
  return rm_ptr_t(new resource_manager_t(managers));
}

void repartition(
  communicator_t& comm,
  remap_relations_t const& _remap,
  map<int, buffer_t>& data)
{
  auto const& remap = _remap.remap;

  auto [remap_gid, g] = create_remap_graph_constructor(_remap);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  int this_rank = comm.get_this_rank();

  _update_map_with_new_tg_inns(
    data, remap_gid, gid_to_inn, _remap, this_rank);

  exec_graph_t execgraph = create_repartition_execgraph(this_rank, taskgraph);

  rm_ptr_t resource_manager = create_repartition_resource_manager(comm, data);

  exec_state_t state(execgraph, resource_manager);
  state.event_loop();

  _update_map_with_new_tg_outs(
    data, remap_gid, gid_to_out, _remap, this_rank);
}

