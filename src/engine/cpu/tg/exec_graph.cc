#include "../../exec_graph.h"

#include "exec_nodes.h"
#include "data_manager.h"
#include "../../channel_manager.h"
#include "../../notifier.h"

tuple<exec_graph_t, map<int, data_manager_t::info_t>>
exec_graph_t::make_cpu_tg_exec_graph(
  taskgraph_t const& taskgraph,
  int this_rank,
  cpu_kernel_executor_t& cpu_executor,
  int num_channels_per_move,
  map<string, scalar_t> const& scalar_vars)
{
  // TODO: remove this once all the kernels compile
  {
    int nfail = 0;
    cpu_kernel_executor_t k;
    for(auto const& node: taskgraph.nodes) {
      if(node.op.is_apply()) {
        einsummable_t e = node.op.get_apply()
          .einsummable
          .replace_scalar_variables(scalar_vars)
          .merge_adjacent_dims();
        auto maybe_worksize = k.build(e);
        if(!maybe_worksize) {
          DOUT(e);
          DOUT(std::get<0>(e.join.to_cpp_bytes()));
          DOUT("");
          nfail++;
        }
      }
    }
    if(nfail > 0) {
      DOUT("num fail: " << nfail);
      throw std::runtime_error("will not be able to compile all the kernels");
    }
  }

  using dinfo_t = data_manager_t::info_t;

  map<int, dinfo_t> dinfos;

  exec_graph_t graph;

  map<int, int> tid_to_eid;

  auto insert_from_tid = [&](op_ptr_t op, int tid) {
    auto const& node = taskgraph.nodes[tid];

    vector<int> inns;
    for(auto const& inn_tid: node.op.inputs()) {
      inns.push_back(tid_to_eid.at(inn_tid));
    }

    int eid = graph.insert(op, inns);

    tid_to_eid.insert({tid, eid});
  };

  int _group_id = 0;
  auto new_group_id = [&] {
    int ret = _group_id;
    _group_id += 1;
    return ret;
  };

  // initialize dinfos first
  for(int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
    if(!taskgraph.is_local_to(tid, this_rank)) {
      continue;
    }

    auto const& node = taskgraph.nodes[tid];

    if(node.op.is_move()) {
      auto const& [src,dst,inn_tid,size] = node.op.get_move();
      if(src == this_rank) {
        // the resulting tensor doesn't actually live here so no need to
        // add to dinfos
      } else if(dst == this_rank) {
        dinfos.insert({tid, dinfo_t {
          .usage_rem = 0,
          .is_save = node.is_save,
          .size = node.op.out_size()
        }});
      } else {
        throw std::runtime_error("should not reach");
      }
    } else {
      dinfos.insert({tid, dinfo_t {
        .usage_rem = 0,
        .is_save = node.is_save,
        .size = node.op.out_size()
      }});
    }
  }

  for(int tid = 0; tid != taskgraph.nodes.size(); ++tid) {
    if(!taskgraph.is_local_to(tid, this_rank)) {
      continue;
    }

    auto const& node = taskgraph.nodes[tid];

    if(node.op.is_input()) {
      op_ptr_t op = std::make_shared<dummy_t>();
      insert_from_tid(op, tid);
    } else if(node.op.is_constant()) {
      auto const& constant = node.op.get_constant();
      auto const& fill = constant.fill;
      cpu_tg_fill_constant_t* op = new cpu_tg_fill_constant_t(
        dinfos,
        tid,
        fill);
      insert_from_tid(op_ptr_t(op), tid);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      einsummable_t e = apply
        .einsummable
        .replace_scalar_variables(scalar_vars)
        .merge_adjacent_dims();

      auto maybe_worksize = cpu_executor.build(e);
      if(!maybe_worksize) {
        DOUT(std::get<0>(e.join.to_cpp_bytes()));
        DOUT(e.join.to_cppstr([](int i) { return "x" + write_with_ss(i); }));
        throw std::runtime_error("could not compile the kernel: " + write_with_ss(e));
      }

      cpu_tg_einsummable_t* op = new cpu_tg_einsummable_t(
        dinfos,
        cpu_executor,
        e,
        tid,
        apply.inns,
        maybe_worksize.value());

      // insert into the graph
      insert_from_tid(op_ptr_t(op), tid);
    } else if(node.op.is_move()) {
      auto const& [src,dst,inn_tid,size] = node.op.get_move();
      if(src == this_rank) {
        int inn_eid = tid_to_eid.at(inn_tid);

        int recv_ready_eid = graph.insert(
          op_ptr_t(new exec_graph_t::wait_recv_ready_t(tid, dst)),
          { inn_eid });

        int send_eid = graph.insert(
          op_ptr_t(new tg_send_t(dinfos, inn_tid, tid, dst)),
          { recv_ready_eid });
      } else if(dst == this_rank) {
        int recv_ready_eid = graph.insert(
          op_ptr_t(new notify_recv_ready_t(tid, src)),
          {} // no deps: always ready to recv
        );

        int recv_eid = graph.insert(
          op_ptr_t(new tg_recv_t(dinfos, tid, size, src)),
          { recv_ready_eid });

        tid_to_eid.insert({tid, recv_eid});
      } else {
        throw std::runtime_error("should not reach: move not local");
      }
    } else if(node.op.is_partialize()) {
      vector<int> tmp_eids;
      for(auto const& touch_group: node.op.get_partialize().as_touches_from()) {
        int group_id = -1;
        if(touch_group.size() > 1) {
          group_id = new_group_id();
        }

        for(auto const& [inn_tid, touch]: touch_group) {
          op_ptr_t op(new tg_touch_t(dinfos, touch, tid, inn_tid, group_id));

          int inn_eid = tid_to_eid.at(inn_tid);
          int tmp_eid = graph.insert(op, { inn_eid });

          tmp_eids.push_back(tmp_eid);
        }
      }

      int eid = tmp_eids[0];
      if(tmp_eids.size() > 1) {
        op_ptr_t dummy_op(new dummy_t());
        eid = graph.insert(dummy_op, tmp_eids);
      }

      tid_to_eid.insert({tid, eid});
    } else {
      throw std::runtime_error("missing einsummable node impl");
    }
  }

  return {graph, dinfos};
}

void cpu_tg_einsummable_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [out_mems, inn_mems] = data_manager_t::get_resource(resources[0]).extract();
  void* out_mem = out_mems[0];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[1]);

  optional<buffer_t> maybe_workspace;
  if(workspace_size > 0) {
    maybe_workspace = make_buffer(workspace_size);
  }

  thread_resource.launch(
    [this, callback, out_mem, inn_mems, maybe_workspace]
    {
      cpu_executor(einsummable, out_mem, inn_mems, maybe_workspace);
      callback();
    }
  );
  // Note: capturing this under the assumption that the exec_graph will not
  //       change and invalidate the this pointer
}

desc_ptr_t cpu_tg_einsummable_t::resource_description() const
{
  vector<desc_ptr_t> ret;
  ret.emplace_back(data_manager_t::make_desc(out, inns));
  ret.emplace_back(threadpool_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

void cpu_tg_fill_constant_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [out_mems, _] = data_manager_t::get_resource(resources[0]).extract();
  void* out_mem = out_mems[0];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[1]);

  thread_resource.launch(
    [this, callback, out_mem]
    {
      initialize_fill(this->fill, out_mem);
      callback();
    });
}

desc_ptr_t
cpu_tg_fill_constant_t::resource_description() const
{
  vector<desc_ptr_t> ret;

  ret.emplace_back(data_manager_t::make_desc(tid, {}));
  ret.emplace_back(threadpool_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

void tg_send_t::launch(resource_ptr_t rsrc, std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  auto const& wire = send_channel_manager_t::get_resource(resources[1]);

  auto const& [_, mems] = data_manager_t::get_resource(resources[0]).extract();
  void const* mem = mems[0];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[3]);

  thread_resource.launch([this, notifier, wire, mem, callback] {
    notifier->notify_send_ready(this->dst, this->dst_tid, wire.channel);

    wire.send(mem, this->size);

    callback();
  });
}

desc_ptr_t tg_send_t::resource_description() const
{
 return resource_manager_t::make_desc(
   vector<desc_ptr_t> {
     notifier_t::make_desc(unit_t{}),
     send_channel_manager_t::make_desc(dst),
     data_manager_t::make_desc({}, { src_tid }),
     threadpool_manager_t::make_desc()
   }
 );
}

void tg_recv_t::launch(resource_ptr_t rsrc, std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [mems, _] = data_manager_t::get_resource(resources[0]).extract();
  void* mem = mems[0];

  auto const& wire = recv_channel_manager_t::get_resource(resources[1]);

  auto& thread_resource = threadpool_manager_t::get_resource(resources[2]);

  thread_resource.launch([this, wire, mem, callback] {
    wire.recv(mem, this->size);
    callback();
  });
}

desc_ptr_t tg_recv_t::resource_description() const
{
  return resource_manager_t::make_desc(
    vector<desc_ptr_t> {
      data_manager_t::make_desc(dst_tid, {}),
      recv_channel_manager_t::make_desc({ dst_tid, src }),
      threadpool_manager_t::make_desc()
    }
  );
}

void tg_touch_t::launch(resource_ptr_t rsrc, std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [out_mems, inn_mems] = data_manager_t::get_resource(resources[0]).extract();
  void*       out_mem = out_mems[0];
  void const* inn_mem = inn_mems[0];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[1]);

  bool is_first = false;
  if(group_id >= 0) {
    tuple<int, bool> const& info = group_manager_t::get_resource(resources[2]);
    is_first = std::get<1>(info);
  }

  touch_t this_touch = touch;
  if(is_first) {
    // if this is the first touch, make sure the touch becomes a copy
    this_touch.castable = std::nullopt;
  }

  thread_resource.launch([this, callback, this_touch, out_mem, inn_mem] {
    execute_touch(this_touch, out_mem, inn_mem);
    callback();
  });
}

desc_ptr_t tg_touch_t::resource_description() const
{
  vector<desc_ptr_t> ret;

  ret.emplace_back(data_manager_t::make_desc(out, {inn}));
  ret.emplace_back(threadpool_manager_t::make_desc());

  if(group_id >= 0) {
    ret.emplace_back(group_manager_t::make_desc(group_id));
  }

  return resource_manager_t::make_desc(ret);
}

