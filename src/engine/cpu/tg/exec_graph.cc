#include "../../exec_graph.h"

#include "exec_nodes.h"
#include "data_manager.h"
#include "../../channel_manager.h"
#include "../../notifier.h"

static
vector<exec_graph_t::op_ptr_t>
create_parallel_bmm(
  full_contraction_t const& c,
  int num_threads,
  map<int, data_manager_t::info_t>& dinfos,
  int out_tid,
  int lhs_tid,
  int rhs_tid)
{
  using op_ptr_t = exec_graph_t::op_ptr_t;
  // All modes in full_contraction_t are one of the following:
  //   on-lhs on-rhs on-out
  //   T      T      T       b
  //   T      T      F       j
  //   T      F      T       i
  //   F      T      T       k
  // bij,bjk->bik

  // Come up with some plan to parallelize over the b and i dimensions,
  // prefering to split along the b dimension where possible.
  //
  // It must be the case that the number of parts (pd_b.num_parts() * pd_i.num_parts())
  // is less than or equal to num_threads.
  partdim_t pd_b;
  partdim_t pd_i;
  if(c.nb >= num_threads) {
    pd_b = partdim_t::split(c.nb, num_threads);
    pd_i = partdim_t::singleton(c.ni);
  } else if(c.ni >= num_threads) {
    pd_b = partdim_t::singleton(c.nb);
    pd_i = partdim_t::split(c.ni, num_threads);
  } else if(num_threads % 2 == 0 && c.nb >= (num_threads / 2)) {
    pd_b = partdim_t::split(c.nb, num_threads / 2);
    pd_i = partdim_t::split(c.ni, 2);
  } else if(num_threads % 2 == 0 && c.ni >= (num_threads / 2)) {
    pd_b = partdim_t::split(c.nb, 2);
    pd_i = partdim_t::split(c.ni, num_threads / 2);
  } else if(c.nb >= c.ni) {
    pd_b = partdim_t::split(c.nb, num_threads);
    pd_i = partdim_t::singleton(c.ni);
  } else {
    pd_b = partdim_t::singleton(c.nb);
    pd_i = partdim_t::split(c.ni, num_threads);
  }

  vector<op_ptr_t> ret;
  ret.reserve(pd_b.num_parts() * pd_i.num_parts());
  for(int which_b = 0; which_b != pd_b.num_parts(); ++which_b) {
  for(int which_i = 0; which_i != pd_i.num_parts(); ++which_i) {
    ret.emplace_back(new cpu_tg_batchmatmul_t(
      dinfos,
      c.dtype,
      c.nb, pd_b.offset_at(which_b), pd_b.size_at(which_b),
      c.ni, pd_i.offset_at(which_i), pd_i.size_at(which_i),
      c.nj, c.nk,
      c.trans_lhs, c.trans_rhs,
      out_tid, lhs_tid, rhs_tid));
  }}

  return ret;
}


tuple<exec_graph_t, map<int, data_manager_t::info_t>>
exec_graph_t::make_cpu_tg_exec_graph(
  taskgraph_t const& taskgraph,
  int this_rank,
  cpu_kernel_executor_t& cpu_executor,
  int num_channels_per_move,
  int num_threads_per_contraction,
  map<string, scalar_t> const& scalar_vars)
{
  // TODO: remove this once all the kernels compile
  {
    int nfail = 0;
    cpu_kernel_executor_t k;
    set<string> ss;
    for(auto const& node: taskgraph.nodes) {
      if(node.op.is_apply()) {
        einsummable_t e = node.op.get_apply()
          .einsummable
          .replace_scalar_variables(scalar_vars)
          .merge_adjacent_dims();
        auto maybe_worksize = k.build(e);
        if(!maybe_worksize) {
          DOUT(e);
          string s = std::get<0>(e.join.to_cpp_bytes());
          DOUT(s);
          ss.insert(write_with_ss(e.join));
          DOUT("");
          nfail++;
        }
      }
    }
    if(nfail > 0) {
      DOUT("num fail: " << nfail);

      for(auto const& s: ss) {
        DOUT("\""+s+"\",");
      }
      DOUT(ss.size());

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

  int _new_tid = 0;
  auto get_new_tid = [&](uint64_t size) {
    _new_tid -= 1;
    dinfos.insert({ _new_tid, dinfo_t {
      .usage_rem = 0,
      .is_save = false,
      .size = size
      }});
    return _new_tid;
  };
  int _group_id = 0;
  auto new_group_id = [&] {
    int ret = _group_id;
    _group_id += 1;
    return ret;
  };

  auto permute_input_if_necc = [&](
    int inn_tid,
    dtype_t dtype,
    optional<full_contraction_t::permute_t> const& maybe)
  {
    if(maybe) {
      auto const& p = maybe.value();
      uint64_t size = dtype_size(dtype) * product(p.inn_shape);
      int out_tid = get_new_tid(size);
      cpu_tg_permute_t* op = new cpu_tg_permute_t(
        dinfos,
        dtype, p.inn_shape, p.out_perm,
        out_tid,
        inn_tid);

      int const& inn_eid = tid_to_eid.at(inn_tid);
      int out_eid = graph.insert(op_ptr_t(op), vector<int>{ inn_eid });
      tid_to_eid.insert({out_tid, out_eid});
      return out_tid;
    } else {
      return inn_tid;
    }
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
        // The resulting tensor doesn't actually live here so no need to
        // add to dinfos. The source tensor does live here and will be initialized
        // into dinfos at another node.
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

  for(int const& tid: taskgraph.get_order()) {
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

      if(e.is_contraction() && num_threads_per_contraction > 1) {
        dtype_t dtype = e.out_dtype();
        full_contraction_t c = full_contraction_t::make(
          dtype, e.join_shape, e.inns[0], e.inns[1], e.out_rank);

        int lhs_tid = apply.inns[0];
        int rhs_tid = apply.inns[1];

        lhs_tid = permute_input_if_necc(lhs_tid, c.dtype, c.perm_lhs);
        rhs_tid = permute_input_if_necc(rhs_tid, c.dtype, c.perm_rhs);

        int tmp_tid = tid;
        if(c.perm_out) {
          auto const& p = c.perm_out.value();
          uint64_t size = dtype_size(dtype) * product(p.inn_shape);
          tmp_tid = get_new_tid(size);
        }

        vector<op_ptr_t> ops = create_parallel_bmm(
          c, num_threads_per_contraction, dinfos, tmp_tid, lhs_tid, rhs_tid);
        vector<int> bmm_eids;

        int lhs_eid = tid_to_eid.at(lhs_tid);
        int rhs_eid = tid_to_eid.at(rhs_tid);
        for(op_ptr_t op: ops) {
          bmm_eids.push_back(graph.insert(op, vector<int>{ lhs_eid, rhs_eid }));
        }

        int out_eid;
        if(c.perm_out) {
          auto const& p = c.perm_out.value();
          cpu_tg_permute_t* op = new cpu_tg_permute_t(
            dinfos,
            dtype, p.inn_shape, p.out_perm,
            tid,
            tmp_tid);
          out_eid = graph.insert(op_ptr_t(op), bmm_eids);
        } else {
          if(bmm_eids.size() == 1) {
            out_eid = bmm_eids[0];
          } else {
            op_ptr_t op = std::make_shared<dummy_t>();
            out_eid = graph.insert(op, bmm_eids);
          }
        }
        tid_to_eid.insert({tid, out_eid});
      } else {
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
      }
    } else if(node.op.is_move()) {
      auto const& [src,dst,inn_tid,size] = node.op.get_move();
      if(src == this_rank) {
        int inn_eid = tid_to_eid.at(inn_tid);

        int ready_eid = graph.insert(
          op_ptr_t(new exec_graph_t::wait_recv_ready_t(tid, dst)),
          { inn_eid });

        int send_eid = graph.insert(
          op_ptr_t(new tg_send_t(dinfos, inn_tid, tid, dst)),
          { ready_eid });
      } else if(dst == this_rank) {
        int ready_eid = graph.insert(
          op_ptr_t(new exec_graph_t::notify_recv_ready_t(tid, src)),
          {} // no deps: always ready to recv
        );

        int recv_eid = graph.insert(
          op_ptr_t(new tg_recv_t(dinfos, tid, size, src)),
          { ready_eid });

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
      string s;
      if(einsummable.is_contraction()) {
        s = "contraction";
      } else if(einsummable.has_aggregation()) {
        s = "reduction";
      } else {
        s = "elementwise:" + write_with_ss(einsummable.join);
      }
      {
        auto g = get_timetracker().make_totals_gremlin(s, product(einsummable.join_shape));
        cpu_executor(einsummable, out_mem, inn_mems, maybe_workspace);
      }
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

void cpu_tg_batchmatmul_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [out_mems, inn_mems] = data_manager_t::get_resource(resources[0]).extract();
  void* out_mem = out_mems[0];
  void const* lhs_mem = inn_mems[0];
  void const* rhs_mem = inn_mems[1];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[1]);

  thread_resource.launch(
    [this, callback, out_mem, lhs_mem, rhs_mem]
    {
      {
        auto g = get_timetracker().make_totals_gremlin("batchmatmul");
        batch_matrix_multiply(dtype,
          offset_b, size_b,
          true, true, true,
          ni, offset_i, size_i,
          nj, nk,
          trans_lhs, trans_rhs,
          out_mem, lhs_mem, rhs_mem,
          false);
      }
      callback();
    }
  );
}

desc_ptr_t cpu_tg_batchmatmul_t::resource_description() const
{
  vector<desc_ptr_t> ret;
  ret.emplace_back(data_manager_t::make_desc(out_tid, vector<int>{ lhs_tid, rhs_tid }));
  ret.emplace_back(threadpool_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

void cpu_tg_permute_t::launch(
  resource_ptr_t rsrc,
  std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  auto const& [out_mems, inn_mems] = data_manager_t::get_resource(resources[0]).extract();
  void* out_mem = out_mems[0];
  void const* inn_mem = inn_mems[0];

  auto& thread_resource = threadpool_manager_t::get_resource(resources[1]);

  thread_resource.launch(
    [this, callback, out_mem, inn_mem]
    {
      { 
        auto g = get_timetracker().make_totals_gremlin("permute");
        permute_kernel(dtype, 1024, inn_shape, out_perm, out_mem, inn_mem);
      }
      callback();
    }
  );
}

desc_ptr_t cpu_tg_permute_t::resource_description() const
{
  vector<desc_ptr_t> ret;
  ret.emplace_back(data_manager_t::make_desc(out_tid, vector<int>{ inn_tid }));
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
      { 
        auto g = get_timetracker().make_totals_gremlin("fill");
        initialize_fill(this->fill, out_mem);
      }
      callback();
    });
}

desc_ptr_t
cpu_tg_fill_constant_t::resource_description() const
{
  vector<desc_ptr_t> ret;

  ret.emplace_back(data_manager_t::make_desc(tid, vector<int>{}));
  ret.emplace_back(threadpool_manager_t::make_desc());

  return resource_manager_t::make_desc(ret);
}

void tg_send_t::launch(resource_ptr_t rsrc, std::function<void()> callback) const
{
  vector<resource_ptr_t> const& resources =
    resource_manager_t::get_resource(rsrc);

  notifier_t* notifier = notifier_t::get_resource(resources[0]).self;

  auto const& wire = send_channel_manager_t::get_resource(resources[1]);

  auto const& [_, mems] = data_manager_t::get_resource(resources[2]).extract();
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
     data_manager_t::make_desc(vector<int>{}, { src_tid }),
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
      data_manager_t::make_desc(dst_tid, vector<int>{}),
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
    {
      auto g = get_timetracker().make_totals_gremlin("touch", 
        product(vector_from_each_member(this_touch.selection, uint64_t, d_out)));
      execute_touch(this_touch, out_mem, inn_mem);
    }
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

