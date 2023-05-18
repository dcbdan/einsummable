#include "memgraph.h"

memloc_t mem_t::as_memloc(int loc) const {
  return memloc_t {
    .offset = offset,
    .size = size,
    .loc = loc
  };
}

mem_t memloc_t::as_mem() const {
  return mem_t {
    .offset = offset,
    .size = size
  };
}

memgraph_t::memgraph_t(
  int nl, int nc, vector<int> const& cs)
  : num_compute_locs(nl), num_cache_locs(nc), cache_locs(cs)
{}

void memgraph_t::print_graphviz(std::ostream& out) const {
  using std::endl;

  string tab = "  ";
  out << "digraph {" << endl;

  for(int id = 0; id != nodes.size(); ++id) {
    node_t const& node = nodes[id];
    op_t const& op = node.op;

    string label;
    string color = "";
    if(op.is_input()) {
      input_t const& input = op.get_input();
      string memloc = write_with_ss(
        memloc_t { input.offset, input.size, input.loc });
      label = "input@" + memloc;
    } else if(op.is_apply()) {
      apply_t const& apply = op.get_apply();
      auto const& aop = apply.op;
      string header;
      string aopstr;
      if(std::holds_alternative<einsummable_t>(aop)) {
        header = "apply";
        aopstr = write_with_ss(std::get<einsummable_t>(aop));
      } else if(std::holds_alternative<touch_t>(aop)) {
        header = "touch";
        aopstr = write_with_ss(std::get<touch_t>(aop).castable) +
          "group" + write_with_ss(apply.group);
      } else {
        throw std::runtime_error("parint graphviz should not reach");
      }
      label = header + "@loc" + write_with_ss(apply.loc) + "." + aopstr;
      for(mem_t const& mem: apply.mems) {
        label += "|" + write_with_ss(mem);
      }
    } else if(op.is_move()) {
      move_t const& move = op.get_move();

      auto const& [src_loc, src_offset] = move.src;
      auto const& [dst_loc, dst_offset] = move.dst;
      auto const& size = move.size;

      label = "move@" +
        write_with_ss(memloc_t { src_offset, size, src_loc }) +
        "->" +
        write_with_ss(memloc_t { dst_offset, size, dst_loc });
      color = "lightgray";
    } else if(op.is_evict()) {
      evict_t const& evict = op.get_evict();
      label = "evict@" +
        write_with_ss(memloc_t { evict.offset, evict.size, evict.loc }) +
        "->cid" +
        write_with_ss(evict.cache_id);
    } else if(op.is_load()) {
      load_t const& load = op.get_load();
      label = "load@" +
        write_with_ss(memloc_t { load.offset, load.size, load.loc }) +
        "->cid" +
        write_with_ss(load.cache_id);
    } else if(op.is_del()) {
      del_t const& del = op.get_del();
      string memloc = write_with_ss(
        memloc_t { del.offset, del.size, del.loc });
      label = "del@" + memloc;
    } else {
      throw std::runtime_error("memgraph print should not happen");
    }

    auto memlocs = op.get_memlocs();
    for(int i = 1; i != memlocs.size(); ++i) {
      if(memlocs[0].offset == memlocs[i].offset) {
        // this argument is donated
        color = "green";
      }
    }

    out << tab
      << "n" << id
      << " [label=\"" << label << "\"";
    if(color != "") {
      out << ", color=" << color;
    }
    out << "]" << endl;

    for(int const& inn_id: node.inns) {
      out << tab << "n" << inn_id << " -> " << "n" << id << endl;
    }
  }
  out << "}" << endl;
}

vector<uint64_t> memgraph_t::mem_sizes() const {
  vector<uint64_t> ret(num_compute_locs, 0);
  for(auto const& node: nodes) {
    for(auto const& memloc: node.op.get_memlocs()) {
      ret[memloc.loc] = std::max(ret[memloc.loc], memloc.offset + memloc.size);
    }
  }
  return ret;
}

int memgraph_t::insert(memgraph_t::op_t op, set<int> const& deps) {
  // Note that deps may include dependencies that are shadowed
  // by other dependencies.
  // Consider inserting node 2 where deps = {0,1} but 0->1 is already
  // an dependency.
  //
  // In this case, adding the 0->2 is shadowd by 1, so nodes[2].inns
  // should only contain {1}.
  //
  //    0------.
  //    |      |
  //    |      v
  //    |      1
  //    -->2<--.

  set<int> inns;

  vector<int> deps_vec(deps.begin(), deps.end());
  if(deps_vec.size() == 0) {
    //
  } else if(deps_vec.size() == 1) {
    inns.insert(deps_vec[0]);
  } else {
    std::sort(deps_vec.begin(), deps_vec.end(), std::greater<int>());
    set<int> unnec;
    for(int i = 0; i != deps_vec.size(); ++i) {
      int const& id = deps_vec[i];
      if(unnec.count(id) == 0) {
        if(i != deps_vec.size() - 1) {
          for(int j = i+1; j != deps_vec.size(); ++j) {
            int const& jd = deps_vec[j];
            if(depends_on(id, jd)) {
              unnec.insert(jd);
            }
          }
        }
        inns.insert(id);
      }
    }
  }

  nodes.push_back(node_t {
    .op = op,
    .inns = inns,
    .outs = {}
  });

  int ret = nodes.size() - 1;

  for(auto const& inn: deps) {
    nodes[inn].outs.insert(ret);
  }

  all_deps.emplace_back(ret, 0);

  vector<char>& ret_deps = all_deps.back();
  for(int const& inn: inns) {
    ret_deps[inn] = 1;

    vector<char>& inn_deps = all_deps[inn];
    for(int i = 0; i != inn_deps.size(); ++i) {
      ret_deps[i] = std::max(ret_deps[i], inn_deps[i]);
    }
  }

  return ret;
}

bool memgraph_t::depends_on(int top, int bot) const {
  if(top > bot) {
    return all_deps[top][bot];
  } else {
    return false;
  }
}

void memgraph_t::op_t::check_op() const {
  if(is_input()) {
    check_input();
  } else if(is_apply()){
    check_apply();
  } else if(is_move()) {
    check_move();
  } else if(is_evict()) {
    check_evict();
  } else if(is_load()) {
    check_load();
  } else if(is_del()) {
    check_del();
  } else {
    throw std::runtime_error("should not reach");
  }
}

void memgraph_t::op_t::check_input() const {}
void memgraph_t::op_t::check_apply() const {}
void memgraph_t::op_t::check_move()  const {
  move_t const& move = get_move();
  auto const& [src, _0] = move.src;
  auto const& [dst, _1] = move.dst;
  if(src == dst) {
    throw std::runtime_error("move cannot be to same location; that's an apply");
  }
}
void memgraph_t::op_t::check_evict() const {}
void memgraph_t::op_t::check_load()  const {}
void memgraph_t::op_t::check_del()   const {}

vector<memloc_t> memgraph_t::op_t::get_memlocs() const
{
  using std::holds_alternative;
  using std::get;

  if(holds_alternative<input_t>(op)) {
    auto const& input = get<input_t>(op);
    return {
      memloc_t { .offset = input.offset, .size = input.size, .loc = input.loc }
    };
  } else if(holds_alternative<apply_t>(op)) {
    auto const& apply = get<apply_t>(op);
    vector<memloc_t> ret;
    for(mem_t const& mem: apply.mems) {
      ret.push_back(mem.as_memloc(apply.loc));
    }
    return ret;
  } else if(holds_alternative<move_t>(op)) {
    auto const& move = get<move_t>(op);
    auto const& [src_loc, src_offset] = move.src;
    auto const& [dst_loc, dst_offset] = move.dst;
    return {
      memloc_t { .offset = src_offset, .size = move.size, .loc = src_loc },
      memloc_t { .offset = dst_offset, .size = move.size, .loc = dst_loc }
    };
  } else if(holds_alternative<evict_t>(op)) {
    auto const& evict = get<evict_t>(op);
    return {
      memloc_t { .offset = evict.offset, .size = evict.size, .loc = evict.loc }
    };
  } else if(holds_alternative<load_t>(op)) {
    auto const& load = get<load_t>(op);
    return {
      memloc_t { .offset = load.offset, .size = load.size, .loc = load.loc }
    };
  } else if(holds_alternative<del_t>(op)) {
    auto const& del = get<del_t>(op);
    return {
      memloc_t { .offset = del.offset, .size = del.size, .loc = del.loc }
    };
  } else {
    throw std::runtime_error("get_memlocs should not reach");
  }
}

// Get all (inn, which_touch_t) from partialize node out
vector<tuple<int, _which_touch_t>> get_which_touches_from(
  taskgraph_t const& taskgraph,
  int out);

vector<_which_touch_t> get_which_touches_from_to(
  taskgraph_t const& tg,
  int out,
  int inn);

vector<std::variant<_which_node_t, _which_touch_t>>
order_taskgraph(taskgraph_t const& taskgraph);
// ^ TODO: This ordering should be "wide" for parallelism,
//         but not too wide for full breadth-first search.
//         At the moment, this guys is built off of
//         taskgraph.get_order()

tuple<
  map<int, mem_t>, // input -> mem
  map<int, mem_t>, // save -> mem
  memgraph_t>
memgraph_t::make_without_evict(
  taskgraph_t const& taskgraph,
  vector<int> const& which_cache)
{
  int const n_compute_locs = taskgraph.num_locs();
  if(which_cache.size() != n_compute_locs) {
    throw std::runtime_error("incorrect which cache length: memgraph_t::make");
  }

  int n_cache_locs = 0;
  for(int const& cache_loc: which_cache) {
    if(cache_loc < 0) {
      throw std::runtime_error("invalid cache loc");
    }
    n_cache_locs = std::max(n_cache_locs, cache_loc + 1);
  }

  for(int i = 0; i != n_cache_locs; ++i) {
    auto iter = std::find(which_cache.begin(), which_cache.end(), i);
    if(iter == which_cache.end()) {
      throw std::runtime_error("cache locs must be 0, ..., n_cache_locs-1; no missing");
    }
  }

  // set up an allocator for each loc,
  // each with a very large amount of available memory
  vector<allocator_t> allocators(
    n_compute_locs,
    std::numeric_limits<uint64_t>::max());

  memgraph_make_state_t state(
    taskgraph,
    which_cache,
    allocators,
    n_compute_locs, n_cache_locs);

  state.allocate_inputs();

  for(auto which_op: order_taskgraph(taskgraph))
  {
    state.add_to_memgraph(which_op);
  }

  return {
    state.input_to_mem,
    state.save_to_mem,
    state.memgraph
  };
}

memgraph_make_state_t::memgraph_make_state_t(
  taskgraph_t const& tg,
  vector<int> const& which_cache,
  vector<allocator_t> const& as,
  int num_compute,
  int num_cache)
  : taskgraph(tg),
    memgraph(num_compute, num_cache, which_cache),
    allocators(as),
    _group(0)
{
  remaining_usage_counts.reserve(taskgraph.nodes.size());
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    remaining_usage_counts.push_back(node.outs.size());
  }
}

void memgraph_make_state_t::allocate_inputs() {
  // Allocate all the input nodes
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()) {
      if(node.outs.size() == 0 && !node.is_save) {
        throw std::runtime_error(
          "This is goofy: an input to memgraph is not used or saved."
          " Call this again after pruning inputs that don't get used"
          " or saved."
        ); // Also: this implementation would have a
           //       memory leak on non-used-non-saved inputs
      }
      int loc = node.op.output_loc();
      uint64_t sz = node.op.tensor_size();
      auto [offset, deps] = allocators[loc].allocate(sz);
      if(deps.size() != 0) {
        throw std::runtime_error("The alligator is broken");
      }

      mem_t mem {
        .offset = offset,
        .size = sz
      };
      input_to_mem.insert({id, mem});
      if(node.is_save) {
        save_to_mem.insert({id, mem});
      }

      current_tensors.insert({id, offset});

      input_t input_op {
          .loc = loc,
          .offset = offset,
          .size = sz
      };
      int memgraph_id = memgraph.insert(op_t(input_op), {});
      task_node_to_mem.insert({id, memgraph_id});
    }
  }
}

void memgraph_make_state_t::add_to_memgraph(
  std::variant<_which_node_t, _which_touch_t> const& which_op)
{
  int id;
  if(std::holds_alternative<_which_node_t>(which_op)) {
    id = std::get<_which_node_t>(which_op).task_id;
  } else {
    id = std::get<_which_touch_t>(which_op).task_id;
  }

  auto const& node = taskgraph.nodes[id];

  vector<int> used_task_tensors;
  set<int> deps;
  optional<op_t> op;

  if(node.op.is_apply()) {
    auto const& [loc, inns, es] = node.op.get_apply();

    vector<mem_t> mems(1 + inns.size());

    auto inn_shapes = es.inn_shapes();
    for(int i = 0; i != inns.size(); ++i) {
      int const& task_inn = inns[i];
      auto sz = product(inn_shapes[i]);
      mems[i+1] = mem_t {
        .offset = current_tensors.at(task_inn),
        .size = sz
      };

      vector<int> inn_deps = task_to_mem(task_inn);
      deps.insert(inn_deps.begin(), inn_deps.end());

      used_task_tensors.push_back(task_inn);
    }

    uint64_t out_offset = get_output_alloc_if_necc(id, deps);

    // The reason mems[0] is being set last is because
    // get_output may invalidate current_tensors at
    // the input nodes
    mems[0] = mem_t {
      .offset = out_offset,
      .size = node.op.tensor_size()
    };

    op = op_t(apply_t {
      .loc = loc,
      .mems = mems,
      .op = es,
      .group = -1
    });
  } else if(node.op.is_move()) {
    auto const& [src,dst,task_inn,size] = node.op.get_move();

    vector<int> inn_deps = task_to_mem(task_inn);
    deps.insert(inn_deps.begin(), inn_deps.end());

    used_task_tensors.push_back(task_inn);

    uint64_t offset_src = current_tensors.at(task_inn);
    uint64_t offset_dst = get_output_alloc_if_necc(id, deps);

    op = op_t(move_t {
      .src = {src, offset_src},
      .dst = {dst, offset_dst},
      .size = size
    });
  } else if(node.op.is_partialize()) {
    auto const& partialize = node.op.get_partialize();

    auto const& [_0, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
    auto [task_inn, touch] = partialize.get_touch(unit_id, touch_id);

    vector<int> inn_deps = task_to_mem(task_inn);
    deps.insert(inn_deps.begin(), inn_deps.end());

    used_task_tensors.push_back(task_inn);

    mem_t inn_mem {
      .offset = current_tensors.at(task_inn),
      .size = taskgraph.nodes[task_inn].op.tensor_size()
    };
    mem_t out_mem {
      .offset = get_output_alloc_if_necc(id, deps),
      .size = node.op.tensor_size(),
    };

    op = op_t(apply_t {
      .loc = partialize.loc,
      .mems = { out_mem, inn_mem },
      .op = touch,
      .group = get_group_at(id, unit_id)
    });
  } else {
    throw std::runtime_error("should not reach");
  }

  int new_memid = memgraph.insert(op.value(), deps);

  if(std::holds_alternative<_which_node_t>(which_op)) {
    task_node_to_mem.insert({id, new_memid});
  } else {
    task_touch_to_mem.insert({
      std::get<_which_touch_t>(which_op),
      new_memid
    });
  }

  // Now try to delete some things
  for(auto const& used_task_id: used_task_tensors) {
    try_to_delete(used_task_id);
  }
}

int memgraph_make_state_t::get_group_at(int task_id, int unit_id)
{
  auto const& node = taskgraph.nodes[task_id];
  auto const& partialize = node.op.get_partialize();
  auto const& unit = partialize.units[unit_id];
  if(unit.inputs.size() == 1) {
    return -1;
  } else {
    if(to_group.count({task_id, unit_id}) == 0) {
      to_group.insert({ {task_id, unit_id}, _group});
      _group += 1;
    }
    return to_group.at({task_id, unit_id});
  }
}

vector<int> memgraph_make_state_t::task_to_mem(int task_id) const
{
  auto const& node = taskgraph.nodes[task_id];
  if(node.op.is_partialize()) {
    vector<int> ret;
    auto const which_touches =
      get_which_touches_from(taskgraph, task_id);
    for(auto const& [_, which_touch]: which_touches) {
      ret.push_back(task_touch_to_mem.at(which_touch));
    }
    return ret;
  } else if(node.op.is_input() || node.op.is_apply() || node.op.is_move()) {
    return {task_node_to_mem.at(task_id)};
  } else {
    throw std::runtime_error("task_to_mem should not reach");
  }
}

void memgraph_make_state_t::try_to_delete(int task_id)
{
  int& rem = remaining_usage_counts[task_id];
  if(rem == 0) {
    throw std::runtime_error("this should not happen");
  }
  rem -= 1;
  if(rem == 0) {
    if(donated.count(task_id) > 0) {
      // can't be delted since this guy was donated to where it was used
      donated.erase(task_id);
      return;
    }

    auto const& node = taskgraph.nodes[task_id];

    if(node.is_save) {
      // can't be deleted since it is marked for saving
      return;
    }

    int loc = node.op.output_loc();
    uint64_t offset = current_tensors.at(task_id);
    del_t del {
      .loc = loc,
      .offset = offset,
      .size = node.op.tensor_size()
    };

    // The delete of task_id depends on
    // 1. the output applys that used task_id and
    // 2. the touch operations that used task_id
    set<int> del_deps;
    for(int task_out: node.outs) {
      auto const& out_node = taskgraph.nodes[task_out];
      if(out_node.op.is_apply() || out_node.op.is_move()) {
        del_deps.insert(task_node_to_mem.at(task_out));
      } else if(out_node.op.is_partialize()) {
        auto const whiches = get_which_touches_from_to(
          taskgraph,
          task_out,
          task_id);
        for(auto const& which: whiches) {
          del_deps.insert(task_touch_to_mem.at(which));
        }
      }
    }

    int del_id = memgraph.insert(op_t(del), del_deps);
    allocators[loc].free(offset, del_id);

    // this tensor is no longer current
    current_tensors.erase(task_id);
  }
}

uint64_t memgraph_make_state_t::get_output_alloc_if_necc(
  int task_id,
  set<int>& deps)
{
  // partial ops may already have an output
  if(current_tensors.count(task_id) > 0) {
    return current_tensors.at(task_id);
  }
  // otherwise allocate the output or grab
  // memory from something that can be donated
  auto const& node = taskgraph.nodes[task_id];

  mem_t output_mem;
  output_mem.size = node.op.tensor_size();
  // output_mem.offset needs to be set

  bool did_get_donation = false;

  // see if we can avoid allocating memory
  if(node.op.is_apply()) {
    auto const& apply = node.op.get_apply();
    einsummable_t const& es = apply.einsummable;
    if(es.is_straight_elementwise()) {
      for(int const& inn_id: apply.inns) {
        // this tensor can be donated if
        // 1. it does not come from a save node
        // 2. it only has one usage (here)
        auto const& inn_node = taskgraph.nodes[inn_id];
        if(!inn_node.is_save && inn_node.outs.size() == 1) {
          did_get_donation = true;
          output_mem.offset = current_tensors.at(inn_id);

          // this will keep it from getting deleted
          donated.insert(inn_id);

          // inn_id no longer has a tensor
          current_tensors.erase(inn_id);

          break;
        }
      }
    }
  }
  // TODO: if the node is a partialize that only has
  //       one input, and that input is the same size as the output,
  //       and it's not a save, and it has a singleton usage,
  //       the input can be donated

  if(!did_get_donation) {
    int loc = node.op.output_loc();
    auto [offset_, ds] = allocators[loc].allocate(output_mem.size);
    output_mem.offset = offset_;
    deps.insert(ds.begin(), ds.end());
  }

  current_tensors.insert({task_id, output_mem.offset});

  if(node.is_save) {
    save_to_mem.insert({task_id, output_mem});
  }

  return output_mem.offset;
}

vector<std::variant<_which_node_t, _which_touch_t>>
order_taskgraph(taskgraph_t const& taskgraph)
{
  vector<std::variant<_which_node_t, _which_touch_t>> ret;
  ret.reserve(2*taskgraph.nodes.size());

  for(auto const& id: taskgraph.get_order()) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()) {
      // Input nodes have already been provided
    } else if(node.op.is_partialize()) {
      // Every partialize touch should be accounted for
      // from the corresponding input. The idea is
      // if input A becomes ready, immediately
      // increment do touch op B += A.
    } else {
      // this is an apply or move node, but the
      // distinction doesn't matter here
      ret.emplace_back(_which_node_t { .task_id = id });
    }

    // Now that this id is now available, add touches from
    // this id to a partialize out
    for(auto const& out: node.outs) {
      auto const& out_node = taskgraph.nodes[out];
      if(out_node.op.is_partialize()) {
        auto which_touches = get_which_touches_from_to(taskgraph, out, id);
        for(auto const& w: which_touches) {
          ret.emplace_back(w);
        }
      }
    }
  }

  return ret;
}

vector<tuple<int, _which_touch_t>> get_which_touches_from(
  taskgraph_t const& taskgraph,
  int out)
{
  vector<tuple<int, _which_touch_t>> ret;

  auto const& out_node = taskgraph.nodes[out];
  if(!out_node.op.is_partialize()){
    throw std::runtime_error("invalid call to get_which_touches_from");
  }
  // get the partialize.
  // partialize consists of units, units consist of input ops.
  // Figure out if each input op of each unit matches to this
  //   tensor just formed. If so, add it.
  auto const& partialize = out_node.op.get_partialize();
  for(int which_unit = 0; which_unit != partialize.units.size(); ++which_unit)
  {
    auto const& unit = partialize.units[which_unit];
    for(int which_inn = 0; which_inn != unit.inputs.size(); ++which_inn)
    {
      auto const& inn = unit.inputs[which_inn].id;
      ret.push_back({inn, _which_touch_t {
          .task_id = out,
          .unit_id = which_unit,
          .touch_id = which_inn
        }
      });
    }
  }

  return ret;
}

vector<_which_touch_t> get_which_touches_from_to(
  taskgraph_t const& tg,
  int out,
  int inn)
{
  vector<_which_touch_t> ret;
  for(auto const& [inn_,w]: get_which_touches_from(tg, out)) {
    if(inn == inn_) {
      ret.push_back(w);
    }
  }
  return ret;
}


allocator_t::allocator_t(uint64_t memsize)
{
  if(memsize == 0) {
    throw std::runtime_error("invalid memsize for allocator");
  }
  blocks.push_back(block_t {
    .beg = 0,
    .end = memsize,
    .dep = -1
  });
}

void allocator_t::block_t::free(int d) {
  if(!occupied()) {
    throw std::runtime_error("cannot free unoccupied memory block");
  }
  dep = d;
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_available(uint64_t size) {
  using return_t = tuple<iter_t, iter_t, uint64_t>;
  optional<return_t> return_block;
  int min_dep = std::numeric_limits<int>::max();
  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter) {
    block_t& block = *iter;
    if(iter->available()) {
      iter_t ret = iter;
      uint64_t sz = 0;
      int inner_max_dep = -1;
      for(; iter != blocks.end() && iter->available(); ++iter) {
        inner_max_dep = std::max(inner_max_dep,iter->dep.value());
        sz += iter->size();
        if(sz >= size && inner_max_dep <= min_dep) {
          return_block = {ret, iter + 1, sz};
          break;
        }
      }
    }
  }
  return return_block;
}

optional< tuple<uint64_t, vector<int>> >
allocator_t::try_to_allocate(uint64_t size)
{
  using return_t = tuple<uint64_t, vector<int>>;

  auto maybe_info = find_available(size);
  if(maybe_info) {
    auto const& [beg,end,sz] = maybe_info.value();

    // collect the output information
    uint64_t offset = beg->beg;
    vector<int> deps;
    for(auto iter = beg; iter != end; ++iter) {
      if(!iter->dep) {
        throw std::runtime_error("invalid find_available return");
      }
      int const& d = iter->dep.value();
      if(d >= 0) {
        deps.push_back(d);
      }
    }

    // fix blocks
    block_t last_block_copy = *(end-1);

    auto iter = blocks.erase(beg, end);
    auto occupied_iter = blocks.insert(iter, block_t {
      .beg = offset,
      .end = offset+size,
      .dep = optional<int>()
    });
    if(size != sz) {
      blocks.insert(occupied_iter+1, block_t {
        .beg = offset + size,
        .end = last_block_copy.end,
        .dep = last_block_copy.dep
      });
    }
    return optional<return_t>({offset, deps});
  } else {
    return optional<return_t>();
  }
}

tuple<uint64_t, vector<int>>
allocator_t::allocate(uint64_t size)
{
  auto maybe_succ = try_to_allocate(size);
  if(maybe_succ) {
    return maybe_succ.value();
  }
  throw std::runtime_error("allocator_t: could not allocate");
}

void allocator_t::free(uint64_t offset, int del) {
  auto iter = std::lower_bound(blocks.begin(), blocks.end(), offset,
    [](block_t const& blk, uint64_t const& val) {
      return blk.beg < val;
    }
  );

  if(iter == blocks.end() || iter->beg != offset) {
    throw std::runtime_error("cannot del this block");
  }

  block_t& block = *iter;
  block.free(del);
}

void allocator_t::print() const {
  auto& out = std::cout;

  for(auto const& blk: blocks) {
    out << "[" << blk.beg << "," << blk.end << ")@";
    if(blk.dep) {
      int const& d = blk.dep.value();
      if(d < 0) {
        out << "neverassigned";
      } else {
       out << "free;dep" << blk.dep.value();
      }
    } else {
      out << "occupied";
    }
    out << std::endl;
  }
}

std::ostream& operator<<(std::ostream& out, mem_t const& mem) {
  out << "[" << mem.offset << "," << mem.offset+mem.size << ")";
  return out;
}
std::ostream& operator<<(std::ostream& out, memloc_t const& memloc) {
  out << "loc" << memloc.loc << memloc.as_mem();
  return out;
}

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_eq(lhs, rhs);
}
bool operator<(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_lt(lhs, rhs);
}


