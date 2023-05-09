#include "memgraph.h"

memgraph_t::memgraph_t(
  int nl, int nc, vector<int> const& cs)
  : num_compute_locs(nl), num_cache_locs(nc), cache_locs(cs)
{}

struct _which_node_t {
  int task_id;
};
struct _which_touch_t {
  int task_id;
  int unit_id;
  int touch_id;
};

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_eq(lhs, rhs);
}
bool operator<(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_lt(lhs, rhs);
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

// allocator_t contains a vector of blocks that either
// have been (1) deleted, or (2) are currently occupied
struct allocator_t {
  allocator_t(uint64_t memsize_t);

  // Allocate this much memory if possible and return
  // the offset and all dependents. If there is not
  // free memory of this size, none is returned.
  optional< tuple<uint64_t, vector<int>> >
  try_to_allocate(uint64_t size);

  tuple<uint64_t, vector<int>>
  allocate(uint64_t size);

  // delete this memory, storing the delete dependent
  // for future use of this memory block
  void free(uint64_t offset, int del);

private:
  struct block_t {
    uint64_t beg;
    uint64_t end;

    // dep is none:
    //   this memory is occupied
    // dep is < 0:
    //   this memory is free and can be used without
    //   adding a dependency
    // dep is >= 0:
    //   this memory is free and can only be used
    //   after dep id has been deleted
    optional<int> dep;

    uint64_t size() const { return end - beg; }
    bool occupied() const  { return !dep.has_value(); }
    bool available() const { return !occupied(); }
    void free(int dep);
  };

  vector<block_t> blocks;

  using iter_t = vector<block_t>::iterator;

  optional<tuple<iter_t, iter_t, uint64_t>>
  find_available(uint64_t size);
};

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

  // the output data
  map<int, mem_t> input_to_mem;
  map<int, mem_t> save_to_mem;
  memgraph_t memgraph(n_compute_locs, n_cache_locs, which_cache);

  // set up an allocator for each loc,
  // each with a very large amount of available memory
  vector<allocator_t> allocators(
    n_compute_locs,
    std::numeric_limits<uint64_t>::max());

  // taskgraph ids to offsets
  map<int, uint64_t> current_tensors;

  map<int, int> task_node_to_mem;
  map<_which_touch_t, int> task_touch_to_mem;

  int _group = 0;
  map<tuple<int,int>, int> to_group;
  auto get_group_at = [&](int task_id, int unit_id) {
    auto const& node = taskgraph.nodes[task_id];
    auto const& partialize = node.op.get_partialize();
    if(partialize.units.size() == 1) {
      return -1;
    } else {
      if(to_group.count({task_id, unit_id}) == 0) {
        to_group.insert({ {task_id, unit_id}, _group});
        _group += 1;
      }
      return to_group.at({task_id, unit_id});
    }
  };

  auto task_to_mem = [&](int task_id) -> vector<int> {
    auto const& node = taskgraph.nodes[task_id];
    if(node.op.is_input()) {
      return {};
    } else if(node.op.is_partialize()) {
      vector<int> ret;
      auto const which_touches =
        get_which_touches_from(taskgraph, task_id);
      for(auto const& [_, which_touch]: which_touches) {
        ret.push_back(task_touch_to_mem.at(which_touch));
      }
      return ret;
    } else if(node.op.is_apply() || node.op.is_move()) {
      return {task_node_to_mem.at(task_id)};
    } else {
      throw std::runtime_error("task_to_mem should not reach");
    }
  };

  vector<int> remaining_usage_counts;
  remaining_usage_counts.reserve(taskgraph.nodes.size());
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    remaining_usage_counts.push_back(node.outs.size());
  }

  auto try_to_delete = [&](int task_id) {
    int& rem = remaining_usage_counts[task_id];
    if(rem == 0) {
      throw std::runtime_error("this should not happen");
    }
    rem -= 1;
    if(rem == 0) {
      auto const& node = taskgraph.nodes[task_id];
      if(node.is_save) {
        // stop here since this node can't be deleted
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
  };

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
    }
  }

  for(auto which_op: order_taskgraph(taskgraph)) {
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

    // allocate the output memory if this hasn't already
    // happened (it would've only happened for some partials)
    if(current_tensors.count(id) == 0) {
      int loc = node.op.output_loc();
      uint64_t sz = node.op.tensor_size();
      auto [offset, ds] = allocators[loc].allocate(sz);
      current_tensors.insert({id, offset});
      deps.insert(ds.begin(), ds.end());

      if(node.is_save) {
        mem_t mem {
          .offset = offset,
          .size = sz
        };
        save_to_mem.insert({id, mem});
      }
    }

    if(node.op.is_apply()) {
      auto const& [loc, inns, es] = node.op.get_apply();

      vector<mem_t> mems;

      mems.push_back(mem_t {
        .offset = current_tensors.at(id),
        .size = node.op.tensor_size()
      });

      auto inn_shapes = es.inn_shapes();
      for(int i = 0; i != inns.size(); ++i) {
        int const& task_inn = inns[i];
        auto sz = product(inn_shapes[i]);
        mems.push_back(mem_t {
          .offset = current_tensors.at(task_inn),
          .size = sz
        });

        vector<int> inn_deps = task_to_mem(task_inn);
        deps.insert(inn_deps.begin(), inn_deps.end());

        used_task_tensors.push_back(task_inn);
      }

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

      op = op_t(move_t {
        .src = {src, current_tensors.at(task_inn)},
        .dst = {src, current_tensors.at(id)      },
        .size = size
      });
    } else if(node.op.is_partialize()) {
      auto const& partialize = node.op.get_partialize();

      auto const& [_0, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
      auto [task_inn, touch] = partialize.get_touch(unit_id, touch_id);

      vector<int> inn_deps = task_to_mem(task_inn);
      deps.insert(inn_deps.begin(), inn_deps.end());

      used_task_tensors.push_back(task_inn);

      mem_t out_mem {
        .offset = current_tensors.at(id),
        .size = node.op.tensor_size(),
      };
      mem_t inn_mem {
        .offset = current_tensors.at(task_inn),
        .size = taskgraph.nodes[task_inn].op.tensor_size()
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

  return {input_to_mem, save_to_mem, memgraph};
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
      continue;
    }
    if(node.op.is_partialize()) {
      // Every partialize touch should be accounted for
      // from the corresponding input. The idea is
      // if input A becomes ready, immediately
      // increment do touch op B += A.
      continue;
    }

    // this is an apply or move node, but the
    // distinction doesn't matter here
    ret.emplace_back(_which_node_t { .task_id = id });

    // Now that this id is now available, so add any touches
    // from id.
    for(auto const& out: node.outs) {
      auto const& out_node = taskgraph.nodes[out];
      if(out_node.op.is_partialize()) {
        auto which_touches = get_which_touches_from(taskgraph, out);
        for(auto const& [inn, w]: which_touches) {
          if(inn == id) {
            ret.emplace_back(w);
          }
        }
      }
    }
  }

  return ret;
}

allocator_t::allocator_t(uint64_t memsize)
{
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

  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter) {
    if(iter->available()) {
      iter_t ret = iter;
      uint64_t sz = 0;
      for(; iter != blocks.end() && iter->available(); ++iter) {
        sz += iter->size();
        if(sz >= size) {
          return optional<return_t>({ret, iter + 1, sz});
        }
      }
    }
  }

  return optional<return_t>();
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

int memgraph_t::insert(memgraph_t::op_t op, set<int> const& deps) {
  nodes.push_back(node_t {
    .op = op,
    .inns = deps,
    .outs = {}
  });

  int ret = nodes.size() - 1;

  for(auto const& inn: deps) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

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

vector<uint64_t> memgraph_t::mem_sizes() const {
  vector<uint64_t> ret(num_compute_locs, 0);
  for(auto const& node: nodes) {
    for(auto const& memloc: node.op.get_memlocs()) {
      ret[memloc.loc] = std::max(ret[memloc.loc], memloc.offset + memloc.size);
    }
  }
  return ret;
}

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

