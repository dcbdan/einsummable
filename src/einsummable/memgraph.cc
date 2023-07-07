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

allocator_settings_t allocator_settings_t::default_settings()
{
  return allocator_settings_t {
    .strat = allocator_strat_t::lowest_dependency,
    .alignment_power = 0
  };
}

memgraph_t::memgraph_t(
  int nl, int nc, vector<int> const& cs)
  : num_compute_locs(nl), num_cache_locs(nc), cache_locs(cs)
{}

void memgraph_t::print_graphviz(std::ostream& out) const {
  using std::endl;

  vector<string> colors{
    "#61B292",
    "#AED09E",
    "#F1E8A7",
    "#A8896C",
    "#A8D8EA",
    "#AA96DA",
    "#FCBAD3",
    "#FFFFD2"
  };

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
      //label = "input " + write_with_ss(id);
      if(input.loc < colors.size()) {
        color = colors[input.loc];
      }
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
      //label = "apply " + write_with_ss(id);
      if(apply.loc < colors.size()) {
        color = colors[apply.loc];
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
      //label = "move " + write_with_ss(id);
      color = "lightgray";
    } else if(op.is_evict()) {
      evict_t const& evict = op.get_evict();
      //label = "evict@" +
        write_with_ss(memloc_t { evict.offset, evict.size, evict.loc }) +
        "->cid" +
        write_with_ss(evict.cache_id);
      if(evict.loc < colors.size()) {
        color = colors[evict.loc];
      }
    } else if(op.is_load()) {
      load_t const& load = op.get_load();
      label = "load@" +
        write_with_ss(memloc_t { load.offset, load.size, load.loc }) +
        "->cid" +
        write_with_ss(load.cache_id);
      if(load.loc < colors.size()) {
        color = colors[load.loc];
      }
    } else if(op.is_partialize()) {
      partialize_t const& par = op.get_partialize();
      string memloc = write_with_ss(
        memloc_t { par.offset, par.size, par.loc });
      label = "partialize@" + memloc;
      //label = "partialize " + write_with_ss(id);
      if(par.loc < colors.size()) {
        color = colors[par.loc];
      }
    } else if(op.is_del()) {
      del_t const& del = op.get_del();
      string memloc = write_with_ss(
        memloc_t { del.offset, del.size, del.loc });
      label = "del@" + memloc;
      //label = "del " + write_with_ss(id);
      if(del.loc < colors.size()) {
        color = colors[del.loc];
      }
    } else {
      throw std::runtime_error("memgraph print should not happen");
    }

    label = write_with_ss(id) + " " + label;

    //auto memlocs = op.get_memlocs();
    //for(int i = 1; i != memlocs.size(); ++i) {
    //  if(memlocs[0].offset == memlocs[i].offset) {
    //    // this argument is donated
    //    color = "green";
    //  }
    //}

    out << tab
      << "n" << id
      << " [style=filled,label=\"" << label << "\"";
    if(color != "") {
      out << ", color=\"" << color << "\"";
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

string memgraph_t::to_wire() const {
  es_proto::MemGraph mg;

  mg.set_num_compute_locs(num_compute_locs);
  mg.set_num_cache_locs(num_cache_locs);
  for(auto const& cl: cache_locs) {
    mg.add_cache_locs(cl);
  }

  for(auto const& node: nodes) {
    es_proto::MemGraphNode* n = mg.add_nodes();

    if(node.op.is_input()) {
      auto const& [loc,offset,size] = node.op.get_input();
      es_proto::MGInput* i = n->mutable_input();
      i->set_loc(loc);
      i->set_offset(offset);
      i->set_size(size);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      auto const& [loc, mems, _, group] = apply;

      es_proto::MGApply* a = n->mutable_apply();

      a->set_loc(loc);

      for(auto const& [offset,size]: mems) {
        a->add_mems_offset(offset);
        a->add_mems_size(size);
      }

      if(apply.is_einsummable()) {
        es_proto::Einsummable* e = a->mutable_einsummable();
        apply.get_einsummable().to_proto(*e);
      } else if(apply.is_touch()) {
        es_proto::Touch* t = a->mutable_touch();
        apply.get_touch().to_proto(*t);
      } else {
        throw std::runtime_error(
          "should not reach: need to impl another apply type");
      }

      a->set_group(group);
    } else if(node.op.is_move()) {
      auto const& [src,dst,size] = node.op.get_move();
      auto const& [src_loc, src_offset] = src;
      auto const& [dst_loc, dst_offset] = dst;
      es_proto::MGMove* m = n->mutable_move();
      m->set_src_loc(src_loc);
      m->set_src_offset(src_offset);
      m->set_dst_loc(dst_loc);
      m->set_dst_offset(dst_offset);
      m->set_size(size);
    } else if(node.op.is_evict()) {
      auto const& [loc,cache_id,offset,size] = node.op.get_evict();
      es_proto::MGEvict* e = n->mutable_evict();
      e->set_loc(loc);
      e->set_cache_id(cache_id);
      e->set_offset(offset);
      e->set_size(size);
    } else if(node.op.is_load()) {
      auto const& [cache_id,loc,offset,size] = node.op.get_load();
      es_proto::MGLoad* l = n->mutable_load();
      l->set_cache_id(cache_id);
      l->set_loc(loc);
      l->set_offset(offset);
      l->set_size(size);
    } else if(node.op.is_partialize()) {
      auto const& [loc,offset,size] = node.op.get_partialize();
      es_proto::MGPartialize* p = n->mutable_partialize();
      p->set_loc(loc);
      p->set_offset(offset);
      p->set_size(size);
    } else if(node.op.is_del()) {
      auto const& [loc,offset,size] = node.op.get_del();
      es_proto::MGDel* d = n->mutable_del();
      d->set_loc(loc);
      d->set_offset(offset);
      d->set_size(size);
    }

    for(auto const& inn: node.inns) {
      n->add_inns(inn);
    }
  }

  string ret;
  mg.SerializeToString(&ret);
  return ret;
}

memgraph_t memgraph_t::from_wire(string const& str) {
  es_proto::MemGraph mg;
  if(!mg.ParseFromString(str)) {
    throw std::runtime_error("could not parse memgraph!");
  }

  auto cls = mg.cache_locs();
  vector<int> cache_locs(cls.begin(), cls.end());

  memgraph_t ret(
    mg.num_compute_locs(),
    mg.num_cache_locs(),
    cache_locs);

  for(int id = 0; id != mg.nodes_size(); ++id) {
    es_proto::MemGraphNode const& n = mg.nodes(id);

    optional<op_t> op;
    if(n.has_input()) {
      auto const& i = n.input();
      op = op_t(input_t {
        .loc = i.loc(),
        .offset = i.offset(),
        .size = i.size() });
    } else if(n.has_apply()) {
      auto const& a = n.apply();

      vector<mem_t> mems;
      int nmem = a.mems_offset_size();
      if(nmem != a.mems_size_size()) {
        throw std::runtime_error("invalid apply_t: mems len must match");
      }
      for(int i = 0; i != nmem; ++i) {
        mems.push_back(mem_t {
          .offset = a.mems_offset(i),
          .size = a.mems_size(i)
        });
      }

      std::variant<einsummable_t, touch_t> aop = [&]()
        -> std::variant<einsummable_t, touch_t>
      {
        if(a.has_einsummable()) {
          return einsummable_t::from_proto(a.einsummable());
        } else if(a.has_touch()) {
          return touch_t::from_proto(a.touch());
        } else {
          throw std::runtime_error("apply op from proto: should not reach");
        }
      }();

      op = op_t(apply_t {
        .loc = a.loc(),
        .mems = mems,
        .op = aop,
        .group = a.group()
      });
    } else if(n.has_move()) {
      auto const& m = n.move();
      op = op_t(move_t {
        .src = { m.src_loc(), m.src_offset() },
        .dst = { m.dst_loc(), m.dst_offset() },
        .size = m.size()
      });
    } else if(n.has_evict()) {
      auto const& e = n.evict();
      op = op_t(evict_t {
        .loc = e.loc(),
        .cache_id = e.cache_id(),
        .offset = e.offset(),
        .size = e.size()
      });
    } else if(n.has_load()) {
      auto const& l = n.load();
      op = op_t(load_t {
        .cache_id = l.cache_id(),
        .loc = l.loc(),
        .offset = l.offset(),
        .size = l.size()
      });
    } else if(n.has_partialize()) {
      auto const& p = n.partialize();
      op = op_t(partialize_t {
        .loc = p.loc(),
        .offset = p.offset(),
        .size = p.size()
      });
    } else if(n.has_del()) {
      auto const& d = n.del();
      op = op_t(del_t {
        .loc = d.loc(),
        .offset = d.offset(),
        .size = d.size()
      });
    }

    set<int> inns;
    for(int i = 0; i != n.inns_size(); ++i) {
      inns.insert(n.inns(i));
    }

    ret.insert(op.value(), inns);
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

  for(auto const& inn: inns) {
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
  } else if(is_partialize()) {
    check_partialize();
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
void memgraph_t::op_t::check_evict()      const {}
void memgraph_t::op_t::check_load()       const {}
void memgraph_t::op_t::check_partialize() const {}
void memgraph_t::op_t::check_del()        const {}

vector<memloc_t> memgraph_t::op_t::get_memlocs() const
{
  if(is_input()) {
    auto const& input = get_input();
    return {
      memloc_t { .offset = input.offset, .size = input.size, .loc = input.loc }
    };
  } else if(is_apply()) {
    auto const& apply = get_apply();
    vector<memloc_t> ret;
    for(mem_t const& mem: apply.mems) {
      ret.push_back(mem.as_memloc(apply.loc));
    }
    return ret;
  } else if(is_move()) {
    auto const& move = get_move();
    auto const& [src_loc, src_offset] = move.src;
    auto const& [dst_loc, dst_offset] = move.dst;
    return {
      memloc_t { .offset = src_offset, .size = move.size, .loc = src_loc },
      memloc_t { .offset = dst_offset, .size = move.size, .loc = dst_loc }
    };
  } else if(is_evict()) {
    auto const& evict = get_evict();
    return {
      memloc_t { .offset = evict.offset, .size = evict.size, .loc = evict.loc }
    };
  } else if(is_load()) {
    auto const& load = get_load();
    return {
      memloc_t { .offset = load.offset, .size = load.size, .loc = load.loc }
    };
  } else if(is_partialize()) {
    auto const& par = get_partialize();
    return {
      memloc_t { .offset = par.offset, .size = par.size, .loc = par.loc }
    };
  } else if(is_del()) {
    auto const& del = get_del();
    return {
      memloc_t { .offset = del.offset, .size = del.size, .loc = del.loc }
    };
  } else {
    throw std::runtime_error("get_memlocs should not reach");
  }
}

memloc_t memgraph_t::op_t::get_output_memloc() const
{
  if(is_input()) {
    auto const& input = get_input();
    return memloc_t {
      .offset = input.offset,
      .size = input.size,
      .loc = input.loc
    };
  } else if(is_apply()) {
    auto const& apply = get_apply();
    auto const& out_mem = apply.mems[0];
    return out_mem.as_memloc(apply.loc);
  } else if(is_move()) {
    auto const& move = get_move();
    auto const& [dst_loc, dst_offset] = move.dst;
    return memloc_t {
      .offset = dst_offset,
      .size = move.size,
      .loc = dst_loc
    };
  } else if(is_evict()) {
    throw std::runtime_error("evict has no output mem_t");
  } else if(is_load()) {
    auto const& load = get_load();
    return memloc_t {
      .offset = load.offset,
      .size = load.size,
      .loc = load.loc
    };
  } else if(is_partialize()) {
    auto const& par = get_partialize();
    return memloc_t {
      .offset = par.offset,
      .size = par.size,
      .loc = par.loc
    };
  } else if(is_del()) {
    throw std::runtime_error("del has no output mem_t");
  } else {
    throw std::runtime_error("get_output_memloc should not reach");
  }
}

mem_t memgraph_t::op_t::get_output_mem() const {
  return get_output_memloc().as_mem();
}

bool memgraph_t::op_t::is_local_to(int loc) const {
  if(is_input()) {
    return loc == get_input().loc;
  } else if(is_apply()) {
    return loc == get_apply().loc;
  } else if(is_move()) {
    auto const& move = get_move();
    return loc == move.get_src_loc() || loc == move.get_dst_loc();
  } else if(is_evict()) {
    return loc == get_evict().loc;
  } else if(is_load()) {
    return loc == get_load().loc;
  } else if(is_partialize()) {
    return loc == get_partialize().loc;
  } else if(is_del()) {
    return loc == get_del().loc;
  } else {
    throw std::runtime_error("is_local_to should not reach");
  }
}

bool memgraph_t::apply_t::is_einsummable() const {
  return std::holds_alternative<einsummable_t>(op);
}

bool memgraph_t::apply_t::is_touch() const {
  return std::holds_alternative<touch_t>(op);
}

einsummable_t const&
memgraph_t::apply_t::get_einsummable() const {
  return std::get<einsummable_t>(op);
}

touch_t const&
memgraph_t::apply_t::get_touch() const {
  return std::get<touch_t>(op);
}

dtype_t
memgraph_t::apply_t::out_dtype() const {
  if(is_einsummable()) {
    return get_einsummable().out_dtype();
  }
  if(is_touch()) {
    return get_touch().dtype;
  }
  throw std::runtime_error("should not reach");
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
  vector<int> const& which_cache,
  vector<uint64_t> mem_sizes,
  allocator_settings_t settings)
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

  vector<allocator_t> allocators;
  if(mem_sizes.size() == 0) {
    // set up an allocator for each loc,
    // each with a very large amount of available memory
    allocators = vector<allocator_t>(
      n_compute_locs,
      allocator_t(std::numeric_limits<uint64_t>::max(), settings));
  } else {
    if(mem_sizes.size() != n_compute_locs) {
      throw std::runtime_error("must have mem sizes for each device");
    }
    allocators.reserve(mem_sizes.size());
    for(auto const& m: mem_sizes) {
      allocators.emplace_back(m, settings);
    }
  }

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

  // Collect the input to memory and save to memory
  map<int, mem_t> input_to_mem;
  map<int, mem_t> save_to_mem;
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];

    optional<mem_t> mem;

    // Regardless of whether or not this is a save node,
    // call task_to_mem for all partializes so that
    // dummy partialize ops are correctly inserted
    // (any node that is not used and not a save
    //  really should not be in the taskgraph, but this
    //  function is not worrying about that)
    if(node.op.is_partialize()) {
      int mem_id = state.task_to_mem(id);
      mem = state.memgraph.nodes[mem_id].op.get_output_mem();
    }

    if(node.op.is_input()) {
      int mem_id = state.task_to_mem(id);
      mem = state.memgraph.nodes[mem_id].op.get_output_mem();
      input_to_mem.insert({id, mem.value()});
    }

    if(node.is_save) {
      if(!mem) {
        int mem_id = state.task_to_mem(id);
        mem = state.memgraph.nodes[mem_id].op.get_output_mem();
      }
      save_to_mem.insert({id, mem.value()});
    }
  }

  return {
    input_to_mem,
    save_to_mem,
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
  remaining_usage_counts = vector<int>(taskgraph.nodes.size(), 0);

  // We may have an einsummable y = x + x. In this case,
  // x gets used once by y.
  //
  // We may also have  a partialize
  //   y = touch from the left  side of x
  //   y = touch from the right side of x
  // In this case, x gets used twice in the formation of y,
  // once for each touch.

  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_partialize()) {
      auto const& partialize = node.op.get_partialize();
      // each touch incurs a usage, even if there are multiple touches
      // from the sasme input
      for(auto const& [inn,_]: partialize.as_touches_from_flat()) {
        remaining_usage_counts[inn] += 1;
      }
    } else {
      set<int> inns = node.op.inputs();
      // for input nodes, inns is empty
      // for move nodes, there is only one input
      // for apply nodes, we can't double count like x in y = x + x,
      //   so inns being a set is desired
      for(auto const& inn: inns) {
        remaining_usage_counts[inn] += 1;
      }
    }
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
      int loc = node.op.out_loc();
      uint64_t sz = node.op.out_size();
      auto [offset, deps] = allocators[loc].allocate(sz);
      if(deps.size() != 0) {
        throw std::runtime_error("The alligator is broken");
      }

      mem_t mem {
        .offset = offset,
        .size = sz
      };

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

  set<int> used_task_tensors;
  set<int> deps;
  optional<op_t> op;

  if(node.op.is_apply()) {
    auto const& [loc, inns, es] = node.op.get_apply();

    vector<mem_t> mems(1 + inns.size());

    auto inn_dtypes = es.inn_dtypes();
    auto inn_shapes = es.inn_shapes();
    for(int i = 0; i != inns.size(); ++i) {
      int const& task_inn = inns[i];
      auto sz = dtype_size(inn_dtypes[i]) * product(inn_shapes[i]);
      mems[i+1] = mem_t {
        .offset = current_tensors.at(task_inn),
        .size = sz
      };

      deps.insert(task_to_mem(task_inn));

      used_task_tensors.insert(task_inn);
    }

    uint64_t out_offset = get_output_alloc_if_necc(id, deps);

    // The reason mems[0] is being set last is because
    // get_output may invalidate current_tensors at
    // the input nodes
    mems[0] = mem_t {
      .offset = out_offset,
      .size = node.op.out_size()
    };

    op = op_t(apply_t {
      .loc = loc,
      .mems = mems,
      .op = es,
      .group = -1
    });
  } else if(node.op.is_move()) {
    auto const& [src,dst,task_inn,size] = node.op.get_move();

    deps.insert(task_to_mem(task_inn));

    used_task_tensors.insert(task_inn);

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

    deps.insert(task_to_mem(task_inn));

    used_task_tensors.insert(task_inn);

    mem_t inn_mem {
      .offset = current_tensors.at(task_inn),
      .size = taskgraph.nodes[task_inn].op.out_size()
    };
    mem_t out_mem {
      .offset = get_output_alloc_if_necc(id, deps),
      .size = node.op.out_size(),
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

int memgraph_make_state_t::task_to_mem(int task_id)
{
  auto const& node = taskgraph.nodes[task_id];
  if(node.op.is_partialize()) {
    // There are two cases of partialize nodes:
    //   1. there is only one touch to the partialize
    //   2. there is more than one touch to the partialize
    // For case 1, the task_id is just the singleton touch input and no
    //             dummy partialize node is inserted
    // For case 2, task_id may already exist in task_node and the the corresponding
    //             mem node is a dummy partialize node. In this case, if the dummy
    //             node doesn't exist, create it.
    auto iter = task_node_to_mem.find(task_id);
    if(iter == task_node_to_mem.end()) {
      set<int> deps;
      auto const which_touches =
        get_which_touches_from(taskgraph, task_id);
      for(auto const& [_, which_touch]: which_touches) {
        deps.insert(task_touch_to_mem.at(which_touch));
      }

      if(deps.size() == 1) {
        // case 1: this is a singleton partialize
        int const& ret = *deps.begin();
        task_node_to_mem.insert({task_id, ret});
        return ret;
      }

      // case 2: create a dummy partialize node
      partialize_t op;
      {
        auto const& an_touch_node = memgraph.nodes[*deps.begin()];
        auto const& [loc, mems, _0, _1] = an_touch_node.op.get_apply();
        op.loc    = loc;
        op.offset = mems[0].offset;
        op.size   = mems[0].size;
      }

      int ret = memgraph.insert(op, deps);
      task_node_to_mem.insert({task_id, ret});
      return ret;
    } else {
      return iter->second;
    }
  } else if(node.op.is_input() || node.op.is_apply() || node.op.is_move()) {
    return task_node_to_mem.at(task_id);
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
      // can't be deleted since this guy was donated to where it was used
      donated.erase(task_id);
      return;
    }

    auto const& node = taskgraph.nodes[task_id];

    if(node.is_save) {
      // can't be deleted since it is marked for saving
      return;
    }

    int loc = node.op.out_loc();
    uint64_t offset = current_tensors.at(task_id);
    del_t del {
      .loc = loc,
      .offset = offset,
      .size = node.op.out_size()
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
  output_mem.size = node.op.out_size();
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
  // TODO: If the node is a partialize that only has
  //       one input, and that input is the same size as the output,
  //       and it's not a save, and it has a singleton usage,
  //       the input can be donated.
  //
  //       If the node is a partial reduction,
  //       is it even clear which one comes first?

  if(!did_get_donation) {
    int loc = node.op.out_loc();
    auto [offset_, ds] = allocators[loc].allocate(output_mem.size);
    output_mem.offset = offset_;
    deps.insert(ds.begin(), ds.end());
  }

  current_tensors.insert({task_id, output_mem.offset});

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

allocator_t::allocator_t(uint64_t memsize, allocator_settings_t s)
  : strat(s.strat), alignment_power(s.alignment_power)
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
allocator_t::find_first_available(uint64_t size) {
  using return_t = tuple<iter_t, iter_t, uint64_t>;

  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter) {
    if(iter->available()) {
      iter_t ret = iter;
      uint64_t sz  = 0;
      uint64_t rem = align_to_power_of_two(iter->beg, alignment_power) - iter->beg;
      for(; iter != blocks.end() && iter->available(); ++iter) {
        sz += iter->size();
        if(rem != 0 && sz > rem) {
          rem = 0;
          sz -= rem;
        }
        if(rem == 0 && sz >= size) {
          return optional<return_t>({ret, iter + 1, sz});
        }
      }
    }
  }

  return std::nullopt;
}

optional<tuple<allocator_t::iter_t, allocator_t::iter_t, uint64_t>>
allocator_t::find_lowest_dependency_available(uint64_t size) {
  using return_t = tuple<iter_t, iter_t, uint64_t>;
  optional<return_t> return_block;
  int min_dep = std::numeric_limits<int>::max();
  for(iter_t iter = blocks.begin(); iter != blocks.end(); ++iter) {
    if(iter->available()) {
      iter_t ret = iter;
      uint64_t sz = 0;
      uint64_t rem = align_to_power_of_two(iter->beg, alignment_power) - iter->beg;
      int inner_max_dep = -1;
      for(iter_t inner_iter = iter;
          inner_iter != blocks.end() && inner_iter->available();
          ++inner_iter)
      {
        inner_max_dep = std::max(inner_max_dep,inner_iter->dep.value());
        sz += inner_iter->size();
        if(rem != 0 && sz > rem) {
          rem = 0;
          sz -= rem;
        }
        if(rem == 0 && sz >= size && inner_max_dep <= min_dep) {
          min_dep = inner_max_dep;
          return_block = {ret, inner_iter + 1, sz};
          break;
        }
      }
    }
  }
  return return_block;
}

optional< tuple<uint64_t, vector<int>> >
allocator_t::try_to_allocate(uint64_t size_without_rem)
{
  using return_t = tuple<uint64_t, vector<int>>;

  optional<tuple<iter_t, iter_t, uint64_t>> maybe_info;
  if(strat == allocator_strat_t::lowest_dependency) {
    maybe_info = find_lowest_dependency_available(size_without_rem);
  } else if(strat == allocator_strat_t::first) {
    maybe_info = find_first_available(size_without_rem);
  } else {
    throw std::runtime_error("should not reach");
  }
  if(maybe_info) {
    auto const& [beg,end,sz] = maybe_info.value();

    // collect the output information
    uint64_t offset = beg->beg;
    uint64_t aligned_offset = align_to_power_of_two(beg->beg, alignment_power);

    uint64_t size = size_without_rem + (aligned_offset - offset);

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
    return optional<return_t>({aligned_offset, deps});
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
  auto iter = binary_search_find(blocks.begin(), blocks.end(),
    [&offset](block_t const& blk) {
      return blk.beg <= offset;
    }
  );

  if(iter == blocks.end()) {
    throw std::runtime_error("did not find a block");
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


