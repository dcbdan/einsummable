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

mem_t const& memsto_t::get_mem() const {
  if(!_is_mem) {
    throw std::runtime_error("cannot get mem");
  }
  return info.mem;
}

int const& memsto_t::get_sto() const {
  if(_is_mem) {
    throw std::runtime_error("cannot get sto");
  }
  return info.sto_id;
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
  : num_compute_locs(nl), num_storage_locs(nc), storage_locs(cs)
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
    if(op.is_inputmem()) {
      memloc_t input = op.get_inputmem().as_memloc();
      string memloc = write_with_ss(input);
      label = "inputmem@" + memloc;
      if(input.loc < colors.size()) {
        color = colors[input.loc];
      }
    } else if(op.is_inputsto()) {
      auto const& input = op.get_inputsto();
      string label = "inputsto@";
      if(input.loc < colors.size()) {
        color = colors[input.loc];
      }
      //label = "input " + write_with_ss(id);
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
      auto const& [memloc, stoloc] = node.op.get_evict();
      label = "evict@" +
        write_with_ss(memloc) +
        "->storage_id" +
        write_with_ss(stoloc.id);
      if(memloc.loc < colors.size()) {
        color = colors[memloc.loc];
      }
    } else if(op.is_load()) {
      auto const& [stoloc, memloc] = node.op.get_load();
      label = string("load@") +
        "storage_id" + write_with_ss(stoloc.id) + "->" +
        write_with_ss(memloc);
      if(memloc.loc < colors.size()) {
        color = colors[memloc.loc];
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
    } else if(op.is_alloc()) {
      alloc_t const& alloc = op.get_alloc();
      string memloc = write_with_ss(alloc.as_memloc());
      label = "alloc@" + memloc;
      //label = "alloc " + write_with_ss(id);
      if(alloc.loc < colors.size()) {
        color = colors[alloc.loc];
      }
    } else if(op.is_del()) {
      del_t const& del = op.get_del();
      string memloc = write_with_ss(del.as_memloc());
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
  mg.set_num_storage_locs(num_storage_locs);
  for(auto const& cl: storage_locs) {
    mg.add_storage_locs(cl);
  }

  for(auto const& node: nodes) {
    es_proto::MemGraphNode* n = mg.add_nodes();

    if(node.op.is_inputmem()) {
      auto const& input = node.op.get_inputmem();
      es_proto::MGInputMem* i = n->mutable_inputmem();
      i->set_loc(input.loc);
      i->set_offset(input.offset);
      i->set_size(input.size);
    } else if(node.op.is_inputsto()) {
      auto const& input = node.op.get_inputsto();
      es_proto::MGInputSto* i = n->mutable_inputsto();
      i->set_loc(input.loc);
      i->set_storage_loc(input.storage_loc);
      i->set_storage_id(input.storage_id);
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
      auto const& [memloc, stoloc] = node.op.get_evict();
      es_proto::MGEvict* e = n->mutable_evict();
      e->set_storage_loc(stoloc.loc);
      e->set_storage_id(stoloc.id);
      e->set_loc(memloc.loc);
      e->set_offset(memloc.offset);
      e->set_size(memloc.size);
    } else if(node.op.is_load()) {
      auto const& [stoloc, memloc] = node.op.get_load();
      es_proto::MGLoad* l = n->mutable_load();
      l->set_storage_loc(stoloc.loc);
      l->set_storage_id(stoloc.id);
      l->set_loc(memloc.loc);
      l->set_offset(memloc.offset);
      l->set_size(memloc.size);
    } else if(node.op.is_partialize()) {
      auto const& [loc,offset,size] = node.op.get_partialize();
      es_proto::MGPartialize* p = n->mutable_partialize();
      p->set_loc(loc);
      p->set_offset(offset);
      p->set_size(size);
    } else if(node.op.is_alloc()) {
      auto const& [loc,offset,size] = node.op.get_alloc();
      es_proto::MGAlloc* a = n->mutable_alloc();
      a->set_loc(loc);
      a->set_offset(offset);
      a->set_size(size);
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

  auto cls = mg.storage_locs();
  vector<int> storage_locs(cls.begin(), cls.end());

  memgraph_t ret(
    mg.num_compute_locs(),
    mg.num_storage_locs(),
    storage_locs);

  for(int id = 0; id != mg.nodes_size(); ++id) {
    es_proto::MemGraphNode const& n = mg.nodes(id);

    optional<op_t> op;
    if(n.has_inputmem()) {
      auto const& i = n.inputmem();
      op = op_t(inputmem_t {
        .loc = i.loc(),
        .offset = i.offset(),
        .size = i.size()
      });
    } else if(n.has_inputsto()) {
      auto const& i = n.inputsto();
      op = op_t(inputsto_t {
        .loc = i.loc(),
        .storage_loc = i.storage_loc(),
        .storage_id = i.storage_id()
      });
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
        .src = memloc_t {
          .offset = e.offset(),
          .size = e.size(),
          .loc = e.loc()
        },
        .dst = stoloc_t {
          .loc = e.storage_loc(),
          .id = e.storage_id()
        }
      });
    } else if(n.has_load()) {
      auto const& l = n.load();
      op = op_t(load_t {
        .src = stoloc_t {
          .loc = l.storage_loc(),
          .id = l.storage_id()
        },
        .dst = memloc_t {
          .offset = l.offset(),
          .size = l.size(),
          .loc = l.loc()
        }
      });
    } else if(n.has_partialize()) {
      auto const& p = n.partialize();
      op = op_t(partialize_t {
        .loc = p.loc(),
        .offset = p.offset(),
        .size = p.size()
      });
    } else if(n.has_alloc()) {
      auto const& a = n.alloc();
      op = op_t(alloc_t {
        .loc = a.loc(),
        .offset = a.offset(),
        .size = a.size()
      });
    } else if(n.has_del()) {
      auto const& d = n.del();
      op = op_t(del_t {
        .loc = d.loc(),
        .offset = d.offset(),
        .size = d.size()
      });
    } else {
      throw std::runtime_error("proto node op contains something unexpected");
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

memgraph_t::inputmem_t
memgraph_t::inputmem_t::from_memloc(memloc_t const& m) {
  return inputmem_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

memgraph_t::partialize_t
memgraph_t::partialize_t::from_memloc(memloc_t const& m) {
  return partialize_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

memgraph_t::alloc_t
memgraph_t::alloc_t::from_memloc(memloc_t const& m) {
  return alloc_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

memgraph_t::del_t
memgraph_t::del_t::from_memloc(memloc_t const& m) {
  return del_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

void memgraph_t::op_t::check_op() const {
  if(is_inputmem()) {
    check_inputmem();
  } else if(is_inputsto()) {
    check_inputsto();
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
  } else if(is_alloc()) {
    check_alloc();
  } else if(is_del()) {
    check_del();
  } else {
    throw std::runtime_error("should not reach");
  }
}

void memgraph_t::op_t::check_inputmem() const {}
void memgraph_t::op_t::check_inputsto() const {}
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
void memgraph_t::op_t::check_alloc()      const {}
void memgraph_t::op_t::check_del()        const {}

vector<memloc_t> memgraph_t::op_t::get_memlocs() const
{
  if(is_inputmem()) {
    auto const& input = get_inputmem();
    return { input.as_memloc() };
  } else if(is_inputsto()) {
    return {};
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
      evict.src
    };
  } else if(is_load()) {
    auto const& load = get_load();
    return {
      load.dst
    };
  } else if(is_partialize()) {
    auto const& par = get_partialize();
    return {
      par.as_memloc()
    };
  } else if(is_alloc()) {
    auto const& alloc = get_alloc();
    return {
      alloc.as_memloc()
    };
  } else if(is_del()) {
    auto const& del = get_del();
    return {
      del.as_memloc()
    };
  } else {
    throw std::runtime_error("get_memlocs should not reach");
  }
}

memstoloc_t memgraph_t::op_t::get_output_memstoloc() const {
  if(is_inputmem()) {
    return get_inputmem().as_memloc();
  } else if(is_inputsto()) {
    return get_inputsto().as_stoloc();
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
    return get_evict().dst;
  } else if(is_load()) {
    return get_load().dst;
  } else if(is_partialize()) {
    return get_partialize().as_memloc();
  } else if(is_alloc()) {
    return get_alloc().as_memloc();
  } else if(is_del()) {
    throw std::runtime_error("del has no output memstoloc_t");
  } else {
    throw std::runtime_error("get_output_memstoloc should not reach");
  }
}

memloc_t memgraph_t::op_t::get_output_memloc() const
{
  if(is_evict()) {
    throw std::runtime_error("evict has no output memory");
  }

  memstoloc_t ret = get_output_memstoloc();
  if(!ret.is_memloc()) {
    throw std::runtime_error("this output is not in memory");
  }
  return ret.get_memloc();
}

mem_t memgraph_t::op_t::get_output_mem() const {
  return get_output_memloc().as_mem();
}

bool memgraph_t::op_t::is_local_to(int loc) const {
  if(is_inputmem()) {
    return loc == get_inputmem().loc;
  } else if(is_inputsto()) {
    return loc == get_inputsto().loc;
  } else if(is_apply()) {
    return loc == get_apply().loc;
  } else if(is_move()) {
    auto const& move = get_move();
    return loc == move.get_src_loc() || loc == move.get_dst_loc();
  } else if(is_evict()) {
    return loc == get_evict().src.loc;
  } else if(is_load()) {
    return loc == get_load().dst.loc;
  } else if(is_partialize()) {
    return loc == get_partialize().loc;
  } else if(is_alloc()) {
    return loc == get_alloc().loc;
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
  vector<uint64_t> mem_sizes,
  allocator_settings_t settings)
{
  int const n_compute_locs = taskgraph.num_locs();

  vector<int> which_storage(n_compute_locs);
  std::iota(which_storage.begin(), which_storage.end(), 0);

  auto [inn_to_memdata, save_to_memdata, memgraph] =
    make(taskgraph, which_storage, mem_sizes, {}, settings, false);

  map<int, mem_t> inn_to_mem;
  for(auto const& [tid, memdata]: inn_to_memdata) {
    inn_to_mem.insert({tid, memdata.get_memloc().as_mem()});
  }

  map<int, mem_t> save_to_mem;
  for(auto const& [tid, memdata]: save_to_memdata) {
    save_to_mem.insert({tid, memdata.get_memloc().as_mem()});
  }

  return {inn_to_mem, save_to_mem, memgraph};
}

tuple<
  map<int, memstoloc_t>, // input -> data
  map<int, memstoloc_t>, // save  -> data
  memgraph_t>
memgraph_t::make(
  taskgraph_t const& taskgraph,
  vector<int> const& which_storage,
  vector<uint64_t> mem_sizes,
  map<int, memstoloc_t> input_tid_to_data,
  allocator_settings_t settings,
  bool use_storage)
{
  int const n_compute_locs = taskgraph.num_locs();
  if(which_storage.size() != n_compute_locs) {
    throw std::runtime_error("incorrect which storage length: memgraph_t::make");
  }

  int n_storage_locs = 0;
  for(int const& storage_loc: which_storage) {
    if(storage_loc < 0) {
      throw std::runtime_error("invalid storage loc");
    }
    n_storage_locs = std::max(n_storage_locs, storage_loc + 1);
  }

  for(int i = 0; i != n_storage_locs; ++i) {
    auto iter = std::find(which_storage.begin(), which_storage.end(), i);
    if(iter == which_storage.end()) {
      throw std::runtime_error("storage locs must be 0, ..., n_storage_locs-1; no missing");
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
    which_storage,
    allocators,
    input_tid_to_data,
    n_compute_locs, n_storage_locs,
    use_storage);

  if(!use_storage) {
    // Without storage, it makes more sense to allocate all the inputs
    // before proceeding
    for(int id = 0; id != taskgraph.nodes.size(); ++id) {
      auto const& node = taskgraph.nodes[id];
      if(node.op.is_input()) {
        if(node.outs.size() == 0 && !node.is_save) {
          throw std::runtime_error(
            "This is goofy: an input to memgraph is not used or saved."
            " Call this again after pruning inputs that don't get used"
            " or saved."
          );
        }

        // It could be the case that the used initialized the input
        if(!state.input_has_been_initialized(id)) {
          state.initialize_input(id);
        }
      }
    }
  }

  for(auto which_op: order_taskgraph(taskgraph))
  {
    state.add_to_memgraph(which_op);
  }

  map<int, memstoloc_t> save_to_data;
  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.is_save) {
      int memid = state.task_tensor_to_mem_node.at(id);
      auto const& memnode = state.memgraph.nodes[memid];
      save_to_data.insert({ id, memnode.op.get_output_memstoloc() });
    }
  }

  return {
    input_tid_to_data,
    save_to_data,
    state.memgraph
  };
}

memgraph_make_state_t::memgraph_make_state_t(
  taskgraph_t const& tg,
  vector<int> const& which_storage,
  vector<allocator_t> const& as,
  map<int, memstoloc_t>& ittd,
  int num_compute,
  int num_storage,
  bool use_storage)
  : taskgraph(tg),
    memgraph(num_compute, num_storage, which_storage),
    allocators(as),
    _group(0),
    use_storage(use_storage),
    input_tid_to_data(ittd)
{
  _sto_id = 0;
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

void memgraph_make_state_t::initialize_input(int inn){
  auto const& node = taskgraph.nodes[inn];
  int loc = node.op.out_loc();
  uint64_t size = node.op.out_size();

  auto maybe = allocators[loc].try_to_allocate_without_deps(size);
  if (maybe) {
    // If we are able to allocate without deps on memory, insert a inputmem_t
    auto const& offset = maybe.value();

    inputmem_t input_mem = {.loc = loc, .offset = offset, .size = size };
    input_tid_to_data[inn] = memstoloc_t(input_mem.as_memloc());

    op_t input_op = op_t(input_mem);
    int memid = memgraph.insert(input_op, {});
    task_tensor_to_mem_node.insert_or_assign(inn, memid);
  } else {
    // If we are not able to allocate on memory, insert into inputsto_t

    if(!use_storage) {
      throw std::runtime_error("no more memory to initialize inputs; use storage?");
    }

    inputsto_t input_sto = {
      .loc = loc,
      .storage_loc = memgraph.storage_locs[loc],
      .storage_id = _sto_id++
    };
    input_tid_to_data[inn] = memstoloc_t(input_sto.as_stoloc());

    op_t input_op = op_t(input_sto);
    int memid = memgraph.insert(input_op, {});
    task_tensor_to_mem_node.insert_or_assign(inn, memid);
  }
}

bool memgraph_make_state_t::input_has_been_initialized(int inn){
  return input_tid_to_data.find(inn) != input_tid_to_data.end();
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

  // loop through all inns so we make sure all inns are
  // ready for this node before starts
  for (int const& inn: node.op.inputs()){
    auto const& inn_node = taskgraph.nodes[inn];
    if(inn_node.op.is_input() && !input_has_been_initialized(inn)) {
      initialize_input(inn);
    }
  }

  set<int> used_task_tensors;
  set<int> deps;
  optional<op_t> op;

  optional<int> touch_output_memid;

  // TODO: this method should support tensor donation

  if(node.op.is_apply()) {
    auto const& [loc, inns, es] = node.op.get_apply();

    vector<int> out_then_inns(inns.size() + 1);
    out_then_inns[0] = id;
    std::copy(inns.begin(), inns.end(), out_then_inns.begin()+1);

    auto [vector_deps, mems] = vector_unzip(
      get_tensors_in_memory(out_then_inns));
    deps = set<int>(vector_deps.begin(), vector_deps.end());

    for(auto const& task_inn: inns) {
      used_task_tensors.insert(task_inn);
    }

    op = op_t(apply_t {
      .loc = loc,
      .mems = mems,
      .op = es,
      .group = -1
    });
  } else if(node.op.is_move()) {
    auto const& [src,dst,task_inn,size] = node.op.get_move();

    auto info = get_tensors_in_memory({task_inn, id});
    auto const& [src_mem_id, src_mem] = info[0];
    auto const& [dst_mem_id, dst_mem] = info[1];

    deps.insert(src_mem_id);
    deps.insert(dst_mem_id);

    used_task_tensors.insert(task_inn);

    op = op_t(move_t {
      .src = {src, src_mem.offset},
      .dst = {dst, dst_mem.offset},
      .size = size
    });
  } else if(node.op.is_partialize()) {
    auto const& partialize = node.op.get_partialize();

    auto const& [_0, unit_id, touch_id] = std::get<_which_touch_t>(which_op);
    auto [task_inn, touch] = partialize.get_touch(unit_id, touch_id);

    auto info = get_tensors_in_memory({task_inn, id});
    auto const& [inn_mem_id, inn_mem] = info[0];
    auto const& [out_mem_id, out_mem] = info[1];

    touch_output_memid = out_mem_id;

    deps.insert(inn_mem_id);
    deps.insert(out_mem_id);

    used_task_tensors.insert(task_inn);

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
    task_node_to_mem_node.insert({
      std::get<_which_node_t>(which_op),
      new_memid
    });

    // For apply and move nodes, insert the newly created
    // memid into the tensor mapping
    task_tensor_to_mem_node.insert_or_assign(id, new_memid);
  } else {
    // This is a touch in a partialize node.

    task_touch_to_mem_node.insert({
      std::get<_which_touch_t>(which_op),
      new_memid
    });

    int num_touches_in_partialize =
      node.op.get_partialize().get_num_touches();

    auto& in_progress = partializes_in_progress[id];
    in_progress.push_back(new_memid);

    bool is_last_touch = (in_progress.size() == num_touches_in_partialize);

    if(is_last_touch) {
      // This partialize is complete
      if(num_touches_in_partialize == 1) {
        // then insert the newly created memid into the tensor mapping
        task_tensor_to_mem_node.insert_or_assign(id, new_memid);
      } else {
        // create a partialize node that depends on everything in in_progress
        // and insert that into the tensor mapping
        auto new_memloc = memgraph.nodes[new_memid].op.get_output_memloc();
        partialize_t new_partialize = partialize_t::from_memloc(new_memloc);
        int partialize_memid = memgraph.insert(
          op_t(new_partialize),
          set<int>(in_progress.begin(), in_progress.end()));
        task_tensor_to_mem_node.insert_or_assign(id, partialize_memid);
      }
      partializes_in_progress.erase(id);
    } else {
      // This partialize is still in progress. Insert the allocated
      // output memid.
      task_tensor_to_mem_node.insert_or_assign(id, touch_output_memid.value());
    }
  }

  // Now try to delete some tensors
  for(auto const& used_task_id: used_task_tensors) {
    register_usage(used_task_id);
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

vector<tuple<int, mem_t>>
memgraph_make_state_t::get_tensors_in_memory(
  vector<int> const& task_ids)
{
  vector<tuple<int, mem_t>> ret;
  for(auto const& tid: task_ids) {
    auto iter = task_tensor_to_mem_node.find(tid);
    if(iter != task_tensor_to_mem_node.end()) {
      int const& memid = iter->second;
      auto maybe_mem = memgraph.nodes[memid].op.get_output_memstoloc();
      if(maybe_mem.is_memloc()) {
        ret.emplace_back(memid, maybe_mem.get_memloc().as_mem());
      } else {
        throw std::runtime_error("not implemented: loading from storage");
      }
    } else {
      auto const& node = taskgraph.nodes[tid];
      if(node.op.is_input()) {
        throw std::runtime_error("The input node SHOULD BE in task_tensor_to_mem!");
      }
      int loc = node.op.out_loc();
      auto size = node.op.out_size();
      auto maybe = allocators[loc].try_to_allocate(size);
      if(maybe) {
        auto const& [offset, vector_deps] = maybe.value();
        alloc_t alloc {
          .loc = loc,
          .offset = offset,
          .size = size
        };
        set<int> deps(vector_deps.begin(), vector_deps.end());
        int new_memid = memgraph.insert(op_t(alloc), deps);
        ret.emplace_back(
          new_memid,
          mem_t { .offset = offset, .size = size });

        // make sure to add the memid into task_tensor_to_mem_node
        // so we don't keep allocating this memory!
        task_tensor_to_mem_node.insert({tid, new_memid});
      } else if(use_storage) {
        throw std::runtime_error(
          "not implemented: evicting things to make room for allocation");
      } else {
        throw std::runtime_error("storage not enabled and ran out of memory");
      }
    }
  }
  return ret;
}

void memgraph_make_state_t::register_usage(int task_id)
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

    int completing_memnode = task_tensor_to_mem_node.at(task_id);
    auto const& memnode = memgraph.nodes[completing_memnode];
    memstoloc_t data = memnode.op.get_output_memstoloc();

    if(data.is_stoloc()) {
      // TODO: this could get triggered if we're accomplishing a
      //       taskgraph move via a memgraph load
      throw std::runtime_error("this tensor should have been deleted, not evicted!");
    }

    memloc_t memloc = data.get_memloc();
    del_t del = del_t::from_memloc(memloc);

    // The delete of task_id depends on
    // 1. the output applys that used task_id and
    // 2. the touch operations that used task_id
    set<int> del_deps;
    for(int task_out: node.outs) {
      auto const& out_node = taskgraph.nodes[task_out];
      if(out_node.op.is_apply() || out_node.op.is_move()) {
        _which_node_t _task_out { task_out };
        del_deps.insert(task_node_to_mem_node.at(_task_out));
      } else if(out_node.op.is_partialize()) {
        auto const whiches = get_which_touches_from_to(
          taskgraph,
          task_out,
          task_id);
        for(auto const& which: whiches) {
          del_deps.insert(task_touch_to_mem_node.at(which));
        }
      }
    }

    int del_id = memgraph.insert(op_t(del), del_deps);

    allocators[memloc.loc].free(memloc.offset, del_id);

    task_tensor_to_mem_node.erase(task_id);
  }
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

optional<uint64_t>
allocator_t::try_to_allocate_without_deps(uint64_t size) {
  auto const& maybe = try_to_allocate_impl(size, true);
  if (maybe) {
    auto const& [offset, d] = maybe.value();
    return offset;
  } else {
    return optional<uint64_t>();
  }
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

optional<tuple<uint64_t, vector<int>>>
allocator_t::try_to_allocate_impl(uint64_t size_without_rem, bool no_deps)
{
  using return_t = tuple<uint64_t, vector<int>>;

  optional<tuple<iter_t, iter_t, uint64_t>> maybe_info;
  if (no_deps) {
    maybe_info = find_lowest_dependency_available(size_without_rem);
  } else {
    if(strat == allocator_strat_t::lowest_dependency) {
      maybe_info = find_lowest_dependency_available(size_without_rem);
    } else if(strat == allocator_strat_t::first) {
      maybe_info = find_first_available(size_without_rem);
    } else {
      throw std::runtime_error("should not reach");
    }
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

    if(no_deps && deps.size() > 0) {
      // if deps aren't allowed and there would be some, then fail here
      return optional<return_t>();
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

optional< tuple<uint64_t, vector<int>> >
allocator_t::try_to_allocate(uint64_t size_without_rem)
{
 return try_to_allocate_impl(size_without_rem, false);
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

bool operator==(_which_node_t const& lhs, _which_node_t const& rhs)
{
  return lhs.task_id == rhs.task_id;
}
bool operator<(_which_node_t const& lhs, _which_node_t const& rhs)
{
  return lhs.task_id < rhs.task_id;
}

bool operator==(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_eq(lhs, rhs);
}
bool operator<(_which_touch_t const& lhs, _which_touch_t const& rhs)
{
  return three_tuple_lt(lhs, rhs);
}


