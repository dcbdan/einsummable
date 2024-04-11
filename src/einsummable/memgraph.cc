#include "memgraph.h"

memloc_t mem_t::as_memloc(int loc) const {
  return memloc_t{
      .offset = offset,
      .size = size,
      .loc = loc};
}

mem_t mem_t::from_proto(es_proto::Mem const& m) {
  return mem_t{m.offset(), m.size()};
}

void mem_t::to_proto(es_proto::Mem& m) const {
  m.set_offset(offset);
  m.set_size(size);
}

mem_t memloc_t::as_mem() const {
  return mem_t{
      .offset = offset,
      .size = size};
}

memloc_t memloc_t::from_proto(es_proto::MemLoc const& m) {
  return memloc_t{m.offset(), m.size(), m.loc()};
}

void memloc_t::to_proto(es_proto::MemLoc& m) const {
  m.set_offset(offset);
  m.set_size(size);
  m.set_loc(loc);
}

mem_t const& memsto_t::get_mem() const {
  if(!_is_mem)
  {
    throw std::runtime_error("cannot get mem");
  }
  return info.mem;
}

int const& memsto_t::get_sto() const {
  if(_is_mem)
  {
    throw std::runtime_error("cannot get sto");
  }
  return info.sto_id;
}

allocator_settings_t allocator_settings_t::default_settings() {
  return allocator_settings_t{
      .strat = allocator_strat_t::lowest_dependency,
      .alignment_power = 0};
}

allocator_settings_t allocator_settings_t::gpu_alignment_settings() {
  return allocator_settings_t{
      .strat = allocator_strat_t::lowest_dependency,
      .alignment_power = 4};
}

memgraph_t::memgraph_t(
  int nl, int nc, vector<int> const& cs, bool pe)
  : num_compute_locs(nl), num_storage_locs(nc), storage_locs(cs), prune_edges(pe)
{}

memgraph_t::memgraph_t(
  memgraph_t const& mg)
  : num_compute_locs(mg.num_compute_locs),
    num_storage_locs(mg.num_storage_locs),
    storage_locs(mg.storage_locs),
    nodes(mg.nodes),
    all_deps(mg.all_deps)
{}

void memgraph_t::print_graphviz(std::ostream &out) const {
  using std::endl;

  vector<string> colors{
      "#61B292",
      "#AED09E",
      "#F1E8A7",
      "#A8896C",
      "#A8D8EA",
      "#AA96DA",
      "#FCBAD3",
      "#FFFFD2"};

  string tab = "  ";
  out << "digraph {" << endl;

  for(int id = 0; id != nodes.size(); ++id) {
    node_t const& node = nodes[id];
    op_t const& op = node.op;

    string label;
    string color = "";
    if(op.is_inputmem())
    {
      memloc_t input = op.get_inputmem().as_memloc();
      string memloc = write_with_ss(input);
      label = "inputmem@" + memloc;
      if(input.loc < colors.size())
      {
        color = colors[input.loc];
      }
    } else if(op.is_inputsto()) {
      auto const& input = op.get_inputsto();
      // std::cout << "Label is: " + label << std::endl;
      // label = "input " + write_with_ss(id);
      label = "inputsto@sto_id=" + write_with_ss(input.storage_id);
    } else if(op.is_constant()) {
      memloc_t const& data = op.get_constant().as_memloc();
      string data_str = write_with_ss(data);
      label = "constant@" + data_str;
      if(data.loc < colors.size())
      {
        color = colors[data.loc];
      }
    } else if(op.is_apply()) {
      apply_t const& apply = op.get_apply();
      auto const& aop = apply.op;
      string header;
      string aopstr;
      if(std::holds_alternative<einsummable_t>(aop))
      {
        header = "apply";
        auto const& e = std::get<einsummable_t>(aop);
        aopstr = write_with_ss(e) + "," + write_with_ss(e.out_dtype());
      }
      else if(std::holds_alternative<touch_t>(aop))
      {
        header = "touch";
        aopstr = write_with_ss(std::get<touch_t>(aop).castable) +
                 "group" + write_with_ss(apply.group);
      }
      else
      {
        throw std::runtime_error("parint graphviz should not reach");
      }
      label = header + "@loc" + write_with_ss(apply.loc) + "." + aopstr;
      for(mem_t const& mem: apply.mems)
      {
        label += "|" + write_with_ss(mem);
      }
      // label = "apply " + write_with_ss(id);
      if(apply.loc < colors.size())
      {
        color = colors[apply.loc];
      }
    } else if(op.is_move()) {
      move_t const& move = op.get_move();

      auto const& [src_loc, src_offset] = move.src;
      auto const& [dst_loc, dst_offset] = move.dst;
      auto const& size = move.size;

      label = "move@" +
              write_with_ss(memloc_t{src_offset, size, src_loc}) +
              "->" +
              write_with_ss(memloc_t{dst_offset, size, dst_loc});
      // label = "move " + write_with_ss(id);
      color = "pink";
    } else if(op.is_evict()) {
      auto const& [memloc, stoloc] = node.op.get_evict();
      label = "evict@" +
              write_with_ss(memloc) +
              "->sto_id" +
              write_with_ss(stoloc.id);
      if(memloc.loc < colors.size())
      {
        color = "pink"; // colors[memloc.loc];
      }
      color = "pink";
    } else if(op.is_load()) {
      auto const& [stoloc, memloc] = node.op.get_load();
      label = string("load@") +
              "sto_id" + write_with_ss(stoloc.id) + "->" +
              write_with_ss(memloc);
      if(memloc.loc < colors.size())
      {
        color = "pink"; // colors[memloc.loc];
      }
      color = "pink";
    } else if(op.is_partialize()) {
      partialize_t const& par = op.get_partialize();
      string memloc = write_with_ss(
          memloc_t{par.offset, par.size, par.loc});
      label = "partialize@" + memloc;
      // label = "partialize " + write_with_ss(id);
      if(par.loc < colors.size())
      {
        color = colors[par.loc];
      }
    } else if(op.is_alloc()) {
      alloc_t const& alloc = op.get_alloc();
      string memloc = write_with_ss(alloc.as_memloc());
      label = "alloc@" + memloc;
      // label = "alloc " + write_with_ss(id);
      if(alloc.loc < colors.size())
      {
        color = colors[alloc.loc];
      }
    } else if(op.is_del()) {
      del_t const& del = op.get_del();
      string memloc = write_with_ss(del.as_memloc());
      label = "del@" + memloc;
      // label = "del " + write_with_ss(id);
      if(del.loc < colors.size())
      {
        color = colors[del.loc];
      }
    } else {
      throw std::runtime_error("memgraph print should not happen");
    }
    label = write_with_ss(id) + " " + label;

    // for(int const& _id: {0,1,2,3,4,5,6,7,8,9,10,11,12,13}) {
    //   if(id == _id) {
    //     color = "pink";
    //   }
    // }

    // auto memlocs = op.get_memlocs();
    // for(auto const& memloc: memlocs) {
    //   auto const& [offset, size] = memloc.as_mem();
    //   if(interval_intersect(
    //       {offset, offset+size},
    //       {1000000, 2000000}))
    //   {
    //     color = "chocolate2";
    //   }
    // }
    // for(auto const& x: {13,14,97,129,137,147}) {
    //   if(id == x) {
    //     color = "x11purple";
    //   }
    // }
    // for(auto const& x: {60,61}) {
    //   if(id == x) {
    //     color = "linen";
    //   }
    // }

    // auto memlocs = op.get_memlocs();
    // for(int i = 1; i != memlocs.size(); ++i) {
    //   if(memlocs[0].offset == memlocs[i].offset) {
    //     // this argument is donated
    //     color = "green";
    //   }
    // }

    out << tab
        << "n" << id
        << " [style=filled,label=\"" << label << "\"";
    if(color != "") {
      out << ", color=\"" << color << "\"";
    }
    out << "]" << endl;

    for(int const& inn_id: node.inns) {
      out << tab << "n" << inn_id << " -> "
          << "n" << id << endl;
    }
  }
  out << "}" << endl;
}

vector<uint64_t> memgraph_t::mem_sizes() const
{
  vector<uint64_t> ret(num_compute_locs, 0);
  for(auto const& node: nodes)
  {
    for(auto const& memloc : node.op.get_memlocs())
    {
      ret[memloc.loc] = std::max(ret[memloc.loc], memloc.offset + memloc.size);
    }
  }
  return ret;
}

string memgraph_t::to_wire() const
{
  es_proto::MemGraph mg;
  to_proto(mg);
  string ret;
  mg.SerializeToString(&ret);
  return ret;
}

void memgraph_t::to_proto(es_proto::MemGraph &mg) const
{
  mg.set_num_compute_locs(num_compute_locs);
  mg.set_num_storage_locs(num_storage_locs);
  for(auto const& cl: storage_locs) {
    mg.add_storage_locs(cl);
  }

  for(auto const& node: nodes) {
    es_proto::MemGraphNode *n = mg.add_nodes();

    if(node.op.is_inputmem()) {
      auto const& input = node.op.get_inputmem();
      es_proto::MGInputMem *i = n->mutable_inputmem();
      i->set_loc(input.loc);
      i->set_offset(input.offset);
      i->set_size(input.size);
    } else if(node.op.is_inputsto()) {
      auto const& input = node.op.get_inputsto();
      es_proto::MGInputSto *i = n->mutable_inputsto();
      i->set_storage_loc(input.storage_loc);
      i->set_storage_id(input.storage_id);
      i->set_size(input.size);
    } else if(node.op.is_constant()) {
      auto const& constant = node.op.get_constant();
      es_proto::MGConstant *c = n->mutable_constant();
      c->set_loc(constant.loc);
      c->set_offset(constant.offset);
      es_proto::Fill* f = c->mutable_fill();
      constant.fill.to_proto(*f);
    } else if(node.op.is_apply()) {
      auto const& apply = node.op.get_apply();
      auto const& [loc, mems, _, group] = apply;

      es_proto::MGApply *a = n->mutable_apply();

      a->set_loc(loc);

      for(auto const& [offset, size]: mems)
      {
        a->add_mems_offset(offset);
        a->add_mems_size(size);
      }

      if(apply.is_einsummable())
      {
        es_proto::Einsummable *e = a->mutable_einsummable();
        apply.get_einsummable().to_proto(*e);
      }
      else if(apply.is_touch())
      {
        es_proto::Touch *t = a->mutable_touch();
        apply.get_touch().to_proto(*t);
      }
      else
      {
        throw std::runtime_error(
            "should not reach: need to impl another apply type");
      }

      a->set_group(group);
    } else if(node.op.is_move()) {
      auto const& [src, dst, size] = node.op.get_move();
      auto const& [src_loc, src_offset] = src;
      auto const& [dst_loc, dst_offset] = dst;
      es_proto::MGMove *m = n->mutable_move();
      m->set_src_loc(src_loc);
      m->set_src_offset(src_offset);
      m->set_dst_loc(dst_loc);
      m->set_dst_offset(dst_offset);
      m->set_size(size);
    } else if(node.op.is_evict()) {
      auto const& [memloc, stoloc] = node.op.get_evict();
      es_proto::MGEvict *e = n->mutable_evict();
      e->set_storage_loc(stoloc.loc);
      e->set_storage_id(stoloc.id);
      e->set_loc(memloc.loc);
      e->set_offset(memloc.offset);
      e->set_size(memloc.size);
    } else if(node.op.is_load()) {
      auto const& [stoloc, memloc] = node.op.get_load();
      es_proto::MGLoad *l = n->mutable_load();
      l->set_storage_loc(stoloc.loc);
      l->set_storage_id(stoloc.id);
      l->set_loc(memloc.loc);
      l->set_offset(memloc.offset);
      l->set_size(memloc.size);
    } else if(node.op.is_partialize()) {
      auto const& [loc, offset, size] = node.op.get_partialize();
      es_proto::MGPartialize *p = n->mutable_partialize();
      p->set_loc(loc);
      p->set_offset(offset);
      p->set_size(size);
    } else if(node.op.is_alloc()) {
      auto const& [loc, offset, size] = node.op.get_alloc();
      es_proto::MGAlloc *a = n->mutable_alloc();
      a->set_loc(loc);
      a->set_offset(offset);
      a->set_size(size);
    } else if(node.op.is_del()) {
      auto const& [loc, offset, size] = node.op.get_del();
      es_proto::MGDel *d = n->mutable_del();
      d->set_loc(loc);
      d->set_offset(offset);
      d->set_size(size);
    }

    for(auto const& inn: node.inns) {
      n->add_inns(inn);
    }
  }
}

memgraph_t memgraph_t::from_wire(string const& str)
{
  es_proto::MemGraph mg;
  if(!mg.ParseFromString(str)) {
    throw std::runtime_error("could not parse memgraph!");
  }
  return from_proto(mg);
}

memgraph_t memgraph_t::from_proto(es_proto::MemGraph const& mg)
{
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
      op = op_t(inputmem_t{
          .loc = i.loc(),
          .offset = i.offset(),
          .size = i.size()});
    } else if(n.has_inputsto()) {
      auto const& i = n.inputsto();
      op = op_t(inputsto_t{
          .storage_loc = i.storage_loc(),
          .storage_id = i.storage_id(),
          .size = i.size()});
    } else if(n.has_constant()) {
      auto const& c = n.constant();
      auto const& f = c.fill();

      op = op_t(constant_t {
        .loc = c.loc(),
        .offset = c.offset(),
        .fill = fill_t::from_proto(f)
      });
    } else if(n.has_apply()) {
      auto const& a = n.apply();

      vector<mem_t> mems;
      int nmem = a.mems_offset_size();
      if(nmem != a.mems_size_size())
      {
        throw std::runtime_error("invalid apply_t: mems len must match");
      }
      for(int i = 0; i != nmem; ++i)
      {
        mems.push_back(mem_t{
            .offset = a.mems_offset(i),
            .size = a.mems_size(i)});
      }

      std::variant<einsummable_t, touch_t> aop = [&]()
          -> std::variant<einsummable_t, touch_t>
      {
        if(a.has_einsummable())
        {
          return einsummable_t::from_proto(a.einsummable());
        }
        else if(a.has_touch())
        {
          return touch_t::from_proto(a.touch());
        }
        else
        {
          throw std::runtime_error("apply op from proto: should not reach");
        }
      }();

      op = op_t(apply_t{
          .loc = a.loc(),
          .mems = mems,
          .op = aop,
          .group = a.group()});
    } else if(n.has_move()) {
      auto const& m = n.move();
      op = op_t(move_t{
          .src = {m.src_loc(), m.src_offset()},
          .dst = {m.dst_loc(), m.dst_offset()},
          .size = m.size()});
    } else if(n.has_evict()) {
      auto const& e = n.evict();
      op = op_t(evict_t{
          .src = memloc_t{
              .offset = e.offset(),
              .size = e.size(),
              .loc = e.loc()},
          .dst = stoloc_t{.loc = e.storage_loc(), .id = e.storage_id()}});
    } else if(n.has_load()) {
      auto const& l = n.load();
      op = op_t(load_t{
          .src = stoloc_t{
              .loc = l.storage_loc(),
              .id = l.storage_id()},
          .dst = memloc_t{.offset = l.offset(), .size = l.size(), .loc = l.loc()}});
    } else if(n.has_partialize()) {
      auto const& p = n.partialize();
      op = op_t(partialize_t{
          .loc = p.loc(),
          .offset = p.offset(),
          .size = p.size()});
    } else if(n.has_alloc()) {
      auto const& a = n.alloc();
      op = op_t(alloc_t{
          .loc = a.loc(),
          .offset = a.offset(),
          .size = a.size()});
    } else if(n.has_del()) {
      auto const& d = n.del();
      op = op_t(del_t{
          .loc = d.loc(),
          .offset = d.offset(),
          .size = d.size()});
    } else {
      throw std::runtime_error("proto node op contains something unexpected");
    }

    set<int> inns;
    for(int i = 0; i != n.inns_size(); ++i)
    {
      inns.insert(n.inns(i));
    }

    ret.insert(op.value(), inns);
  }

  return ret;
}

vector<int>
memgraph_t::get_locs_from_storage_loc(int sto_loc) const
{
  vector<int> ret;
  for(int loc = 0; loc != storage_locs.size(); ++loc) {
    if(storage_locs[loc] == sto_loc) {
      ret.push_back(loc);
    }
  }
  if(ret.size() == 0) {
    throw std::runtime_error("invalid sto_loc");
  }
  return ret;
}

int memgraph_t::insert(memgraph_t::op_t op, set<int> const& deps)
{
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

  if(prune_edges) {
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
            for(int j = i + 1; j != deps_vec.size(); ++j) {
              int const& jd = deps_vec[j];
              if(depends_on(id, jd))
              {
                unnec.insert(jd);
              }
            }
          }
          inns.insert(id);
        }
      }
    }
  } else {
    inns = deps;
  }

  // std::cout << "Inserting node in the memgraph: " << op.get_name() << std::endl;

  nodes.push_back(node_t{
    .op = op,
    .inns = inns,
    .outs = {}
  });

  int ret = nodes.size() - 1;

  for(auto const& inn: inns) {
    nodes[inn].outs.insert(ret);
  }

  if(prune_edges) {
    all_deps.emplace_back(ret, 0);

    vector<char> &ret_deps = all_deps.back();
    for(int const& inn : inns) {
      ret_deps[inn] = 1;

      vector<char> &inn_deps = all_deps[inn];
      for(int i = 0; i != inn_deps.size(); ++i) {
        ret_deps[i] = std::max(ret_deps[i], inn_deps[i]);
      }
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
  return inputmem_t{
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
memgraph_t::alloc_t::from_memloc(memloc_t const& m)
{
  return alloc_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

memgraph_t::del_t
memgraph_t::del_t::from_memloc(memloc_t const& m)
{
  return del_t {
    .loc = m.loc,
    .offset = m.offset,
    .size = m.size
  };
}

string memgraph_t::op_t::get_name() const
{
  if(is_alloc()) {
    return "alloc";
  } else if(is_apply()) {
    return "apply";
  } else if(is_evict()) {
    return "evict";
  } else if(is_load()) {
    return "load";
  } else if(is_inputsto()) {
    return "inputsto";
  } else if(is_move()) {
    return "move";
  } else if(is_del()) {
    return "del";
  } else if(is_inputmem()) {
    return "inputmem";
  } else if(is_constant()) {
    return "constant";
  } else if(is_partialize()) {
    return "partialize";
  }

  throw std::runtime_error("get_name not implemented for all");
}

void memgraph_t::op_t::check_op() const
{
  if(is_inputmem()) {
    check_inputmem();
  } else if(is_inputsto()) {
    check_inputsto();
  } else if(is_constant()) {
    check_constant();
  } else if(is_apply()) {
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
void memgraph_t::op_t::check_constant() const {}
void memgraph_t::op_t::check_apply()    const {}
void memgraph_t::op_t::check_move() const
{
  move_t const& move = get_move();
  auto const& [src, _0] = move.src;
  auto const& [dst, _1] = move.dst;
  if(src == dst)
  {
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
    return {input.as_memloc()};
  } else if(is_inputsto()) {
    return {};
  } else if(is_constant()) {
    auto const& constant = get_constant();
    return {constant.as_memloc()};
  } else if(is_apply()) {
    auto const& apply = get_apply();
    vector<memloc_t> ret;
    for(mem_t const& mem: apply.mems)
    {
      ret.push_back(mem.as_memloc(apply.loc));
    }
    return ret;
  } else if(is_move()) {
    auto const& move = get_move();
    auto const& [src_loc, src_offset] = move.src;
    auto const& [dst_loc, dst_offset] = move.dst;
    return {
        memloc_t{.offset = src_offset, .size = move.size, .loc = src_loc},
        memloc_t{.offset = dst_offset, .size = move.size, .loc = dst_loc}};
  } else if(is_evict()) {
    auto const& evict = get_evict();
    return {
        evict.src};
  } else if(is_load()) {
    auto const& load = get_load();
    return {
        load.dst};
  } else if(is_partialize()) {
    auto const& par = get_partialize();
    return {
        par.as_memloc()};
  } else if(is_alloc()) {
    auto const& alloc = get_alloc();
    return {
        alloc.as_memloc()};
  } else if(is_del()) {
    auto const& del = get_del();
    return {
        del.as_memloc()};
  } else {
    throw std::runtime_error("get_memlocs should not reach");
  }
}

memstoloc_t memgraph_t::op_t::get_output_memstoloc() const
{
  if(is_inputmem()) {
    return get_inputmem().as_memloc();
  } else if(is_inputsto()) {
    return get_inputsto().as_stoloc();
  } else if(is_constant()) {
    return get_constant().as_memloc();
  } else if(is_apply()) {
    auto const& apply = get_apply();
    auto const& out_mem = apply.mems[0];
    return out_mem.as_memloc(apply.loc);
  } else if(is_move()) {
    auto const& move = get_move();
    auto const& [dst_loc, dst_offset] = move.dst;
    return memloc_t{
        .offset = dst_offset,
        .size = move.size,
        .loc = dst_loc};
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

mem_t memgraph_t::op_t::get_output_mem() const
{
  return get_output_memloc().as_mem();
}

stoloc_t memgraph_t::op_t::get_stoloc() const
{
  if(!(is_evict() || is_inputsto() || is_load())) {
    throw std::runtime_error(
        "There's no incident stoloc for nodes that are not evict/inputsto/load.");
  } else
  {
    if(is_evict()) {
      return get_evict().dst;
    } else if(is_load()) {
      return get_load().src;
    } else if(is_inputsto()) {
      return get_inputsto().as_stoloc();
    } else {
      throw std::runtime_error("should not reach");
    }
  }
}

bool memgraph_t::is_local_to(int id, int loc) const
{
  node_t const& node = nodes[id];
  if(node.op.is_inputsto()) {
    // Input storage nodes are special in that they don't map to
    // a single location. We determine if this inputsto occurs here if
    // any of its outgoing edges occur here.
    for(int const& out_id : node.outs) {
      if(is_local_to(out_id, loc))
      {
        return true;
      }
    }
    return false;
  } else {
    return node.op.is_local_to(loc);
  }
}

bool memgraph_t::op_t::is_local_to(int loc) const
{
  if(is_inputmem()) {
    return loc == get_inputmem().loc;
  } else if(is_inputsto()) {
    // If we have an input sto, we need to know which locs this storage loc
    // maps to. But even then, is it local to all of those maps?
    throw std::runtime_error("inputsto node: can't tell if local to loc");
  } else if(is_constant()) {
    return loc == get_constant().loc;
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

bool memgraph_t::apply_t::is_einsummable() const
{
  return std::holds_alternative<einsummable_t>(op);
}

bool memgraph_t::apply_t::is_touch() const
{
  return std::holds_alternative<touch_t>(op);
}

einsummable_t const&
memgraph_t::apply_t::get_einsummable() const
{
  return std::get<einsummable_t>(op);
}

touch_t const&
memgraph_t::apply_t::get_touch() const
{
  return std::get<touch_t>(op);
}

dtype_t
memgraph_t::apply_t::out_dtype() const
{
  if(is_einsummable()) {
    return get_einsummable().out_dtype();
  }
  if(is_touch()) {
    return get_touch().dtype;
  }
  throw std::runtime_error("should not reach");
}

vector<uint64_t>
memgraph_t::get_numbyte_on_evict() const
{
  vector<uint64_t> total_each_loc(num_compute_locs, 0);
  for(int id = 0; id != nodes.size(); ++id) {
    node_t const& node = nodes[id];
    op_t const& op = node.op;
    if (op.is_evict()) 
    {
      auto evict_node = op.get_evict();
      int loc = evict_node.src.loc;
      uint64_t size = evict_node.src.size;
      total_each_loc[loc] += size;
      
    } else if (op.is_load()) 
    {
      auto load_node = op.get_load();
      int loc = load_node.dst.loc;
      uint64_t size = load_node.dst.size;
      total_each_loc[loc] += size;
    }
  }

  return total_each_loc;
}

std::ostream &operator<<(std::ostream &out, mem_t const& mem)
{
  out << "[" << mem.offset << "," << mem.offset + mem.size << ")";
  return out;
}
std::ostream &operator<<(std::ostream &out, memloc_t const& memloc)
{
  out << "loc" << memloc.loc << memloc.as_mem();
  return out;
}

