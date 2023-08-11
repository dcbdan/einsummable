#include "manager.h"

#include "repartition.h"

tg_manager_t::tg_manager_t(mpi_t* mpi, execute_taskgraph_settings_t const& settings)
  : mpi(mpi), settings(settings)
{}

void tg_manager_t::listen() {
  if(!mpi) {
    throw std::runtime_error("should not call listen if mpi is not setup");
  }
  if(mpi->this_rank == 0) {
    throw std::runtime_error("rank zero should not call listen method");
  }

  while(true) {
    cmd_t cmd = recv_cmd();
    if(cmd == cmd_t::execute) {
      taskgraph_t tg = taskgraph_t::from_wire(mpi->recv_str(0));
      _execute(tg);
    } else if(cmd == cmd_t::update_km) {
      string str = mpi->recv_str(0);
      es_proto::EinsummableList es;
      if(!es.ParseFromString(str)) {
        throw std::runtime_error("could not parse einsummable list!");
      }
      _update_km(es);
    } else if(cmd == cmd_t::unpartition) {
      map<int, buffer_t> tmp = data;
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      repartition(mpi, remap, tmp);
    } else if(cmd == cmd_t::partition_into) {
      map<int, buffer_t> tmp;
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      repartition(mpi, remap, tmp);
      copy_into_data(tmp, remap);
    } else if(cmd == cmd_t::remap) {
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      repartition(mpi, remap, data);
    } else if(cmd == cmd_t::max_tid) {
      int max_tid_here = data.size() == 0 ? -1 : data.rbegin()->first;
      mpi->send_int(max_tid_here, 0, 0);
    } else if(cmd == cmd_t::registered_cmd) {
      string key = mpi->recv_str(0);
      listeners.at(key)(this);
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void tg_manager_t::register_listen(
  string key, std::function<void(manager_base_t*)> f)
{
  if(!mpi || mpi->this_rank == 0) {
    throw std::runtime_error("rank zero should not call register_listen method");
  }
  if(listeners.count(key) > 0) {
    throw std::runtime_error("this key has already been set");
  }
  listeners.insert({key, f});
}

void tg_manager_t::execute(taskgraph_t const& taskgraph)
{
  gremlin_t gremlin("execute from tg manager");
  broadcast_cmd(cmd_t::execute);
  broadcast_str(taskgraph.to_wire());
  _execute(taskgraph);
}

dbuffer_t tg_manager_t::get_tensor(relation_t const& relation) {
  broadcast_cmd(cmd_t::unpartition);

  map<int, buffer_t> tmp = data;

  remap_relations_t remap;

  remap.insert(
    relation, relation.as_singleton(99));

  broadcast_str(remap.to_wire());

  repartition(mpi, remap, tmp);

  return dbuffer_t(relation.dtype, tmp.at(99));
}

void tg_manager_t::partition_into(
  relation_t const& relation,
  dbuffer_t src_tensor)
{
  if(relation.dtype != src_tensor.dtype) {
    throw std::runtime_error("can't partition different dtypes");
  }

  broadcast_cmd(cmd_t::partition_into);

  remap_relations_t remap;
  remap.insert(
    relation.as_singleton(99), relation);

  broadcast_str(remap.to_wire());

  map<int, buffer_t> tmp;
  tmp.insert({ 99, src_tensor.data });

  repartition(mpi, remap, tmp);

  copy_into_data(tmp, remap);

}

void tg_manager_t::remap(remap_relations_t const& remap) {
  broadcast_cmd(cmd_t::remap);

  broadcast_str(remap.to_wire());

  repartition(mpi, remap, data);
}

void tg_manager_t::update_kernel_manager(taskgraph_t const& tg) {
  int world_size = bool(mpi) ? mpi->world_size : 1;
  vector<std::unordered_set<einsummable_t>> es(world_size);
  for(auto const& node: tg.nodes) {
    auto const& op = node.op;
    if(op.is_apply()) {
      int loc = op.out_loc();
      auto const& e = op.get_apply().einsummable;
      es[loc].insert(e.merge_adjacent_dims());
    }
  }

  broadcast_cmd(cmd_t::update_km);
  _broadcast_es(es);
  _update_km(es[0]);
}

void tg_manager_t::custom_command(string key) {
  broadcast_cmd(cmd_t::registered_cmd);
  broadcast_str(key);
}

void tg_manager_t::_broadcast_es(
  vector<std::unordered_set<einsummable_t>> const& einsummables)
{
  for(int loc = 1; loc != einsummables.size(); ++loc) {
    es_proto::EinsummableList es;
    for(auto const& einsummable: einsummables[loc]) {
      es_proto::Einsummable* e = es.add_es();
      einsummable.to_proto(*e);
    }
    string ret;
    es.SerializeToString(&ret);
    mpi->send_str(ret, loc);
  }
}

void tg_manager_t::_update_km(
  std::unordered_set<einsummable_t> const& es)
{
  for(auto const& e: es) {
    auto maybe = kernel_manager.build(e);
    if(!maybe) {
      throw std::runtime_error("tg manager: could not build a kernel");
    }
  }
}

void tg_manager_t::_update_km(es_proto::EinsummableList const& es) {
  int n = es.es_size();
  for(int i = 0; i != n; ++i) {
    es_proto::Einsummable const& e = es.es(i);
    einsummable_t einsummable = einsummable_t::from_proto(e);
    auto maybe = kernel_manager.build(einsummable);
    if(!maybe) {
      throw std::runtime_error("tg manager: could not build a kernel");
    }
  }
}

int tg_manager_t::get_max_tid() {
  int ret = data.size() == 0 ? -1 : data.rbegin()->first;

  if(!mpi) { return ret; }

  broadcast_cmd(cmd_t::max_tid);

  for(int i = 1; i != mpi->world_size; ++i) {
    ret = std::max(ret, mpi->recv_int_from_anywhere(0));
  }

  return ret;
}

void tg_manager_t::shutdown() {
  broadcast_cmd(cmd_t::shutdown);
}

void tg_manager_t::broadcast_cmd(tg_manager_t::cmd_t const& cmd) {
  broadcast_str(write_with_ss(cmd));
}

void tg_manager_t::broadcast_str(string const& str) {
  if(!mpi) { return; }

  if(mpi->this_rank != 0) {
    throw std::runtime_error("only rank 0 should do broadcasting");
  }

  for(int i = 1; i != mpi->world_size; ++i) {
    mpi->send_str(str, i);
  }
}

tg_manager_t::cmd_t tg_manager_t::recv_cmd() {
  return parse_with_ss<cmd_t>(mpi->recv_str(0));
}

void tg_manager_t::_execute(taskgraph_t const& taskgraph)
{
  execute_taskgraph(taskgraph, settings, kernel_manager, mpi, data);
}

void tg_manager_t::copy_into_data(
  map<int, buffer_t>& tmp,
  remap_relations_t const& remap)
{
  for(auto const& [_, dst]: remap.remap) {
    vector<int> const& locs = dst.placement.locations.get();
    vector<int> const& tids = dst.tids.get();
    for(int i = 0; i != locs.size(); ++i) {
      if(mpi && locs[i] == mpi->this_rank) {
        int const& tid = tids[i];
        data.insert_or_assign(tid, tmp.at(tid));
      }
    }
  }
}

mg_manager_t::mg_manager_t(
  mpi_t* mpi,
  execute_memgraph_settings_t const& exec_sts,
  uint64_t memory_size,
  allocator_settings_t alloc_sts)
  : mpi(mpi), exec_settings(exec_sts), alloc_settings(alloc_sts)
{
  mem = make_buffer(memory_size);
}

void mg_manager_t::listen() {
  if(!mpi) {
    throw std::runtime_error("should not call listen if mpi is not setup");
  }
  if(mpi->this_rank == 0) {
    throw std::runtime_error("rank zero should not call listen method");
  }
  while(true) {
    cmd_t cmd = recv_cmd();
    if(cmd == cmd_t::execute_tg) {
      execute_taskgraph_as_memgraph_client(
        exec_settings, kernel_manager, 0,
        mpi, data_locs, mem, storage);
    } else if(cmd == cmd_t::execute_mg) {
      memgraph_t mg = memgraph_t::from_wire(mpi->recv_str(0));
      _execute_mg(mg);
    } else if(cmd == cmd_t::update_km) {
      string str = mpi->recv_str(0);
      es_proto::EinsummableList es;
      if(!es.ParseFromString(str)) {
        throw std::runtime_error("could not parse einsummable list!");
      }
      _update_km(es);
    } else if(cmd == cmd_t::unpartition) {
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      _unpartition(remap);
    } else if(cmd == cmd_t::partition_into) {
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      map<int, buffer_t> tmp;
      _remap_into_here(remap, tmp);
    } else if(cmd == cmd_t::remap) {
      repartition_client(mpi, 0, data_locs, mem, storage);
    } else if(cmd == cmd_t::max_tid) {
      int max_tid_here = data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;
      mpi->send_int(max_tid_here, 0, 0);
    } else if(cmd == cmd_t::registered_cmd) {
      string key = mpi->recv_str(0);
      listeners.at(key)(this);
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void mg_manager_t::register_listen(
  string key, std::function<void(manager_base_t*)> f)
{
  if(!mpi || mpi->this_rank == 0) {
    throw std::runtime_error("rank zero should not call register_listen method");
  }
  if(listeners.count(key) > 0) {
    throw std::runtime_error("this key has already been set");
  }
  listeners.insert({key, f});
}

void mg_manager_t::update_kernel_manager(taskgraph_t const& tg) {
  int world_size = bool(mpi) ? mpi->world_size : 1;
  vector<std::unordered_set<einsummable_t>> es(world_size);
  for(auto const& node: tg.nodes) {
    auto const& op = node.op;
    if(op.is_apply()) {
      int loc = op.out_loc();
      auto const& e = op.get_apply().einsummable;
      es[loc].insert(e.merge_adjacent_dims());
    }
  }

  broadcast_cmd(cmd_t::update_km);
  _broadcast_es(es);
  _update_km(es[0]);
}

void mg_manager_t::update_kernel_manager(memgraph_t const& mg) {
  vector<std::unordered_set<einsummable_t>> es;
  for(auto const& node: mg.nodes) {
    auto const& op = node.op;
    if(op.is_apply()) {
      auto const& apply = op.get_apply();
      if(apply.is_einsummable()) {
        int const& loc = apply.loc;
        auto const& e = apply.get_einsummable();
        es[loc].insert(e.merge_adjacent_dims());
      }
    }
  }
  broadcast_cmd(cmd_t::update_km);
  _broadcast_es(es);
  _update_km(es[0]);
}

void mg_manager_t::custom_command(string key) {
  broadcast_cmd(cmd_t::registered_cmd);
  broadcast_str(key);
}

void mg_manager_t::_broadcast_es(
  vector<std::unordered_set<einsummable_t>> const& einsummables)
{
  for(int loc = 1; loc != einsummables.size(); ++loc) {
    es_proto::EinsummableList es;
    for(auto const& einsummable: einsummables[loc]) {
      es_proto::Einsummable* e = es.add_es();
      einsummable.to_proto(*e);
    }
    string ret;
    es.SerializeToString(&ret);
    mpi->send_str(ret, loc);
  }
}

void mg_manager_t::_update_km(
  std::unordered_set<einsummable_t> const& es)
{
  for(auto const& e: es) {
    auto maybe = kernel_manager.build(e);
    if(!maybe) {
      throw std::runtime_error("mg manager: could not build a kernel");
    }
  }
}

void mg_manager_t::_update_km(es_proto::EinsummableList const& es) {
  int n = es.es_size();
  for(int i = 0; i != n; ++i) {
    es_proto::Einsummable const& e = es.es(i);
    einsummable_t einsummable = einsummable_t::from_proto(e);
    auto maybe = kernel_manager.build(einsummable);
    if(!maybe) {
      throw std::runtime_error("mg manager: could not build a kernel");
    }
  }
}

int mg_manager_t::get_max_tid() {
  int ret = data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;

  if(!mpi) { return ret; }

  broadcast_cmd(cmd_t::max_tid);

  for(int i = 1; i != mpi->world_size; ++i) {
    ret = std::max(ret, mpi->recv_int_from_anywhere(0));
  }

  return ret;
}

void mg_manager_t::shutdown() {
  broadcast_cmd(cmd_t::shutdown);
}

void mg_manager_t::execute(taskgraph_t const& taskgraph) {
  gremlin_t gremlin("execute taskgraph from mg manager");
  broadcast_cmd(cmd_t::execute_tg);
  execute_taskgraph_as_memgraph_server(
    taskgraph, exec_settings, kernel_manager, alloc_settings,
    mpi, data_locs, mem, storage);
}

void mg_manager_t::execute(memgraph_t const& memgraph) {
  gremlin_t gremlin("execute memgraph from manager");
  broadcast_cmd(cmd_t::execute_mg);
  broadcast_str(memgraph.to_wire());
  _execute_mg(memgraph);
}

dbuffer_t mg_manager_t::get_tensor(relation_t const& relation) {
  broadcast_cmd(cmd_t::unpartition);

  remap_relations_t remap;

  remap.insert(
    relation, relation.as_singleton(99));

  broadcast_str(remap.to_wire());

  // _unpartition will copy the data completely, so
  // all the buffers are fresh and not referencing the large
  // buffer
  map<int, buffer_t> data = _unpartition(remap);

  return dbuffer_t(relation.dtype, data.at(99));
}


void mg_manager_t::remap(remap_relations_t const& remap) {
  broadcast_cmd(cmd_t::remap);
  repartition_server(mpi, remap, alloc_settings, data_locs, mem, storage);
}

void mg_manager_t::_execute_mg(memgraph_t const& mg) {
  execute_memgraph(mg, exec_settings, kernel_manager, mpi, mem, storage);
}

buffer_t mg_manager_t::get_copy_of_data(int tid) {
  memsto_t const& loc = data_locs.at(tid);
  if(loc.is_mem()) {
    mem_t const& m = loc.get_mem();
    buffer_t src = make_buffer_reference(mem->data + m.offset, m.size);
    buffer_t dst = make_buffer(m.size);
    std::memcpy(dst->raw(), src->raw(), m.size);
    return dst;
  } else if(loc.is_sto()) {
    int const& id = loc.get_sto();
    return storage.load(id);
  } else {
    throw std::runtime_error("should not reach: memsto");
  }
}

map<int, buffer_t> mg_manager_t::_unpartition(remap_relations_t const& remap) {
  int this_rank = bool(mpi) ? mpi->this_rank : 0;

  map<int, buffer_t> data;
  for(auto const& [inn_rel, _]: remap.remap) {
    auto const& locs = inn_rel.placement.locations.get();
    auto const& tids = inn_rel.tids.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      int const& tid = tids[bid];
      if(loc == this_rank) {
        data.insert({tid, get_copy_of_data(tid)});
      }
    }
  }

  repartition(mpi, remap, data);

  return data;
}

void mg_manager_t::broadcast_cmd(mg_manager_t::cmd_t const& cmd) {
  broadcast_str(write_with_ss(cmd));
}

void mg_manager_t::broadcast_str(string const& str) {
  if(!mpi) { return; }

  if(mpi->this_rank != 0) {
    throw std::runtime_error("only rank 0 should do broadcasting");
  }

  for(int i = 1; i != mpi->world_size; ++i) {
    mpi->send_str(str, i);
  }
}

mg_manager_t::cmd_t mg_manager_t::recv_cmd() {
  return parse_with_ss<cmd_t>(mpi->recv_str(0));
}

void mg_manager_t::partition_into(
  relation_t const& relation,
  dbuffer_t src_tensor)
{
  if(relation.dtype != src_tensor.dtype) {
    throw std::runtime_error("can't partition different dtypes");
  }

  broadcast_cmd(cmd_t::partition_into);

  remap_relations_t remap;
  remap.insert(
    relation.as_singleton(99), relation);

  broadcast_str(remap.to_wire());

  map<int, buffer_t> tmp;
  tmp.insert({ 99, src_tensor.data });

  _remap_into_here(remap, tmp);
}

void mg_manager_t::_remap_into_here(
  remap_relations_t const& remap,
  map<int, buffer_t> data)
{
  int this_rank = bool(mpi) ? mpi->this_rank : 0;

  repartition(mpi, remap, data);

  // create an allocator that represents the current utilization
  // of this buffer
  allocator_t allocator(mem->size, alloc_settings);
  for(auto const& [_, memsto]: data_locs) {
    if(memsto.is_mem()) {
      auto const& [offset, size] = memsto.get_mem();
      allocator.allocate_at_without_deps(offset, size);
    }
  }

  int _sto_id = storage.get_max_id() + 1;

  for(auto const& [_, rel]: remap.remap) {
    auto const& locs = rel.placement.locations.get();
    auto const& tids = rel.tids.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      int const& tid = tids[bid];
      if(loc == this_rank) {
        auto& buffer = data.at(tid);
        uint64_t const& size = buffer->size;
        auto maybe = allocator.try_to_allocate_without_deps(size);
        if(maybe) {
          uint64_t offset = maybe.value();
          data_locs.insert({tid, memsto_t(mem_t{ offset, size })});
          std::copy(buffer->data, buffer->data + size, mem->data + offset);
        } else {
          int new_sto_id = _sto_id++;
          data_locs.insert({tid, memsto_t(new_sto_id)});
          storage.write(buffer, new_sto_id);
        }
      }
    }
  }
}

std::ostream& operator<<(std::ostream& out, tg_manager_t::cmd_t const& c) {
  auto const& items = tg_manager_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, tg_manager_t::cmd_t& c) {
  c = tg_manager_t::cmd_t(istream_expect_or(inn, tg_manager_t::cmd_strs()));
  return inn;
}

std::ostream& operator<<(std::ostream& out, mg_manager_t::cmd_t const& c) {
  auto const& items = mg_manager_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, mg_manager_t::cmd_t& c) {
  c = mg_manager_t::cmd_t(istream_expect_or(inn, mg_manager_t::cmd_strs()));
  return inn;
}

