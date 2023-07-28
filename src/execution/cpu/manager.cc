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
      listeners.at(key)(*this);
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void tg_manager_t::register_listen(
  string key, std::function<void(tg_manager_t&)> f)
{
  listeners.insert({key, f});
}

void tg_manager_t::execute(taskgraph_t const& taskgraph)
{
  gremlin_t gremlin("execute from manager");
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

void tg_manager_t::broadcast_cmd(cmd_t const& cmd) {
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
  update_kernel_manager(kernel_manager, taskgraph);
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

//mg_manager_t::mg_manager_t(
//  mpi_t* mpi,
//  execute_memgraph_settings_t const& exec_sts,
//  uint64_t memory_size,
//  allocator_settings_t alloc_sts)
//  : mpi(mpi), exec_settings(exec_sts), alloc_settings(alloc_sts)
//{
//  mem = make_buffer(memory_size);
//}
//
//void mg_manager_t::listen() {
//  if(!mpi) {
//    throw std::runtime_error("should not call listen if mpi is not setup");
//  }
//  if(mpi->this_rank == 0) {
//    throw std::runtime_error("rank zero should not call listen method");
//  }
//  while(true) {
//    cmd_t cmd = recv_cmd();
//    if(cmd == cmd_t::execute) {
//      memgraph_t mg = taskgraph_t::from_wire(mpi->recv_str(0));
//      _execute(mg);
//    } else if(cmd == cmd_t::unpartition) {
//      // TODO: how to unpartition?
//    } else if(cmd == cmd_t::partition_into) {
//      // TODO: how to partition_into this guy
//    } else if(cmd == cmd_t::remap) {
//      remap_relations_t remap; // not actually used
//      repartition(mpi, remap, alloc_settings, data_locs, mem, storage);
//    } else if(cmd == cmd_t::max_tid) {
//      int max_tid_here = data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;
//      mpi->send_int(max_tid_here, 0, 0);
//    } else if(cmd == cmd_t::shutdown) {
//      break;
//    }
//  }
//}
//
//void mg_manager_t::execute(taskgraph_t const& taskgraph) {
//  vector<int> which_storage(world_size);
//  std::iota(which_storage.begin(), which_storage.end(), 0);
//
//  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
//    memgraph_t::make(
//      taskgraph, which_storage, mem_sizes,
//      data_locs, alloc_settings, true);
//
//  vector<vector<std::array<int, 2>>> storage_remaps =
//    create_storage_remaps(mem_sizes.size(), data_locs, inn_tg_to_loc);
//
//  // TODO: remap all the storage items
//
//
//}
//
//void mg_manager_t::execute(memgraph_t const& memgraph) {
//  gremlin_t gremlin("execute memgraph from manager");
//  broadcast_cmd(cmd_t::execute);
//  broadcast_str(memgraph.to_wire());
//  _execute(memgraph);
//}
//
//void mg_manager_t::remap(remap_relations_t const& remap) {
//  broadcast_cmd(cmd_t::remap);
//  repartition(mpi, remap, alloc_settings, data_locs, mem, storage);
//}
//
//void mg_manager_t::_execute(memgraph_t const& mg) {
//  update_kernel_manager(kernel_manager, memgraph);
//  execute_memgraph(memgraph, exec_settings, kernel_manager, mpi, mem, storage);
//}

std::ostream& operator<<(std::ostream& out, tg_manager_t::cmd_t const& c) {
  auto const& items = tg_manager_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, tg_manager_t::cmd_t& c) {
  c = tg_manager_t::cmd_t(istream_expect_or(inn, tg_manager_t::cmd_strs()));
  return inn;
}

//std::ostream& operator<<(std::ostream& out, mg_manager_t::cmd_t const& c) {
//  auto const& items = mg_manager_t::cmd_strs();
//  out << items.at(int(c));
//  return out;
//}
//
//std::istream& operator>>(std::istream& inn, mg_manager_t::cmd_t& c) {
//  c = mg_manager_t::cmd_t(istream_expect_or(inn, mg_manager_t::cmd_strs()));
//  return inn;
//}

