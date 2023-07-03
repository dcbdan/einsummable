#include "manager.h"

#include "repartition.h"

loc_manager_t::loc_manager_t(mpi_t* mpi, execute_taskgraph_settings_t const& settings)
  : mpi(mpi), settings(settings)
{}

void loc_manager_t::listen() {
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
    } else if(cmd == cmd_t::partition_into_data) {
      map<int, buffer_t> tmp;
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      repartition(mpi, remap, tmp);
      copy_into_data(tmp, remap);
    } else if(cmd == cmd_t::remap_data) {
      auto remap = remap_relations_t::from_wire(mpi->recv_str(0));
      repartition(mpi, remap, data);
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void loc_manager_t::execute(taskgraph_t const& taskgraph)
{
  gremlin_t gremlin("execute from manager");
  broadcast_cmd(cmd_t::execute);
  broadcast_str(taskgraph.to_wire());
  _execute(taskgraph);
}

dbuffer_t loc_manager_t::unpartition(relation_t const& relation) {
  broadcast_cmd(cmd_t::unpartition);

  map<int, buffer_t> tmp = data;

  remap_relations_t remap;

  remap.insert(
    relation, relation.as_singleton(99));

  broadcast_str(remap.to_wire());

  repartition(mpi, remap, tmp);

  return dbuffer_t(relation.dtype, tmp.at(99));
}

void loc_manager_t::partition_into_data(
  relation_t const& relation,
  dbuffer_t src_tensor)
{
  if(relation.dtype != src_tensor.dtype) {
    throw std::runtime_error("can't partition different dtypes");
  }

  broadcast_cmd(cmd_t::partition_into_data);

  remap_relations_t remap;
  remap.insert(
    relation.as_singleton(99), relation);

  broadcast_str(remap.to_wire());

  map<int, buffer_t> tmp;
  tmp.insert({ 99, src_tensor.data });

  repartition(mpi, remap, tmp);

  copy_into_data(tmp, remap);
}

void loc_manager_t::remap_data(remap_relations_t const& remap) {
  broadcast_cmd(cmd_t::remap_data);

  broadcast_str(remap.to_wire());

  repartition(mpi, remap, data);
}

void loc_manager_t::shutdown() {
  broadcast_cmd(cmd_t::shutdown);
}

void loc_manager_t::broadcast_cmd(cmd_t const& cmd) {
  broadcast_str(write_with_ss(cmd));
}

void loc_manager_t::broadcast_str(string const& str) {
  if(!mpi) { return; }

  if(mpi->this_rank != 0) {
    throw std::runtime_error("only rank 0 should do broadcasting");
  }

  for(int i = 1; i != mpi->world_size; ++i) {
    mpi->send_str(str, i);
  }
}

loc_manager_t::cmd_t loc_manager_t::recv_cmd() {
  return parse_with_ss<cmd_t>(mpi->recv_str(0));
}

void loc_manager_t::_execute(taskgraph_t const& taskgraph)
{
  update_kernel_manager(kernel_manager, taskgraph);
  ::execute(taskgraph, settings, kernel_manager, mpi, data);
}

void loc_manager_t::copy_into_data(
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

std::ostream& operator<<(std::ostream& out, loc_manager_t::cmd_t const& c) {
  auto const& items = loc_manager_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, loc_manager_t::cmd_t& c) {
  c = loc_manager_t::cmd_t(istream_expect_or(inn, loc_manager_t::cmd_strs()));
  return inn;
}

