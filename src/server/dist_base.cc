#include "base.h"
#include "../engine/repartition.h"

void server_dist_base_t::listen() {
  if(comm.get_this_rank() == 0) {
    throw std::runtime_error("rank zero should not call listen method");
  }
  while(true) {
    cmd_t cmd = recv_cmd();
    if(cmd == cmd_t::execute_tg) {
      execute_tg_client();
    } else if(cmd == cmd_t::remap) {
      remap_client();
    } else if(cmd == cmd_t::get_tensor) {
      remap_relations_t remap = remap_relations_t::from_wire(comm.recv_string(0));
      map<int, buffer_t> data = local_copy_source_data(remap);
      convert_remap_to_compute_node(remap);
      repartition(comm, remap, data, get_cpu_threadpool());
    } else if(cmd == cmd_t::insert_relation) {
      insert_relation_helper(
        remap_relations_t::from_wire(comm.recv_string(0)),
        map<int, buffer_t>());
    } else if(cmd == cmd_t::max_tid) {
      int max_tid_here = local_get_max_tid();
      comm.send_int(0, max_tid_here);
    } else if(cmd == cmd_t::registered_cmd) {
      string key = comm.recv_string(0);
      listeners.at(key)();
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void server_dist_base_t::register_listen(
  string key, std::function<void()> f)
{
  if(comm.get_this_rank() == 0) {
    throw std::runtime_error("rank zero should not call register_listen method");
  }
  if(listeners.count(key) > 0) {
    throw std::runtime_error("this key has already been set");
  }
  listeners.insert({key, f});
}

void server_dist_base_t::execute(
  taskgraph_t const& taskgraph,
  map<string, scalar_t> const& scalar_vars)
{
  broadcast_cmd(cmd_t::execute_tg);

  execute_tg_server(taskgraph, scalar_vars);
}

dbuffer_t server_dist_base_t::get_tensor(
  relation_t const& relation)
{
  broadcast_cmd(cmd_t::get_tensor);

  remap_relations_t remap;

  // remap here is with respect to locations
  remap.insert(
    relation,
    relation.as_singleton(99, local_candidate_location()));

  comm.broadcast_string(remap.to_wire());

  map<int, buffer_t> data = local_copy_source_data(remap);

  convert_remap_to_compute_node(remap);

  // remap here is with respect to compute-node
  repartition(comm, remap, data, get_cpu_threadpool());

  return dbuffer_t(relation.dtype, data.at(99));
}

void server_dist_base_t::insert_relation(
  relation_t const& relation,
  dbuffer_t src_tensor)
{
  if(relation.dtype != src_tensor.dtype) {
    throw std::runtime_error("can't partition different dtypes");
  }

  broadcast_cmd(cmd_t::insert_relation);

  // remap here is with respect to locations
  remap_relations_t remap;
  remap.insert(
    relation.as_singleton(99, local_candidate_location()),
    relation);

  comm.broadcast_string(remap.to_wire());

  map<int, buffer_t> tmp;
  tmp.insert({ 99, src_tensor.data });

  // remap here is with respect to locations
  insert_relation_helper(remap, tmp);
}

void server_dist_base_t::insert_relation_helper(
  remap_relations_t remap,
  map<int, buffer_t> data)
{
  // At this point, the remap is with respect to _locations_ but
  // it needs to be with respect to _compute-nodes_.

  // First, collect a map from out tid to loc which will be used
  // to insert the data
  map<int, int> out_tid_to_loc;
  for(auto const& [_, rel]: remap.remap) {
    auto const& locs = rel.placement.locations.get();
    auto const& tids = rel.tids.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      int const& tid = tids[bid];
      if(is_local_location(loc)) {
        out_tid_to_loc.insert({tid, loc});
      }
    }
  }

  // Next, convert each relation to be with respect to
  // the _compute-node_
  convert_remap_to_compute_node(remap);

  // Now we repartition with respect to the compute nodes
  repartition(comm, remap, data, get_cpu_threadpool());

  // And insert with respect to locations
  map<int, tuple<int, buffer_t>> ret;
  for(auto const& [tid, buffer]: data) {
    ret.insert({tid, {out_tid_to_loc.at(tid), buffer}});
  }
  local_insert_tensors(ret);
}

void server_dist_base_t::remap(remap_relations_t const& remap_relations)
{
  broadcast_cmd(cmd_t::remap);
  remap_server(remap_relations);
}

int server_dist_base_t::get_max_tid() {
  int ret = local_get_max_tid();

  broadcast_cmd(cmd_t::max_tid);

  for(int rank = 1; rank != comm.get_world_size(); ++rank) {
    ret = std::max(ret, comm.recv_int(rank));
  }

  return ret;
}

void server_dist_base_t::shutdown() {
  broadcast_cmd(cmd_t::shutdown);
}

void server_dist_base_t::convert_remap_to_compute_node(
  remap_relations_t& remap)
{
  auto fix = [&](relation_t& rel) {
    auto& locs = rel.placement.locations.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int& loc = locs[bid];
      loc = loc_to_compute_node(loc);
    }
  };

  for(auto& [inn_rel, out_rel]: remap.remap) {
    fix(inn_rel);
    fix(out_rel);
  }
}

map<int, buffer_t>
server_dist_base_t::local_copy_source_data(remap_relations_t const& remap)
{
  int this_rank = comm.get_this_rank();

  map<int, buffer_t> data;

  for(auto const& [inn_rel, _]: remap.remap) {
    auto const& locs = inn_rel.placement.locations.get();
    auto const& tids = inn_rel.tids.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      int const& tid = tids[bid];
      if(is_local_location(loc)) {
        data.insert({tid, local_copy_data(tid)});
      }
    }
  }

  return data;
}

std::ostream& operator<<(std::ostream& out, server_dist_base_t::cmd_t const& c) {
  auto const& items = server_dist_base_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, server_dist_base_t::cmd_t& c) {
  c = server_dist_base_t::cmd_t(istream_expect_or(inn, server_dist_base_t::cmd_strs()));
  return inn;
}


