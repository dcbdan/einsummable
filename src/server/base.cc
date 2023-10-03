#include "base.h"

#include "../engine/repartition.h"

void server_base_t::execute(
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto make_relation = [&](int gid, vtensor_t<int> const& tids) {
    return relation_t {
      .dtype = graph.out_dtype(gid),
      .placement = placements[gid],
      .tids = tids
    };
  };

  auto [inn_g_to_t, out_g_to_t, taskgraph] =
    taskgraph_t::make(graph, placements);

  remap_relations_t r;
  for(auto const& [gid, dst_tids]: inn_g_to_t) {
    r.insert(
      get_relation(gid),             // src relation
      make_relation(gid, dst_tids)   // dst relation
    );
  }

  remap(r);

  execute(taskgraph);

  gid_map.clear();
  for(auto const& [gid, tids]: out_g_to_t) {
    gid_map.insert({gid, make_relation(gid, tids)});
  }
}

dbuffer_t server_base_t::get_tensor(int gid)
{
  return get_tensor(get_relation(gid));
}

relation_t const& server_base_t::get_relation(int gid) const
{
  return gid_map.at(gid);
}

void server_base_t::insert_tensor(
  int gid,
  relation_t const& dst_relation,
  dbuffer_t src_tensor)
{
  insert_relation(dst_relation, src_tensor);
  gid_map.insert({gid, dst_relation});
}

void server_base_t::remap(
  map<int, relation_t> const& gid_to_new_relations)
{
  remap_relations_t r;
  for(auto const& [gid, new_rel]: gid_to_new_relations) {
    r.insert(gid_map.at(gid), new_rel);
  }

  remap(r);

  gid_map = gid_to_new_relations;
}

void server_base_t::remap_gids(vector<tuple<int,int>> const& remap)
{
  map<int, relation_t> ret;

  for(auto const& [src,dst]: remap) {
    ret.insert({dst, gid_map.at(src)});
  }

  gid_map = ret;
}

void server_mg_base_t::listen() {
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
      repartition(comm, remap, data);
    } else if(cmd == cmd_t::insert_relation) {
      remap_relations_t remap = remap_relations_t::from_wire(comm.recv_string(0));
      map<int, buffer_t> tmp;
      repartition(comm, remap, tmp);
      local_insert_tensors(tmp);
    } else if(cmd == cmd_t::max_tid) {
      int max_tid_here = data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;
      int dst = 0;
      comm.send_int(dst, max_tid_here);
    } else if(cmd == cmd_t::registered_cmd) {
      string key = comm.recv_string(0);
      listeners.at(key)(this);
    } else if(cmd == cmd_t::shutdown) {
      break;
    }
  }
}

void server_mg_base_t::register_listen(
  string key, std::function<void(server_base_t*)> f)
{
  if(comm.get_this_rank() == 0) {
    throw std::runtime_error("rank zero should not call register_listen method");
  }
  if(listeners.count(key) > 0) {
    throw std::runtime_error("this key has already been set");
  }
  listeners.insert({key, f});
}

int server_mg_base_t::get_max_tid() {
  int ret = data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;

  broadcast_cmd(cmd_t::max_tid);

  for(int rank = 1; rank != comm.get_world_size(); ++rank) {
    ret = std::max(ret, comm.recv_int(rank));
  }

  return ret;
}

void server_mg_base_t::execute(taskgraph_t const& taskgraph)
{
  broadcast_cmd(cmd_t::execute_tg);
  execute_tg_server(taskgraph);
}

dbuffer_t server_mg_base_t::get_tensor(
  relation_t const& relation)
{
  broadcast_cmd(cmd_t::get_tensor);

  remap_relations_t remap;

  remap.insert(
    relation, relation.as_singleton(99));

  comm.broadcast_string(remap.to_wire());

  map<int, buffer_t> data = local_copy_source_data(remap);

  repartition(comm, remap, data);

  return dbuffer_t(relation.dtype, data.at(99));
}

void server_mg_base_t::insert_relation(
  relation_t const& relation,
  dbuffer_t src_tensor)
{
  if(relation.dtype != src_tensor.dtype) {
    throw std::runtime_error("can't partition different dtypes");
  }

  broadcast_cmd(cmd_t::insert_relation);

  remap_relations_t remap;
  remap.insert(
    relation.as_singleton(99), relation);

  comm.broadcast_string(remap.to_wire());

  map<int, buffer_t> tmp;
  tmp.insert({ 99, src_tensor.data });

  repartition(comm, remap, tmp);
  local_insert_tensors(tmp);
}

void server_mg_base_t::remap(remap_relations_t const& remap_relations)
{
  broadcast_cmd(cmd_t::remap);
  remap_server(remap_relations);
}

//////////////////////// remap helpers {{{
template <typename T>
void _update_map_with_new_tg_inns(
  map<int, T>& data,
  vector<tuple<int,int>> const& remap_gid,
  map<int, vtensor_t<int>> const& gid_to_inn,
  remap_relations_t const& _remap,
  optional<int> this_rank)
{
  auto const& remap = _remap.remap;

  map<int, T> tmp;
  for(int i = 0; i != remap_gid.size(); ++i) {
    auto const& gid  = std::get<0>(remap_gid[i]);
    auto const& info = std::get<0>(remap[i]);
    vector<int> const& locs = info.placement.locations.get();
    vector<int> const& inn_tids = info.tids.get();
    vector<int> const& mid_tids = gid_to_inn.at(gid).get();
    if(inn_tids.size() != mid_tids.size()) {
      throw std::runtime_error("!");
    }
    for(int j = 0; j != inn_tids.size(); ++j) {
      if(!this_rank || locs[j] == this_rank.value()) {
        int const& inn_tid = inn_tids[j];
        int const& mid_tid = mid_tids[j];
        tmp.insert({mid_tid, data.at(inn_tid)});
      }
    }
  }
  data = tmp;
}

template <typename T>
void _update_map_with_new_tg_outs(
  map<int, T>& data,
  vector<tuple<int, int>> const& remap_gid,
  map<int, vtensor_t<int>> const& gid_to_out,
  remap_relations_t const& _remap,
  optional<int> this_rank)
{
  auto const& remap = _remap.remap;

  map<int, T> tmp;
  for(int i = 0; i != remap_gid.size(); ++i) {
    auto const& gid  = std::get<1>(remap_gid[i]);
    auto const& info = std::get<1>(remap[i]);
    vector<int> const& locs = info.placement.locations.get();
    vector<int> const& out_tids = info.tids.get();
    vector<int> const& mid_tids = gid_to_out.at(gid).get();
    for(int j = 0; j != out_tids.size(); ++j) {
      if(!this_rank || locs[j] == this_rank.value()) {
        int const& out_tid = out_tids[j];
        int const& mid_tid = mid_tids[j];
        tmp.insert({out_tid, data.at(mid_tid)});
      }
    }
  }
  data = tmp;
}
// }}}

void server_mg_base_t::remap_server(remap_relations_t const& remap_relations)
{
  auto [remap_gid, g] = create_remap_graph_constructor(remap_relations);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  auto [mem_sizes, full_data_locs, which_storage] = recv_make_mg_info();

  // before: full_data_locs is with respect to the remap inn tids
  _update_map_with_new_tg_inns(
    full_data_locs, remap_gid, gid_to_inn, remap_relations, std::nullopt);
  // after: full_data_locs is with respect to the tasgkraph inns

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, which_storage, mem_sizes,
      full_data_locs, alloc_settings, true);

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    create_storage_remaps(comm.get_world_size(), full_data_locs, inn_tg_to_loc);

  // not needed anymore because all the info is in out_tg_to_loc
  full_data_locs.clear();

  storage_remap_server(storage_remaps);

  comm.broadcast_string(memgraph.to_wire());

  execute_memgraph(memgraph);

  _update_map_with_new_tg_outs(
    out_tg_to_loc, remap_gid, gid_to_out, remap_relations, std::nullopt);

  rewrite_data_locs_server(out_tg_to_loc);
}


void server_mg_base_t::remap_client()
{
  // This may be the same exact code as execute_tg_client--however
  // execute_tg_client and remap_client are accomplishing different things.

  send_make_mg_info();

  storage_remap_client();

  memgraph_t memgraph = memgraph_t::from_wire(comm.recv_string(0));

  execute_memgraph(memgraph);

  rewrite_data_locs_client();
}

buffer_t
server_mg_base_t::local_copy_data(int tid) {
  return local_copy_data_at(data_locs.at(tid));
}

map<int, buffer_t>
server_mg_base_t::local_copy_source_data(remap_relations_t const& remap)
{
  int this_rank = comm.get_this_rank();

  map<int, buffer_t> data;

  for(auto const& [inn_rel, _]: remap.remap) {
    auto const& locs = inn_rel.placement.locations.get();
    auto const& tids = inn_rel.tids.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      int const& tid = tids[bid];
      if(loc == this_rank) {
        data.insert({tid, local_copy_data(tid)});
      }
    }
  }

  return data;
}

void server_mg_base_t::execute_tg_server(taskgraph_t const& taskgraph) {
  auto [mem_sizes, full_data_locs, which_storage] =
    recv_make_mg_info();

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, which_storage, mem_sizes,
      full_data_locs, alloc_settings, true);

  //{
  //  std::ofstream f("mg.gv");
  //  memgraph.print_graphviz(f);
  //  DOUT("printed mg.gv");
  //}

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    create_storage_remaps(comm.get_world_size(), full_data_locs, inn_tg_to_loc);

  // this is not needed anymore
  full_data_locs.clear();

  storage_remap_server(storage_remaps);

  comm.broadcast_string(memgraph.to_wire());

  execute_memgraph(memgraph);

  rewrite_data_locs_server(out_tg_to_loc);
}

void server_mg_base_t::execute_tg_client() {
  send_make_mg_info();

  storage_remap_client();

  memgraph_t memgraph = memgraph_t::from_wire(comm.recv_string(0));

  execute_memgraph(memgraph);

  rewrite_data_locs_client();
}

vector<vector<std::array<int, 2>>>
server_mg_base_t::create_storage_remaps(
  int world_size,
  map<int, memstoloc_t> const& full_data_locs,
  map<int, memstoloc_t> const& inn_tg_to_loc)
{
  vector<vector<std::array<int, 2>>> storage_remaps(world_size);
  for(auto const& [id, mg_memstoloc]: inn_tg_to_loc) {
    if(mg_memstoloc.is_stoloc()) {
      auto const& [loc, new_sto_id] = mg_memstoloc.get_stoloc();
      auto const& [_, old_sto_id] = full_data_locs.at(id).get_stoloc();
      storage_remaps[loc].push_back({new_sto_id, old_sto_id});
    }
  }
  return storage_remaps;
}

std::ostream& operator<<(std::ostream& out, server_mg_base_t::cmd_t const& c) {
  auto const& items = server_mg_base_t::cmd_strs();
  out << items.at(int(c));
  return out;
}

std::istream& operator>>(std::istream& inn, server_mg_base_t::cmd_t& c) {
  c = server_mg_base_t::cmd_t(istream_expect_or(inn, server_mg_base_t::cmd_strs()));
  return inn;
}

