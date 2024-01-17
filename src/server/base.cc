#include "base.h"
#include <fstream>
#include "../engine/repartition.h"

void server_base_t::insert_gid_without_data(int gid, relation_t const& relation)
{
  auto iter = gid_map.find(gid);
  if(iter != gid_map.end()) {
    throw std::runtime_error("this gid is already in the server");
  }
  gid_map.insert({gid, relation});
}

void server_base_t::execute_graph(
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
  if(make_parallel_partialize_groups()) {
    for(auto& node: taskgraph.nodes) {
      auto& op = node.op;
      if(op.is_partialize()) {
        auto& partialize = op.get_partialize();
        partialize.make_parallel();
      }
    }
  }

  int num_msgs = 0;
  uint64_t num_bytes = 0;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_move()) {
      num_msgs++;
      num_bytes += node.op.get_move().size;
    }
  }
  //DOUT("executing taskgraph with " << num_msgs << " moves, " << num_bytes << " bytes moved");

  //{
  //  std::ofstream f("tg.gv");
  //  taskgraph.print_graphviz(f);
  //  DOUT("printed tg.gv");
  //}

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

void server_base_t::execute(
  taskgraph_t const& taskgraph,
  map<int, relation_t> const& new_gid_map)
{
  execute(taskgraph);
  gid_map = new_gid_map;
}

dbuffer_t server_base_t::get_tensor_from_gid(int gid)
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

void server_base_t::insert_tensor(
  int gid,
  placement_t const& pl,
  dbuffer_t src_tensor)
{
  // get some new tids to use in the relation
  int t = get_max_tid() + 1;

  vtensor_t<int> tids(pl.block_shape());
  vector<int>& ts = tids.get();
  std::iota(ts.begin(), ts.end(), t);

  relation_t relation {
    .dtype = src_tensor.dtype,
    .placement = pl,
    .tids = tids
  };

  insert_tensor(gid, relation, src_tensor);
}

void server_base_t::insert_tensor(
  int gid,
  vector<uint64_t> const& shape,
  dbuffer_t src_tensor)
{
  insert_tensor(
    gid,
    placement_t(partition_t::singleton(shape)),
    src_tensor);
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

void server_mg_base_t::register_listen(
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

int server_mg_base_t::get_max_tid() {
  int ret = local_get_max_tid();

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

void server_mg_base_t::insert_relation(
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

void server_mg_base_t::insert_relation_helper(
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

void server_mg_base_t::remap(remap_relations_t const& remap_relations)
{
  broadcast_cmd(cmd_t::remap);
  remap_server(remap_relations);
}

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
      full_data_locs, alloc_settings, use_storage_);

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    create_storage_remaps(comm.get_world_size(), full_data_locs, inn_tg_to_loc);

  // not needed anymore because all the info is in out_tg_to_loc
  full_data_locs.clear();

  storage_remap_server(storage_remaps);

  comm.broadcast_string(memgraph.to_wire());

  execute_memgraph(memgraph, true);

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

  execute_memgraph(memgraph, true);

  rewrite_data_locs_client();
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
      if(is_local_location(loc)) {
        data.insert({tid, local_copy_data(tid)});
      }
    }
  }

  return data;
}

void server_mg_base_t::convert_remap_to_compute_node(
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

void server_mg_base_t::execute_tg_server(taskgraph_t const& taskgraph) {
  auto [mem_sizes, full_data_locs, which_storage] =
    recv_make_mg_info();

  //gremlin_t* gremlin = new gremlin_t("making memgraph");
  auto [inn_tg_to_loc, out_tg_to_loc, inputs_everywhere_mg_, core_mg] =
    memgraph_t::make_(
      taskgraph, which_storage, mem_sizes,
      full_data_locs, alloc_settings, use_storage_, split_off_inputs_);
  //delete gremlin;

  //{
  //std::ofstream f("mg.gv");
  //core_mg.print_graphviz(f);
  //DOUT("printed mg.gv");
  //}

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    create_storage_remaps(comm.get_world_size(), full_data_locs, inn_tg_to_loc);

  // this is not needed anymore
  full_data_locs.clear();

  storage_remap_server(storage_remaps);

  int n_mg = bool(inputs_everywhere_mg_) ? 2 : 1;
  comm.broadcast_contig_obj(n_mg);

  if(inputs_everywhere_mg_) {
    auto const& mg = inputs_everywhere_mg_.value();
    comm.broadcast_string(mg.to_wire());
    comm.barrier();
    execute_memgraph(mg, true);
  }

  {
    comm.broadcast_string(core_mg.to_wire());
    comm.barrier();
    execute_memgraph(core_mg, false);
  }

  rewrite_data_locs_server(out_tg_to_loc);
}

void server_mg_base_t::execute_tg_client() {
  send_make_mg_info();

  storage_remap_client();

  int n_mg = comm.recv_int(0);
  for(int i = 0; i != n_mg; ++i) {
    memgraph_t memgraph = memgraph_t::from_wire(comm.recv_string(0));
    comm.barrier();
    execute_memgraph(memgraph, false);
  }

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

