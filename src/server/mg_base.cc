#include "base.h"
#include "../engine/repartition.h"

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

  // remap memgraphs should not need any workspace as they will not have
  // any apply nodes, and only taskgraph apply nodes are allowed to 
  // have workspaces
  map<int, uint64_t> required_workspace = {}; 

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, required_workspace, which_storage, mem_sizes,
      full_data_locs, alloc_settings, has_storage());

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

void server_mg_base_t::execute_tg_server(
  taskgraph_t const& taskgraph,
  map<string, scalar_t> const& scalar_vars)
{
  auto [mem_sizes, full_data_locs, which_storage] =
    recv_make_mg_info();

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  DLINEOUT("storage:           " << has_storage());
  DLINEOUT("inputs everywhere: " << split_off_inputs_);
  DLINEOUT("mem_sizes:         " << mem_sizes);
 
  // some nodes may need workspace!
  map<int, uint64_t> required_workspace = 
    build_required_workspace_info(taskgraph, scalar_vars);
  // TODO: this should really require the inputs of 
  //            all the client servers as well
  // For now, just throw an error:
  if(this->comm.get_world_size() > 1) {
    throw std::runtime_error("build required workspace: only works with world size 1");
  }

  DLINE;

  //gremlin_t* gremlin = new gremlin_t("making memgraph");
  auto [inn_tg_to_loc, out_tg_to_loc, inputs_everywhere_mg_, core_mg] =
    memgraph_t::make_(
      taskgraph, required_workspace, which_storage, mem_sizes,
      full_data_locs, alloc_settings, has_storage(), split_off_inputs_);
  //delete gremlin;

  DLINE;
  //vector<uint64_t> io_bytes_each_loc = core_mg.get_numbyte_on_evict();
  // DOUT("Number of bytes involved in I/O: " << io_bytes_each_loc);

  DLINE;
  if(inputs_everywhere_mg_) {
    std::ofstream f("inputs_mg.gv");
    inputs_everywhere_mg_.value().print_graphviz(f);
    DOUT("printed inputs_mg.gv");
  }

  DLINE;
  {
    std::ofstream f("mg.gv");
    core_mg.print_graphviz(f);
    DOUT("printed mg.gv");
  }

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
    comm.broadcast_string(scalar_vars_to_wire({}));
    comm.barrier();
    execute_memgraph(mg, true);
  }

  // auto start_memgraph_time = std::chrono::high_resolution_clock::now();

  {
    comm.broadcast_string(core_mg.to_wire());
    comm.broadcast_string(scalar_vars_to_wire(scalar_vars));
    comm.barrier();
    execute_memgraph(core_mg, false, scalar_vars);
  }
  // auto end_memgraph_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> elapsed = end_memgraph_time - start_memgraph_time;
  // std::cout << "Memgraph execution elapsed time is " << elapsed.count() << " milliseconds" << std::endl;

  rewrite_data_locs_server(out_tg_to_loc);
}

void server_mg_base_t::execute_tg_client() {
  send_make_mg_info();

  storage_remap_client();

  int n_mg = comm.recv_int(0);
  for(int i = 0; i != n_mg; ++i) {
    memgraph_t memgraph = memgraph_t::from_wire(comm.recv_string(0));
    map<string, scalar_t> scalar_vars = scalar_vars_from_wire(comm.recv_string(0));
    comm.barrier();
    execute_memgraph(memgraph, false, scalar_vars);
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
      storage_remaps.at(loc).push_back({new_sto_id, old_sto_id});
    }
  }
  return storage_remaps;
}


