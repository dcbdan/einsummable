#include "repartition.h"

#include "executetg.h"
#include "executemg.h"

#include "../../einsummable/graph.h"
#include "../../einsummable/taskgraph.h"

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

void repartition(
  mpi_t* mpi,
  remap_relations_t const& _remap,
  map<int, buffer_t>& data)
{
  auto const& remap = _remap.remap;

  auto [remap_gid, g] = create_remap_graph_constructor(_remap);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  int this_rank = bool(mpi) ? mpi->this_rank : 0;

  _update_map_with_new_tg_inns(
    data, remap_gid, gid_to_inn, _remap, this_rank);

  {
    auto settings = execute_taskgraph_settings_t::default_settings();
    kernel_manager_t ks;
    execute_taskgraph(taskgraph, settings, ks, mpi, data);
  }

  _update_map_with_new_tg_outs(
    data, remap_gid, gid_to_out, _remap, this_rank);
}

void repartition_server(
  mpi_t* mpi,
  remap_relations_t const& remap_relations,
  allocator_settings_t const& alloc_settings,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage)
{
  int this_rank = bool(mpi) ? mpi->this_rank : 0;
  _tg_with_mg_helper_t helper(mpi, data_locs, mem, storage, this_rank);

  auto [remap_gid, g] = create_remap_graph_constructor(remap_relations);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  vector<uint64_t> mem_sizes = helper.recv_mem_sizes();

  map<int, memstoloc_t> full_data_locs = helper.recv_full_data_locs();

  // before: full_data_locs is with respect to the remap inn tids
  _update_map_with_new_tg_inns(
    full_data_locs, remap_gid, gid_to_inn, remap_relations, std::nullopt);
  // after: full_data_locs is with respect to the tasgkraph inns

  vector<int> which_storage(helper.world_size);
  std::iota(which_storage.begin(), which_storage.end(), 0);

  auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
    memgraph_t::make(
      taskgraph, which_storage, mem_sizes,
      full_data_locs, alloc_settings, true);

  // memgraph now uses wtvr storage ids it chooses... So for each input,
  // figure out what the remap is
  vector<vector<std::array<int, 2>>> storage_remaps =
    helper.create_storage_remaps(full_data_locs, inn_tg_to_loc);

  // not needed anymore because all the info is in out_tg_to_loc
  full_data_locs.clear();

  helper.storage_remap_server(storage_remaps);

  helper.broadcast_memgraph(memgraph);

  {
    auto exec_sts = execute_memgraph_settings_t::default_settings();
    kernel_manager_t kernel_manager;
    execute_memgraph(memgraph, exec_sts, kernel_manager, mpi, mem, storage);
  }

  _update_map_with_new_tg_outs(
    out_tg_to_loc, remap_gid, gid_to_out, remap_relations, std::nullopt);

  helper.rewrite_data_locs_server(out_tg_to_loc);
}

void repartition_client(
  mpi_t* mpi,
  int server_rank,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage)
{
  _tg_with_mg_helper_t helper(mpi, data_locs, mem, storage, server_rank);

  helper.send_mem_size();

  helper.send_data_locs();

  helper.storage_remap_client();

  memgraph_t memgraph = helper.recv_memgraph();

  {
    auto exec_sts = execute_memgraph_settings_t::default_settings();
    kernel_manager_t kernel_manager;
    execute_memgraph(memgraph, exec_sts, kernel_manager, mpi, mem, storage);
  }

  helper.rewrite_data_locs_client();
}

