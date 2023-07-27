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

struct _dataloc_info_t {
  int id;
  memsto_t memsto;
};

// Assumption: storage loc i == compute loc i
//             for all i

void _send_data_locs(
  mpi_t* mpi,
  map<int, memstoloc_t> const& data_locs,
  int dst)
{
  vector<_dataloc_info_t> items;
  for(auto const& [id, memstoloc]: data_locs) {
    if(memstoloc.is_memloc()) {
      auto const& memloc = memstoloc.get_memloc();
      if(memloc.loc == dst) {
        items.push_back(_dataloc_info_t { id, memloc.as_memsto() });
      }
    } else {
      auto const& stoloc = memstoloc.get_stoloc();
      if(stoloc.loc == dst) {
        items.push_back(_dataloc_info_t { id, stoloc.as_memsto() });
      }
    }
  }

  mpi->send_vector(items, dst);
}

void _send_data_locs(
  mpi_t* mpi,
  map<int, memsto_t> const& data_locs,
  int dst)
{
  vector<_dataloc_info_t> items;
  for(auto const& [id, memsto]: data_locs) {
    items.push_back(_dataloc_info_t { id, memsto });
  }
  mpi->send_vector(items, dst);
}

void _recv_data_locs(
  mpi_t* mpi,
  map<int, memstoloc_t>& data_locs,
  int src)
{
  auto xs = mpi->recv_vector<_dataloc_info_t>(src);
  for(auto const& [id, memsto]: xs) {
    memstoloc_t memstoloc;
    if(memsto.is_mem()) {
      memstoloc = memstoloc_t(memsto.get_mem().as_memloc(src));
    } else {
      int const& sto_id = memsto.get_sto();
      memstoloc = memstoloc_t(stoloc_t { src, sto_id });
    }
    data_locs.insert({ id, memstoloc });
  }
}

void _recv_data_locs(
  mpi_t* mpi,
  map<int, memsto_t>& data_locs,
  int src)
{
  auto xs = mpi->recv_vector<_dataloc_info_t>(src);
  for(auto const& [id, memsto]: xs) {
    data_locs.insert({id, memsto});
  }
}

void repartition(
  mpi_t* mpi,
  remap_relations_t const& remap_relations,
  allocator_settings_t const& alloc_settings,
  map<int, memsto_t>& data_locs,
  buffer_t& mem,
  storage_t& storage)
{
  int this_rank  = bool(mpi) ? mpi->this_rank  : 0;
  int world_size = bool(mpi) ? mpi->world_size : 1;

  if(this_rank == 0) {
    auto [remap_gid, g] = create_remap_graph_constructor(remap_relations);

    auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
      g.graph, g.get_placements());

    vector<uint64_t> mem_sizes;
    mem_sizes.push_back(mem->size);
    for(int src = 1; src != world_size; ++src) {
      vector<uint64_t> singleton = mpi->recv_vector<uint64_t>(src);
      mem_sizes.push_back(singleton[0]);
    }

    map<int, memstoloc_t> full_data_locs;
    for(int src = 1; src != world_size; ++src) {
      _recv_data_locs(mpi, full_data_locs, src);
    }

    // before: full_data_locs is with respect to the remap inn tids
    _update_map_with_new_tg_inns(
      full_data_locs, remap_gid, gid_to_inn, remap_relations, std::nullopt);
    // after: full_data_locs is with respect to the tasgkraph inns

    vector<int> which_storage(world_size);
    std::iota(which_storage.begin(), which_storage.end(), 0);

    auto [inn_tg_to_loc, out_tg_to_loc, memgraph] =
      memgraph_t::make(
        taskgraph, which_storage, mem_sizes,
        full_data_locs, alloc_settings, true);

    // memgraph now uses wtvr storage ids it chooses... So for each input,
    // figure out what the remap is
    vector<vector<std::array<int, 2>>> storage_remaps(world_size);
    for(auto const& [id, mg_memstoloc]: inn_tg_to_loc) {
      if(mg_memstoloc.is_stoloc()) {
        auto const& [loc, new_sto_id] = mg_memstoloc.get_stoloc();
        auto const& [_, old_sto_id] = full_data_locs.at(id).get_stoloc();
        storage_remaps[loc].push_back({new_sto_id, old_sto_id});
      }
    }

    // not needed anymore because all the info is in out_tg_to_loc
    full_data_locs.clear();

    for(int dst = 1; dst != world_size; ++dst) {
      mpi->send_vector(storage_remaps[dst], dst);
    }

    // remap all the storage ids so the refer to wtvr the memgraph wants.
    // Also, any input not in storage_remaps will get implicitly delteted
    storage.remap(storage_remaps[0]);

    for(int dst = 1; dst != world_size; ++dst) {
      mpi->send_str(memgraph.to_wire(), dst);
    }

    {
      auto exec_sts = execute_memgraph_settings_t::default_settings();
      kernel_manager_t kernel_manager;
      execute_memgraph(memgraph, exec_sts, kernel_manager, mpi, mem, storage);
    }

    _update_map_with_new_tg_outs(
      out_tg_to_loc, remap_gid, gid_to_out, remap_relations, std::nullopt);

    for(int dst = 1; dst != world_size; ++dst) {
      _send_data_locs(mpi, out_tg_to_loc, dst);
    }

    data_locs.clear();
    for(auto const& [id, memstoloc]: out_tg_to_loc) {
      if(memstoloc.is_memloc()) {
        auto const& memloc = memstoloc.get_memloc();
        if(memloc.loc == 0) {
          data_locs.insert({ id, memloc.as_memsto() });
        }
      } else {
        auto const& stoloc = memstoloc.get_stoloc();
        if(stoloc.loc == 0) {
          data_locs.insert({ id, stoloc.as_memsto() });
        }
      }
    }
  } else {
    // send the mem sizes
    {
      vector<uint64_t> singleton{ mem->size };
      mpi->send_vector(singleton, 0);
    }

    // send map<int, memsto_t>
    _send_data_locs(mpi, data_locs, 0);

    // recv the storage map and remap the storage
    {
      auto remap = mpi->recv_vector<std::array<int, 2>>(0);
      storage.remap(remap);
    }

    // recv and execute the memgraph
    {
      auto memgraph = memgraph_t::from_wire(mpi->recv_str(0));
      auto exec_sts = execute_memgraph_settings_t::default_settings();
      kernel_manager_t kernel_manager;
      execute_memgraph(memgraph, exec_sts, kernel_manager, mpi, mem, storage);
    }

    // recv data_locs here
    data_locs.clear();
    _recv_data_locs(mpi, data_locs, 0);
  }
}

