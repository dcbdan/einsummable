#include "mg_server.h"

#include "../../engine/exec_graph.h"
#include "../../engine/exec_state.h"

#include "../../engine/managers.h"
#include "../../engine/channel_manager.h"
#include "../../engine/notifier.h"

#include "../../engine/cpu/mg/storage_manager.h"
#include "../../engine/cpu/mg/workspace_manager.h"

void cpu_mg_server_t::execute_memgraph(
  memgraph_t const& memgraph,
  bool for_remap,
  map<string, scalar_t> const& scalar_vars)
{
  int n_threads = threadpool.num_runners();
  if(n_threads == 1) {
    throw std::runtime_error("must have more than one thread in the threadpool");
  }

  int this_rank = comm.get_this_rank();

  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(
      memgraph,
      this_rank,
      kernel_executor,
      num_channels_per_move,
      scalar_vars);

  rm_ptr_t rcm_ptr(new recv_channel_manager_t(comm));
  recv_channel_manager_t& rcm = *static_cast<recv_channel_manager_t*>(rcm_ptr.get());

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(mem->raw())),
      rm_ptr_t(new cpu_storage_manager_t(&storage)),
      rm_ptr_t(new notifier_t(comm, rcm)),
      rm_ptr_t(new send_channel_manager_t(comm, n_threads-1)),
      rcm_ptr,
      rm_ptr_t(new threadpool_manager_t(threadpool)),
    }
  ));

  exec_state_t state(graph, resource_manager, priority_type, this_rank);

  if(for_remap) {
    if(this_rank == 0) {
      //gremlin_t gremlin("execute_memgraph remap or move inputs loop time");
      state.event_loop();
    } else {
      state.event_loop();
    }
  } else {
    if(this_rank == 0) {
      //gremlin_t gremlin("execute_memgraph event loop time");
      state.event_loop();
    } else {
      state.event_loop();
    }
  }
}

buffer_t cpu_mg_server_t::local_copy_data(int tid) {
  memsto_t const& d = data_locs.at(tid);

  if(d.is_mem()) {
    auto const& [offset, size] = d.get_mem();
    buffer_t ret = make_buffer(size);

    std::copy(
      mem->data + offset,
      mem->data + (offset + size),
      ret->data);

    return ret;
  } else if(d.is_sto()) {
    int const& id = d.get_sto();
    return storage.read(id);
  } else {
    throw std::runtime_error("should not happen");
  }
}

struct id_memsto_t {
  int id;
  memsto_t memsto;
};

// Gathering information about tensor location and id on all compute nodes (including server itself)
server_mg_base_t::make_mg_info_t
cpu_mg_server_t::recv_make_mg_info() {
  auto fix = [](memsto_t const& memsto, int loc) {
    if(memsto.is_mem()) {
      return memstoloc_t(memsto.get_mem().as_memloc(loc));
    } else {
      int const& sto_id = memsto.get_sto();
      return memstoloc_t(stoloc_t { loc, sto_id });
    }
  };

  int world_size = comm.get_world_size();

  make_mg_info_t ret;

  ret.which_storage.resize(world_size);
  std::iota(ret.which_storage.begin(), ret.which_storage.end(), 0);

  ret.mem_sizes.push_back(mem->size);
  for(auto const& [id, memsto]: data_locs) {
    ret.data_locs.insert({id, fix(memsto, 0)});
  }

  //if I want to make it not distributed, I need to delete this following part.
  for(int src = 1; src != world_size; ++src) {
    ret.mem_sizes.push_back(comm.recv_contig_obj<uint64_t>(src));
    for(auto const& [id, memsto]: comm.recv_vector<id_memsto_t>(src)) {
      ret.data_locs.insert({id, fix(memsto, src)});
    }
  }

  return ret;
}

void cpu_mg_server_t::send_make_mg_info() {
  comm.send_contig_obj(0, mem->size);

  vector<id_memsto_t> ds;
  ds.reserve(data_locs.size());
  for(auto const& [id, memsto]: data_locs) {
    ds.push_back(id_memsto_t { .id = id, .memsto = memsto });
  }

  comm.send_vector(0, ds);
}

void cpu_mg_server_t::storage_remap_server(
  vector<vector<std::array<int, 2>>> const& remaps)
{
  int world_size = comm.get_world_size();

  storage.remap(remaps[0]);
  for(int dst = 1; dst != world_size; ++dst) {
    comm.send_vector(dst, remaps[dst]);
  }
}

void cpu_mg_server_t::storage_remap_client() {
  auto remap = comm.recv_vector<std::array<int, 2>>(0);
  storage.remap(remap);
}

void cpu_mg_server_t::rewrite_data_locs_server(
  map<int, memstoloc_t> const& new_data_locs)
{
  auto get_loc = [](memstoloc_t const& x) {
    if(x.is_memloc()) {
      return x.get_memloc().loc;
    } else {
      // Note: stoloc_t::loc is a storage location, but we are
      //       assuming that storage loc == compute-node == compute loc
      //       here
      return x.get_stoloc().loc;
    }
  };

  int world_size = comm.get_world_size();
  int this_rank = 0;

  vector<vector<id_memsto_t>> items(world_size);
  data_locs.clear();
  for(auto const& [id, memstoloc]: new_data_locs) {
    int loc = get_loc(memstoloc);
    if(loc == this_rank) {
      data_locs.insert({ id, memstoloc.as_memsto() });
    } else {
      items[loc].push_back(id_memsto_t { id, memstoloc.as_memsto() });
    }
  }

  for(int dst = 0; dst != world_size; ++dst) {
    if(dst != this_rank) {
      comm.send_vector(dst, items[dst]);
    }
  }
}

void cpu_mg_server_t::rewrite_data_locs_client() {
  data_locs.clear();
  auto new_data_locs = comm.recv_vector<id_memsto_t>(0);
  for(auto const& [id, memsto]: new_data_locs) {
    data_locs.insert({id, memsto});
  }
}

int cpu_mg_server_t::local_get_max_tid() const {
  return data_locs.size() == 0 ? -1 : data_locs.rbegin()->first;
}
int cpu_mg_server_t::local_candidate_location() const {
  return comm.get_this_rank();
}
int cpu_mg_server_t::loc_to_compute_node(int loc) const {
  // loc == compute node for the cpu mg server
  return loc;
}

void cpu_mg_server_t::local_insert_tensors(
  map<int, tuple<int, buffer_t>> data)
{
  int this_rank = comm.get_this_rank();

  // create an allocator and fill it in
  allocator_t allocator(mem->size, alloc_settings);

  for(auto const& [id, memsto]: data_locs) {
    if(memsto.is_mem()) {
      mem_t const& mem = memsto.get_mem();
      allocator.allocate_at_without_deps(mem.offset, mem.size);
    }
  }

  for(auto const& [tid, loc_tensor]: data) {
    auto const& [loc, tensor] = loc_tensor;
    if(loc != this_rank) {
      throw std::runtime_error("incorrect loc for this tensor");
    }

    memsto_t memsto;

    auto maybe_offset = allocator.try_to_allocate_without_deps(tensor->size);

    // 1. copy the data over
    // 2. fill out memsto with where the copied data is
    if(maybe_offset) {
      auto const& offset = maybe_offset.value();

      std::copy(
        tensor->data,
        tensor->data + tensor->size,
        mem->data + offset);

      memsto = memsto_t(mem_t {
        .offset = offset,
        .size = tensor->size
      });
    } else {
      int id = 1 + storage.get_max_id();

      storage.write(tensor, id);

      memsto = memsto_t(id);
    }

    auto [_, did_insert] = data_locs.insert({tid, memsto});
    if(!did_insert) {
      throw std::runtime_error("this tid is already in data locs");
    }
  }
}

void cpu_mg_server_t::local_erase_tensors(vector<int> const& tids) {
  for(auto const& tid: tids) {
    auto iter = data_locs.find(tid);
    if(iter == data_locs.end()) {
      throw std::runtime_error("no tid to delete");
    }

    memsto_t const& memsto = iter->second;
    if(memsto.is_mem()) {
      // nothing to do
    } else {
      int const& id = memsto.get_sto();
      storage.remove(id);
    }

    data_locs.erase(iter);
  }
}

void cpu_mg_server_t::print() {
  for(auto const& [tid, loc]: data_locs) {
    DOUT(tid << ": " << dbuffer_t(default_dtype(), local_copy_data(tid)));
  }
}


void server_cpu_mg_local_t::execute_graph(
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<string, scalar_t> const& scalar_vars)
{
  //define a lambda function used below to make a relation_t object
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

  //TODO: shouldn't have any moves or num_msgs but put here for now
  int num_msgs = 0;
  uint64_t num_bytes = 0;
  for(auto const& node: taskgraph.nodes) {
    if(node.op.is_move()) {
      num_msgs++;
      num_bytes += node.op.get_move().size;
    }
  }
  DOUT("executing taskgraph with " << num_msgs << " moves, " << num_bytes << " bytes moved");

  //{
  //  std::ofstream f("tg.gv");
  //  taskgraph.print_graphviz(f);
  //  DOUT("printed tg.gv");
  //}

  // inn_g_to_t is input id to taskid in taskgraph 
  remap_relations_t r;
  for(auto const& [gid, dst_tids]: inn_g_to_t) {
    //map the previous gid,relation to new gid,relation after we make taskgraph
    r.insert(
      get_relation(gid),             // src relation
      make_relation(gid, dst_tids)   // dst relation
    );
  }
  //previously we have remap(r) called. But I don't want the broadcast function inside remap, so just call remap_server
  remap_server(r);

  execute(taskgraph, scalar_vars);

  gid_map.clear();
  for(auto const& [gid, tids]: out_g_to_t) {
    gid_map.insert({gid, make_relation(gid, tids)});
  }
}

void server_cpu_mg_local_t::remap_server(remap_relations_t const& remap_relations)
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
    create_storage_remaps(full_data_locs, inn_tg_to_loc);

  // not needed anymore because all the info is in out_tg_to_loc
  full_data_locs.clear();

  storage_remap_server(storage_remaps);

  execute_memgraph(memgraph, true);

  _update_map_with_new_tg_outs(
    out_tg_to_loc, remap_gid, gid_to_out, remap_relations, std::nullopt);

  rewrite_data_locs_server(out_tg_to_loc);
}


// Gathering information about tensor location and id on all compute nodes (including server itself)
// Because this exist inside server_cpu_mg_local_t, the return make_mg_info_t only has one out of each list (rank 0)
server_cpu_mg_local_t::make_mg_info_t
server_cpu_mg_local_t::recv_make_mg_info() {
  auto fix = [](memsto_t const& memsto, int loc) {
    if(memsto.is_mem()) {
      return memstoloc_t(memsto.get_mem().as_memloc(loc));
    } else {
      int const& sto_id = memsto.get_sto();
      return memstoloc_t(stoloc_t { loc, sto_id });
    }
  };
  make_mg_info_t ret;

  ret.which_storage.resize(1);
  std::iota(ret.which_storage.begin(), ret.which_storage.end(), 0);

  ret.mem_sizes.push_back(mem->size);
  for(auto const& [id, memsto]: data_locs) {
    ret.data_locs.insert({id, fix(memsto, 0)});
  }

  return ret;
}

static vector<vector<std::array<int, 2>>>
server_cpu_mg_local_t::create_storage_remaps(
  map<int, memstoloc_t> const& full_data_locs,
  map<int, memstoloc_t> const& inn_tg_to_loc)
{
  vector<vector<std::array<int, 2>>> storage_remaps(1);
  auto const& [id, mg_memstoloc] = inn_tg_to_loc;
  if(mg_memstoloc.is_stoloc()) {
    auto const& [loc, new_sto_id] = mg_memstoloc.get_stoloc();
    auto const& [_, old_sto_id] = full_data_locs.at(id).get_stoloc();
    storage_remaps[loc].push_back({new_sto_id, old_sto_id});
  }

  return storage_remaps;
}

void server_cpu_mg_local_t::storage_remap_server(
  vector<vector<std::array<int, 2>>> const& remaps)
{
  storage.remap(remaps[0]);
}



void server_cpu_mg_local_t::execute_memgraph(
  memgraph_t const& memgraph,
  bool for_remap,
  map<string, scalar_t> const& scalar_vars)
{
  int n_threads = threadpool.num_runners();
  if(n_threads == 1) {
    throw std::runtime_error("must have more than one thread in the threadpool");
  }

  int this_rank = 0;

  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(
      memgraph,
      this_rank,
      kernel_executor,
      num_channels_per_move,
      scalar_vars);

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(mem->raw())),
      rm_ptr_t(new cpu_storage_manager_t(&storage)),
      rm_ptr_t(new threadpool_manager_t(threadpool)),
    }
  ));

  exec_state_t state(graph, resource_manager, priority_type, this_rank);

  if(for_remap) {
    if(this_rank == 0) {
      //gremlin_t gremlin("execute_memgraph remap or move inputs loop time");
      state.event_loop();
    } else {
      state.event_loop();
    }
  } else {
    if(this_rank == 0) {
      //gremlin_t gremlin("execute_memgraph event loop time");
      state.event_loop();
    } else {
      state.event_loop();
    }
  }
}