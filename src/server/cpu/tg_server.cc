#include "tg_server.h"

#include "../../engine/exec_graph.h"
#include "../../engine/exec_state.h"

#include "../../engine/managers.h"
#include "../../engine/channel_manager.h"
#include "../../engine/notifier.h"

#include "../../engine/cpu/tg/data_manager.h"

#include "../../engine/repartition.h"

void cpu_tg_server_t::_execute_tg(taskgraph_t const& taskgraph)
{
  int n_threads = threadpool.num_runners();
  if(n_threads == 1) {
    throw std::runtime_error("must have more than one thread in the threadpool");
  }

  int this_rank = comm.get_this_rank();

  auto [graph, dinfos] = exec_graph_t::make_cpu_tg_exec_graph(
    taskgraph, this_rank, kernel_executor, num_channels_per_move);

  rm_ptr_t rcm_ptr(new recv_channel_manager_t(comm));
  recv_channel_manager_t& rcm = *static_cast<recv_channel_manager_t*>(rcm_ptr.get());

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new data_manager_t(local_data, dinfos, max_memory_usage)),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new notifier_t(comm, rcm)),
      rm_ptr_t(new send_channel_manager_t(comm, n_threads-1)),
      rcm_ptr,
      rm_ptr_t(new threadpool_manager_t(threadpool))
    }
  ));

  exec_state_t state(
    graph,
    resource_manager,
    exec_state_t::priority_t::given,
    this_rank);

  state.event_loop();
}

void cpu_tg_server_t::execute_tg_server(taskgraph_t const& taskgraph) {
  comm.broadcast_string(taskgraph.to_wire());
  _execute_tg(taskgraph);
}

void cpu_tg_server_t::execute_tg_client() {
  taskgraph_t taskgraph = taskgraph_t::from_wire(comm.recv_string(0));
  _execute_tg(taskgraph);
}

void cpu_tg_server_t::remap_server(remap_relations_t const& remap_relations) {
  comm.broadcast_string(remap_relations.to_wire());
  _remap(remap_relations);
}

void cpu_tg_server_t::remap_client() {
  remap_relations_t remap_relations =
    remap_relations_t::from_wire(comm.recv_string(0));
  _remap(remap_relations);
}

void cpu_tg_server_t::_remap(remap_relations_t const& remap_relations) {
  int this_rank = comm.get_this_rank();

  auto [remap_gid, g] = create_remap_graph_constructor(remap_relations);

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  // before: data_locs is with respect to the remap inn tids
  _update_map_with_new_tg_inns(
    local_data, remap_gid, gid_to_inn, remap_relations, this_rank);
  // after: data_locs is with respect to the tasgkraph inns

  _execute_tg(taskgraph);

  _update_map_with_new_tg_outs(
    local_data, remap_gid, gid_to_out, remap_relations, this_rank);
}

int cpu_tg_server_t::local_get_max_tid() const {
  return local_data.size() == 0 ? -1 : local_data.rbegin()->first;
}

// return a location that exists at this compute-node
int cpu_tg_server_t::local_candidate_location() const {
  return comm.get_this_rank();
}

int cpu_tg_server_t::loc_to_compute_node(int loc) const {
  // loc == compute node for the cpu tg server
  return loc;
}

buffer_t cpu_tg_server_t::local_copy_data(int tid) {
  return make_buffer_copy(local_data.at(tid));
}

void cpu_tg_server_t::local_insert_tensors(
  map<int, tuple<int, buffer_t>> data)
{
  int this_rank = comm.get_this_rank();

  for(auto const& [tid, loc_tensor]: data) {
    auto const& [loc, tensor] = loc_tensor;

    if(loc != this_rank) {
      throw std::runtime_error("incorrect loc for this tensor");
    }

    auto [_, did_insert] = local_data.insert({tid, tensor});

    if(!did_insert) {
      throw std::runtime_error("did not insert the tensor");
    }
  }
}

void cpu_tg_server_t::local_erase_tensors(vector<int> const& tids) {
  for(auto const& tid: tids) {
    local_data.erase(tid);
  }
}

