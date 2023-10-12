#include "server.h"

#include "../../engine/exec_graph.h"
#include "../../engine/exec_state.h"

#include "../../engine/resource_manager.h"
#include "../../engine/channel_manager.h"
#include "../../engine/notifier.h"

#include "../../engine/cpu/storage_manager.h"
#include "../../engine/cpu/workspace_manager.h"

void cpu_mg_server_t::execute_memgraph(
  memgraph_t const& memgraph)
{
  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(
      memgraph,
      comm.get_this_rank(),
      kernel_executor);

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(mem->raw())),
      rm_ptr_t(new cpu_storage_manager_t(&storage)),
      rm_ptr_t(new notifier_t(comm)),
      rm_ptr_t(new channel_manager_t(comm))
    }
  ));

  exec_state_t state(graph, resource_manager);

  state.event_loop();
}

buffer_t cpu_mg_server_t::local_copy_data_at(memsto_t const& loc) {
  // TODO
  return make_buffer(0);
}

server_mg_base_t::make_mg_info_t
cpu_mg_server_t::recv_make_mg_info() {
  // TODO
  return make_mg_info_t {};
}

void cpu_mg_server_t::send_make_mg_info() {
  // TODO
}

void cpu_mg_server_t::storage_remap_server(
  vector<vector<std::array<int, 2>>> const& remaps)
{
  // TODO
}

void cpu_mg_server_t::storage_remap_client() {
  // TODO
}

vector<uint64_t> cpu_mg_server_t::recv_mem_sizes() {
  // TODO
  return {};
}

void cpu_mg_server_t::send_mem_sizes() {
  // TODO
}

void cpu_mg_server_t::rewrite_data_locs_server(
  map<int, memstoloc_t> const& out_tg_to_loc)
{
  // TODO
}

void cpu_mg_server_t::rewrite_data_locs_client() {
  // TODO
}

void cpu_mg_server_t::local_insert_tensors(map<int, buffer_t> data) {
  // TODO
}

void cpu_mg_server_t::local_erase_tensors(vector<int> const& tids) {
  // TODO
}

