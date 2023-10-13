#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/managers.h"
#include "../src/engine/communicator.h"
#include "../src/engine/channel_manager.h"
#include "../src/engine/notifier.h"

#include "../src/engine/cpu/storage_manager.h"
#include "../src/engine/cpu/workspace_manager.h"

#include "../src/einsummable/dbuffer.h"

#include <fstream>

void usage() {
  std::cout << "Setup usage: addr_zero is_client world_size memsize\n";
  std::cout << "Extra usage for server: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

memgraph_t build_memgraph(vector<uint64_t> mem_size, int argc, char** argv);

void execute_memgraph_cpu(
  memgraph_t const& memgraph,
  communicator_t& communicator,
  buffer_t buffer,
  cpu_storage_t& storage);

int main(int argc, char** argv) {
  if(argc < 4) {
    usage();
    throw std::runtime_error("provide addr_zero is_client world_size");
  }

  string addr_zero = parse_with_ss<string>(argv[1]);
  bool is_server = parse_with_ss<int>(argv[2]) == 0;
  int world_size = parse_with_ss<int>(argv[3]);
  communicator_t communicator(addr_zero, is_server, world_size);

  uint64_t mem_size = parse_with_ss<uint64_t>(argv[4]);

  memgraph_t memgraph;
  if(is_server) {
    vector<uint64_t> mem_sizes;
    mem_sizes.push_back(mem_size);
    for(int rank = 1; rank != world_size; ++rank) {
      mem_sizes.push_back(communicator.recv_contig_obj<uint64_t>(rank));
    }

    memgraph = build_memgraph(mem_sizes, argc - 4, argv + 4);

    communicator.broadcast_string(memgraph.to_wire());
  } else {
    communicator.send_contig_obj(0, mem_size);
    memgraph = memgraph_t::from_wire(communicator.recv_string(0));
  }

  buffer_t buffer = make_buffer(mem_size);
  cpu_storage_t storage;

  // Fill all the input data with ones
  dtype_t dtype = default_dtype();
  int this_rank = communicator.get_this_rank();
  for(int mid = 0; mid != memgraph.nodes.size(); ++mid) {
    auto const& node = memgraph.nodes[mid];
    if(!node.op.is_local_to(this_rank)) {
      continue;
    }

    if(node.op.is_inputmem()) {
      auto const& [_,offset,size] = node.op.get_inputmem();
      buffer_t ref = make_buffer_reference(buffer->data + offset, size);
      dbuffer_t tensor(dtype, ref);
      tensor.ones();
    } else if(node.op.is_inputsto()) {
      auto const& [_0,_1,sto_id,size] = node.op.get_inputsto();
      dbuffer_t tensor = make_dbuffer(dtype, size / dtype_size(dtype));
      tensor.ones();
      storage.write(tensor.data, sto_id);
    }
  }

  execute_memgraph_cpu(memgraph, communicator, buffer, storage);
}

memgraph_t build_memgraph(vector<uint64_t> mem_sizes, int argc, char** argv) {
  if(argc != 13) {
    usage();
    throw std::runtime_error("incorrect number of args");
  }

  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  try {
    ni = parse_with_ss<uint64_t>(argv[1]);
    nj = parse_with_ss<uint64_t>(argv[2]);
    nk = parse_with_ss<uint64_t>(argv[3]);

    li = parse_with_ss<int>(argv[4]);
    lj = parse_with_ss<int>(argv[5]);

    rj = parse_with_ss<int>(argv[6]);
    rk = parse_with_ss<int>(argv[7]);

    ji = parse_with_ss<int>(argv[8]);
    jj = parse_with_ss<int>(argv[9]);
    jk = parse_with_ss<int>(argv[10]);

    oi = parse_with_ss<int>(argv[11]);
    ok = parse_with_ss<int>(argv[12]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    throw std::runtime_error("incorrect number of args");
  }

  graph_constructor_t g;
  dtype_t dtype = default_dtype();

  int lhs = g.insert_input(partition_t({
    partdim_t::split(ni, li),
    partdim_t::split(nj, lj) }));
  int rhs = g.insert_input(partition_t({
    partdim_t::split(nj, rj),
    partdim_t::split(nk, rk) }));

  int join = g.insert_einsummable(
    partition_t({
      partdim_t::split(ni, ji),
      partdim_t::split(nk, jk),
      partdim_t::split(nj, jj)
    }),
    einsummable_t::from_matmul(ni, nj, nk),
    {lhs, rhs});

  int out = g.insert_formation(
    partition_t({
      partdim_t::split(ni, oi),
      partdim_t::split(nk, ok)
    }),
    join);

  auto pls = g.get_placements();
  for(int i = 0; i != pls.size(); ++i) {
    DOUT(i << " " << pls[i].partition);
  }

  // randomly assign the locations
  int world_size = mem_sizes.size();
  if(world_size > 1) {
    for(auto& pl: pls) {
      for(auto& loc: pl.locations.get()) {
        loc = runif(world_size);
      }
    }
  }

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, pls);

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  DLINEOUT("Mem sizes are " << mem_sizes);
  auto [inn_to_memstoloc, out_to_memstoloc, memgraph] =
    memgraph_t::make(
      taskgraph,
      vector<int>{},
      mem_sizes);

  {
    std::ofstream f("mg.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }

  return memgraph;
}

void execute_memgraph_cpu(
  memgraph_t const& memgraph,
  communicator_t& communicator,
  buffer_t buffer,
  cpu_storage_t& storage)
{
  cpu_kernel_executor_t executor;

  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(
      memgraph,
      communicator.get_this_rank(),
      executor);

  threadpool_t threadpool(12); // TODO: hardcoded number of threads

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(buffer->raw())),
      rm_ptr_t(new cpu_storage_manager_t(&storage)),
      rm_ptr_t(new notifier_t(communicator)),
      rm_ptr_t(new channel_manager_t(communicator)),
      rm_ptr_t(new threadpool_manager_t(threadpool))
    }
  ));

  exec_state_t state(graph, resource_manager);

  state.event_loop();
}

