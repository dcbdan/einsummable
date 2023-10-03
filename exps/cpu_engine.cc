#include "../src/engine/exec_state.h"
#include "../src/engine/exec_graph.h"
#include "../src/engine/resource_manager.h"
#include "../src/engine/communicator.h"

#include "../src/engine/cpu/storage_manager.h"
#include "../src/engine/cpu/workspace_manager.h"

#include "../src/einsummable/dbuffer.h"

#include <fstream>

void usage() {
  std::cout << "Usage: memsize ni nj nk li lj rj rk ji jj jk oi ok\n";
}

void execute_memgraph_cpu(
  memgraph_t const& memgraph,
  buffer_t buffer,
  cpu_storage_t& storage);

int main(int argc, char** argv) {
  if(argc != 14) {
    usage();
    return 1;
  }

  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  uint64_t mem_size;
  try {
    mem_size = parse_with_ss<uint64_t>(argv[1]);

    ni = parse_with_ss<uint64_t>(argv[2]);
    nj = parse_with_ss<uint64_t>(argv[3]);
    nk = parse_with_ss<uint64_t>(argv[4]);

    li = parse_with_ss<int>(argv[5]);
    lj = parse_with_ss<int>(argv[6]);

    rj = parse_with_ss<int>(argv[7]);
    rk = parse_with_ss<int>(argv[8]);

    ji = parse_with_ss<int>(argv[9]);
    jj = parse_with_ss<int>(argv[10]);
    jk = parse_with_ss<int>(argv[11]);

    oi = parse_with_ss<int>(argv[12]);
    ok = parse_with_ss<int>(argv[13]);
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    usage();
    return 1;
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

  auto [_0, _1, taskgraph] = taskgraph_t::make(g.graph, pls);

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");
  }

  auto [inn_to_memstoloc, out_to_memstoloc, memgraph] =
    memgraph_t::make(
      taskgraph,
      vector<int>{},
      vector<uint64_t>{ mem_size });


  {
    std::ofstream f("mg.gv");
    memgraph.print_graphviz(f);
    DOUT("printed mg.gv");
  }

  buffer_t buffer = make_buffer(mem_size);
  cpu_storage_t storage;

  for(auto const& [inn,memstoloc]: inn_to_memstoloc) {
    uint64_t size = taskgraph.nodes[inn].op.out_size();
    dbuffer_t tensor = make_dbuffer(dtype, size / dtype_size(dtype));
    tensor.ones();

    if(memstoloc.is_memloc()) {
      auto const& memloc = memstoloc.get_memloc();
      auto const& [offset,size_,_] = memloc;
      if(size != size_) {
        throw std::runtime_error("size mismatch");
      }
      std::copy(
        tensor.data->data,
        tensor.data->data + size,
        buffer->data + offset);
    } else {
      int const& sto_id = memstoloc.get_stoloc().id;
      storage.write(tensor.data, sto_id);
    }

    DOUT(tensor);
  }

  DOUT("executing...");
  execute_memgraph_cpu(memgraph, buffer, storage);
  DOUT("executed.");

  for(auto const& [out,memstoloc]: out_to_memstoloc) {
    uint64_t size = taskgraph.nodes[out].op.out_size();
    buffer_t tensor_;
    if(memstoloc.is_memloc()) {
      auto const& memloc = memstoloc.get_memloc();
      auto const& [offset,size_,_] = memloc;
      if(size != size_) {
        throw std::runtime_error("size mismatch");
      }
      tensor_ = make_buffer_reference(buffer->data + offset, size);
    } else {
      int const& sto_id = memstoloc.get_stoloc().id;
      tensor_ = storage.read(sto_id);
    }
    dbuffer_t tensor(dtype, tensor_);
    DOUT(tensor);
  }
}

void execute_memgraph_cpu(
  memgraph_t const& memgraph,
  buffer_t buffer,
  cpu_storage_t& storage)
{
  cpu_kernel_executor_t executor;

  exec_graph_t graph =
    exec_graph_t::make_cpu_exec_graph(memgraph, 0, executor);

  rm_ptr_t resource_manager(new resource_manager_t(
    vector<rm_ptr_t> {
      rm_ptr_t(new cpu_workspace_manager_t()),
      rm_ptr_t(new cpu_storage_manager_t(&storage)),
      rm_ptr_t(new group_manager_t()),
      rm_ptr_t(new global_buffers_t(buffer->raw()))
    }
  ));

  exec_state_t state(graph, resource_manager);

  state.event_loop();
}

