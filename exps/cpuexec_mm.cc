#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"

#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/executemg.h"
#include "../src/execution/cpu/mpi_class.h"

#include <fstream>

#define ROUT(x) if(mpi.this_rank == 0) { std::cout << "(in cpu exec) " << x << std::endl; }

void usage() {
  std::cout << "Usage: ni nj nk li lj rj rk ji jj jk oi ok\n";
}

int main(int argc, char** argv) {
  // TODO: this should have an autoplacer setup...

  bool with_taskgraph = true;
  bool with_memgraph  = true;

  if(argc != 13) {
    usage();
    return 1;
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
    return 1;
  }
  mpi_t mpi(argc, argv);

  if(mpi.this_rank == 0) {
    DOUT("ni nj nk " << (vector<uint64_t>{ni,nj,nk}));
    DOUT("li lj    " << (vector< int    >{li,lj}   ));
    DOUT("rj rk    " << (vector< int    >{rj,rk}   ));
    DOUT("ji jj jk " << (vector< int    >{ji,jj,jk}));
    DOUT("oi ok    " << (vector< int    >{oi,ok}   ));
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

  taskgraph_t taskgraph;
  if(mpi.this_rank == 0) {
    auto pls = g.get_placements();
    for(int i = 0; i != pls.size(); ++i) {
      DOUT(i << " " << pls[i].partition);
    }

    if(mpi.world_size > 1) {
      set_seed(0);
      for(auto& placement: pls) {
        for(auto& loc: placement.locations.get()) {
          loc = runif(mpi.world_size);
        }
      }
    }

    {
      auto [_0, _1, _taskgraph] = taskgraph_t::make(g.graph, pls);
      taskgraph = _taskgraph;
    }

    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed tg.gv");

    for(int r = 1; r != mpi.world_size; ++r) {
      mpi.send_str(taskgraph.to_wire(), r);
    }
  } else {
    taskgraph = taskgraph_t::from_wire(mpi.recv_str(0));
  }

  int num_threads_per_node = 1;
  int num_send_and_recv_threads = 1;

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  if(with_taskgraph) {
    execute_taskgraph_settings_t execute_settings {
      .num_apply_runner = num_threads_per_node,
      .num_send_runner  = num_send_and_recv_threads,
      .num_recv_runner  = num_send_and_recv_threads
    };

    map<int, buffer_t> tensors;
    for(int id = 0; id != taskgraph.nodes.size(); ++id) {
      auto const& node = taskgraph.nodes[id];
      if(node.op.is_input()) {
        auto const& [rank, size] = node.op.get_input();
        if(mpi.this_rank == rank) {
          buffer_t buffer = make_buffer(size);
          dbuffer_t(dtype, buffer).random("-0.0003", "0.003");
          tensors.insert({id, buffer});
        }
      }
    }

    mpi.barrier();
    gremlin_t gremlin("cpuexec_mm time (taskgraph)", mpi.this_rank != 0);
    execute_taskgraph(taskgraph, execute_settings, kernel_manager, &mpi, tensors);
    mpi.barrier();
  }

  if(with_memgraph) {
    execute_memgraph_settings_t execute_settings {
      .num_apply_runner = num_threads_per_node,
      .num_cache_runner = 0,
      .num_send_runner = num_send_and_recv_threads,
      .num_recv_runner = num_send_and_recv_threads
    };

    auto [inn_to_mem, _2, memgraph] = memgraph_t::make_without_evict(taskgraph);

    if(mpi.this_rank == 0) {
      std::ofstream f("mg.gv");
      memgraph.print_graphviz(f);
      DOUT("printed mg.gv");
    }

    buffer_t buffer = make_buffer(memgraph.mem_sizes()[mpi.this_rank]);
    storage_manager_t storage_manager("tensors.dat");

    for(auto const& [inn,mem]: inn_to_mem) {
      if(taskgraph.out_loc(inn) == mpi.this_rank) {
        buffer_t tensor_ = make_buffer_reference(buffer->data + mem.offset, mem.size);
        dbuffer_t tensor(dtype, tensor_);
        tensor.random("-0.003", "0.003");
      }
    }

    mpi.barrier();
    gremlin_t gremlin("cpuexec_mm time (memgraph)", mpi.this_rank != 0);
    execute_memgraph(memgraph, execute_settings, kernel_manager, &mpi, buffer, storage_manager);
    mpi.barrier();
  }
}
