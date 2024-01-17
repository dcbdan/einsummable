#include "../src/einsummable/memgraph.h"
#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/reference.h"
#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/executemg.h"
#include "../src/execution/cpu/mpi_class.h"

#define ROUT(x) if(mpi.this_rank == 0) { std::cout << "(in cpu exec) " << x << std::endl; }

void usage() {
  std::cout << "Usage: ni nj nk li lj rj rk ji jj jk oi ok cache_runner_num\n";
}


#include <fstream>

memgraph_t create_silly_evict_load(memgraph_t const& m0)
{
  memgraph_t m1(m0.num_compute_locs, m0.num_storage_locs, m0.storage_locs); 
  map<int, int> id_0_to_1;
  std::cout << "Node size: " << m0.nodes.size() << std::endl;
  for (int id = 0; id != m0.nodes.size(); ++id)
  {
    auto const& node = m0.nodes[id];
    std::cout << "M1 node size: " << m1.nodes.size() << std::endl;

    set<int> deps1;
    std::cout << "=========================BEGIN=======================" << std::endl;
    std::cout << "Node: " << id << std::endl;
    std::cout << "Dependencies: [" << std::endl;
    for (auto const& i : node.inns)
    {
      std::cout << "(" << i << ", "; 
      auto const& par = deps1.insert(id_0_to_1.at(i));
      std::cout << (*par.first) << "), ";
    }
    std::cout << "]" << std::endl;

    int id1; 
    if (node.op.is_inputmem())
    {
      /*memgraph_t::inputsto_t inputsto = {node.op.get_inputmem().loc, id, id};
      memgraph_t::op_t input_op(inputsto);
      int id0 = m1.insert(input_op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id0 << std::endl;

      memgraph_t::load_t load = {inputsto.as_stoloc(), node.op.get_inputmem().as_memloc()};
      memgraph_t::op_t load_op(load);
      deps1.insert(id0);
      id1 = m1.insert(load_op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id1 << std::endl;*/
    }
    else if(node.op.is_apply())
    {
      if (node.op.get_apply().is_einsummable())
      {
        int id0 = m1.insert(node.op, deps1);

        stoloc_t stoloc = {1, id};
        memloc_t mem = node.op.get_apply().mems[0].as_memloc(node.op.get_apply().loc);
        memgraph_t::evict_t evict = {mem, stoloc}; 
        set<int> deps2;
        deps2.insert(id0);
        int id2 = m1.insert(evict, deps2);

        memgraph_t::load_t load = {stoloc, mem};
        set<int> deps3;
        deps3.insert(id2);
        id1 = m1.insert(load, deps3);
      }
    }
    else 
    {
      id1 = m1.insert(node.op, deps1);
      std::cout << "Inserted a node to m1. Insert returned id of: " << id1 << std::endl;
    }

    std::cout << "Inserted node with id: " << id1 << std::endl;
    std::cout << "=========================END=========================" << std::endl;
    id_0_to_1.insert({id, id1});
  }

  std::cout << "M1 node size: " << m1.nodes.size() << std::endl;
  return m1;
}

int main(int argc, char** argv) {

  bool with_taskgraph = true;
  bool with_memgraph  = true;

  if(argc != 14) {
    usage();
    return 1;
  }

  uint64_t ni, nj, nk;
  int li, lj;
  int rj, rk;
  int ji, jj, jk;
  int oi, ok;
  int crn;
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

    crn = parse_with_ss<int>(argv[13]);
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
      .num_cache_runner = crn,
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
    storage_t storage_manager("tensors.dat");
    memgraph_t m1 = create_silly_evict_load(memgraph);

    {
      std::cout << "silly1.gv" << std::endl;
      std::ofstream f("silly.gv");
      m1.print_graphviz(f);
    }

    for(auto const& [inn,mem]: inn_to_mem) {
      if(taskgraph.out_loc(inn) == mpi.this_rank) {
        buffer_t tensor_ = make_buffer_reference(buffer->data + mem.offset, mem.size);
        dbuffer_t tensor(dtype, tensor_);
        tensor.random("-0.003", "0.003");
      }
    }

    mpi.barrier();
    gremlin_t gremlin("cpuexec_mm time (memgraph)", mpi.this_rank != 0);
    execute_memgraph(m1, execute_settings, kernel_manager, &mpi, buffer, storage_manager);
    mpi.barrier();
  }
}