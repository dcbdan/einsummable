#include "../src/einsummable/graph.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/memgraph.h"

#include "../src/execution/cpu/mpi_class.h"
#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/executemg.h"

#include "../src/execution/cpu/manager.h"

struct full_settings_t {
  execute_taskgraph_settings_t exec_tg;
  execute_memgraph_settings_t exec_mg;
  allocator_settings_t alloc;
  uint64_t memgraph_buffer_size;
};

void test_server(
  mpi_t& mpi,
  full_settings_t const& settings,
  graph_t const& graph,
  vector<placement_t> const& placements,
  map<int, dbuffer_t> gid_inn_data)
{
  auto [inn_gid_to_tids, save_gid_to_tids, taskgraph] =
    taskgraph_t::make(graph, placements);

  map<int, dbuffer_t> tg_out_data;
  {
    tg_manager_t manager(&mpi, settings.exec_tg);

    // partition gid_inn_data into the manager
    for(auto [gid, dbuffer]: gid_inn_data) {
      relation_t rel {
        .dtype = dbuffer.dtype,
        .placement = placements[gid],
        .tids = inn_gid_to_tids[gid]
      };

      manager.partition_into(rel, dbuffer);
    }

    // execute the taskgraph
    manager.execute(taskgraph);

    // get all the outputs
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      auto const& node = graph.nodes[gid];
      if(node.op.is_save()) {
        relation_t rel {
          .dtype = node.op.out_dtype(),
          .placement = placements[gid],
          .tids = save_gid_to_tids[gid]
        };
        tg_out_data.insert({gid, manager.get_tensor(rel)});
      }
    }

    // shutdown so the client can stop!
    manager.shutdown();
  }

  map<int, dbuffer_t> mg_out_data;
  {
    mg_manager_t manager(&mpi, settings.exec_mg, settings.memgraph_buffer_size);

    // partition gid_inn_data into the manager
    for(auto [gid, dbuffer]: gid_inn_data) {
      relation_t rel {
        .dtype = dbuffer.dtype,
        .placement = placements[gid],
        .tids = inn_gid_to_tids[gid]
      };

      manager.partition_into(rel, dbuffer);
    }

    // unlike the taskgraph manager, kernels are not implicitly compiled
    manager.update_kernel_manager(taskgraph);

    // execute the taskgraph
    {
      gremlin_t gremlin("executing the taskgraph as memgraph");
      manager.execute(taskgraph);
    }

    // get all the outputs
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      auto const& node = graph.nodes[gid];
      if(node.op.is_save()) {
        relation_t rel {
          .dtype = node.op.out_dtype(),
          .placement = placements[gid],
          .tids = save_gid_to_tids[gid]
        };
        mg_out_data.insert({gid, manager.get_tensor(rel)});
      }
    }

    // shutdown so the client can stop!
    manager.shutdown();
  }

  // TODO: compare tg and mg out data
}

void test_client(mpi_t& mpi, full_settings_t const& settings) {
  {
    tg_manager_t tg_manager(&mpi, settings.exec_tg);
    tg_manager.listen();
  }

  {
    mg_manager_t mg_manager(&mpi, settings.exec_mg, settings.memgraph_buffer_size);
    mg_manager.listen();
  }
}

tuple<graph_t, vector<placement_t>, map<int, dbuffer_t>>
mm(int world_size, int argc, char** argv)
{
  string usage = "Usage: ni nj nk li lj rj rk ji jj jk oi ok";

  if(argc != 13) {
    throw std::runtime_error(usage);
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
    throw std::runtime_error(usage);
  }

  DOUT("ni nj nk " << (vector<uint64_t>{ni,nj,nk}));
  DOUT("li lj    " << (vector< int    >{li,lj}   ));
  DOUT("rj rk    " << (vector< int    >{rj,rk}   ));
  DOUT("ji jj jk " << (vector< int    >{ji,jj,jk}));
  DOUT("oi ok    " << (vector< int    >{oi,ok}   ));

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

  graph_t const& graph = g.graph;

  vector<placement_t> pls = g.get_placements();
  if(world_size > 1) {
    set_seed(0);
    for(auto& placement: pls) {
      for(auto& loc: placement.locations.get()) {
        loc = runif(world_size);
      }
    }
  }

  map<int, dbuffer_t> data;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& [dtype, shape] = node.op.get_input();

      dbuffer_t d = make_dbuffer(dtype, product(shape));
      d.random("-0.001", "0.001");

      data.insert({gid, d});
    }
  }

  return {graph, pls, data};
}

int main(int argc, char** argv) {
  mpi_t mpi(argc, argv);

  uint64_t ngb = 1;
  int num_threads = 8;

  uint64_t GB = 1000000000;

  full_settings_t settings {
    .exec_tg = execute_taskgraph_settings_t {
      .num_apply_runner = num_threads,
      .num_send_runner = 2,
      .num_recv_runner = 2
    },
    .exec_mg = execute_memgraph_settings_t {
      .num_apply_runner = num_threads,
      .num_storage_runner = 1,
      .num_send_runner = 2,
      .num_recv_runner = 2
    },
    .alloc = allocator_settings_t::default_settings(),
    .memgraph_buffer_size = ngb*GB
  };

  if(mpi.this_rank == 0) {
    auto [graph, placements, input_data] = mm(mpi.world_size, argc, argv);
    test_server(
      mpi, settings, graph, placements, input_data);
  } else {
    test_client(mpi, settings);
  }
}
