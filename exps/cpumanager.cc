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

  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("printed to tg.gv");
  }

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

    //for(auto const& [tid, b]: manager.data) {
    //  DOUT("tid " << tid << " sum " <<
    //    dbuffer_t(default_dtype(), b).sum());
    //}

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
      std::shared_ptr<memgraph_t> mg;
      {
        gremlin_t gremlin("executing the taskgraph as memgraph");
        mg = std::make_shared<memgraph_t>(manager.execute(taskgraph));
      }
      std::ofstream f("mg.gv");
      mg->print_graphviz(f);
      DOUT("printed to mg.gv");
    }

    //for(auto const& [tid, loc]: manager.data_locs) {
    //  if(loc.is_sto()) {
    //    DOUT("tid " << tid << " with sum " <<
    //      dbuffer_t(default_dtype(), manager.storage.read(loc.get_sto())).sum());
    //  } else {
    //    auto const& [offset, size] = loc.get_mem();
    //    buffer_t b = make_buffer_reference(manager.mem->data + offset, size);
    //    DOUT("tid " << tid << " with sum " <<
    //      dbuffer_t(default_dtype(), b).sum());
    //  }
    //}

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

  for(auto const& [gid, tdata]: tg_out_data) {
    auto const& mdata = mg_out_data.at(gid);
    DOUT("gid " << gid << " has sq distance: " << dbuffer_squared_distance(tdata, mdata));
    if(!is_close(tdata, mdata)) {
      //bool ee = true;
      //for(int i = 0; i != tdata.nelem(); ++i) {
      //  bool yy = (tdata.f32()[i] == mdata.f32()[i]);
      //  if(ee != yy) {
      //    int ix = i / 1000;
      //    int iy = i % 1000;
      //    DOUT("(" << ix << "," << iy << ")" << " " << std::boolalpha << yy);
      //    ee = yy;
      //  }
      //}
      DOUT("total nelem of " << tdata.nelem());
      DOUT(tdata.sum());
      DOUT(mdata.sum());
      throw std::runtime_error("not close");
    }
  }
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
  set_seed(0);

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
      d.random("-0.1", "0.1");

      data.insert({gid, d});
    }
  }

  return {graph, pls, data};
}

//int main() {
//  set_default_dtype(dtype_t::f64);
//
//  storage_t storage;
//  for(int id = 0; id != 10; ++id) {
//    dbuffer_t inn = make_dbuffer(default_dtype(), 1000*1000);
//    inn.random();
//    storage.write(inn.data, id);
//    dbuffer_t out(default_dtype(), storage.read(id));
//
//    if(!is_close(inn, out)) {
//      throw std::runtime_error("BOWAJSAAA");
//    }
//  }
//}

int main(int argc, char** argv) {
  mpi_t mpi(argc, argv);

  int num_threads = 1;

  uint64_t ngb = 1;
  uint64_t GB = 1000000000;

  uint64_t buffer_size = 7000000;

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
    .memgraph_buffer_size = buffer_size // ngb*GB
  };

  if(mpi.this_rank == 0) {
    auto [graph, placements, input_data] = mm(mpi.world_size, argc, argv);
    test_server(
      mpi, settings, graph, placements, input_data);
  } else {
    test_client(mpi, settings);
  }
}
