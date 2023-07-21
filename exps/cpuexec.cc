#include "../src/einsummable/taskgraph.h"

#include "../src/autoplace/fsmcmc.h"

#include "../src/execution/cpu/executetg.h"
#include "../src/execution/cpu/mpi_class.h"

#include <thread>
#include <mutex>
#include <condition_variable>

struct cluster_settings_t {
  int num_nodes;
  int num_threads_per_node;
  uint64_t compute_per_thread;
  uint64_t bandwidth;
};

struct autoplace_settings_t {
  int num_steps;
  vector<double> betas;
  bool do_balanced;
  bool do_singleloc;
};

cluster_t make_cluster(cluster_settings_t settings)
{
  int      const& num_nodes            = settings.num_nodes;
  int      const& num_threads_per_node = settings.num_threads_per_node;
  uint64_t const& compute_per_thread   = settings.compute_per_thread;
  uint64_t const& bandwidth            = settings.bandwidth;

  using device_t = cluster_t::device_t;
  using connection_t = cluster_t::connection_t;

  // all workers compute kernels single threaded
  auto f_cost = [compute_per_thread](einsummable_t const& e) {
    uint64_t flops = product(e.join_shape);
    double time = (1.0 / compute_per_thread) * flops;
    return tuple<int,double>{1, time};
  };

  vector<device_t> devices(num_nodes, device_t(
    num_threads_per_node,
    f_cost));

  vector<connection_t> connections;
  for(int i = 0; i != num_nodes; ++i) {
  for(int j = 0; j != num_nodes; ++j) {
    if(i != j) {
      connections.push_back(connection_t {
        .bandwidth = bandwidth,
        .src = i,
        .dst = j
      });
    }
  }}

  return cluster_t::make(devices, connections);
}

graph_t matmul_graph(uint64_t n)
{
  einsummable_t e = einsummable_t::from_matmul(n,n,n);

  graph_t ret;
  int lhs  = ret.insert_input({n,n});
  int rhs  = ret.insert_input({n,n});
  int join = ret.insert_einsummable(e, {lhs,rhs});
  ret.insert_formation(join, true);

  return ret;
}

graph_t seven_matmul_graph(uint64_t n)
{
  einsummable_t e = einsummable_t::from_matmul(n,n,n);

  graph_t ret;

  int x0 = ret.insert_input({n,n});
  int x1 = ret.insert_input({n,n});
  int x2 = ret.insert_input({n,n});
  int x3 = ret.insert_input({n,n});
  int x4 = ret.insert_input({n,n});
  int x5 = ret.insert_input({n,n});
  int x6 = ret.insert_input({n,n});
  int x7 = ret.insert_input({n,n});

  int jy0 = ret.insert_einsummable(e, {x0,x1});
  int jy1 = ret.insert_einsummable(e, {x2,x3});
  int jy2 = ret.insert_einsummable(e, {x4,x5});
  int jy3 = ret.insert_einsummable(e, {x6,x7});

  int y0 = ret.insert_formation(jy0, false);
  int y1 = ret.insert_formation(jy1, false);
  int y2 = ret.insert_formation(jy2, false);
  int y3 = ret.insert_formation(jy3, false);

  int jz0 = ret.insert_einsummable(e, {y0,y1});
  int jz1 = ret.insert_einsummable(e, {y2,y3});

  int z0 = ret.insert_formation(jz0, false);
  int z1 = ret.insert_formation(jz1, false);

  int jw0 = ret.insert_einsummable(e, {z0,z1});

  int w0 = ret.insert_formation(jw0, true);

  return ret;
}

struct run_mcmc_t {
  std::mutex m;
  optional<tuple<double, vector<placement_t>>> best_option;

  void operator()(forwardsim_mcmc_t && mcmc, int num_steps) {
    for(int i = 0; i != num_steps; ++i) {
      mcmc.step();
    }

    std::unique_lock lk(m);
    if(!best_option) {
      best_option = {mcmc.best_makespan, mcmc.best_placements};
    } else {
      auto const& [best_makespan, best_placements] = best_option.value();
      if(mcmc.best_makespan < best_makespan) {
        best_option = {mcmc.best_makespan, mcmc.best_placements};
      }
    }
  }
};

taskgraph_t solve(
  graph_t const& graph,
  cluster_settings_t cluster_settings,
  autoplace_settings_t autoplace_settings)
{
  cluster_t cluster = make_cluster(cluster_settings);

  run_mcmc_t runner;

  vector<std::thread> threads;
  for(auto const& beta: autoplace_settings.betas) {
    if(autoplace_settings.do_balanced) {
      threads.emplace_back([&]() {
        runner(
          forwardsim_mcmc_t::init_balanced(cluster, graph, beta),
          autoplace_settings.num_steps);
      });
    }
    if(autoplace_settings.do_singleloc) {
      threads.emplace_back([&]() {
        runner(
          forwardsim_mcmc_t::init_with_single_loc(cluster, graph, beta),
          autoplace_settings.num_steps);
      });
    }
  }

  for(std::thread& thread: threads) {
    thread.join();
  }

  auto const& [_0, pls] = runner.best_option.value();
  auto [_1, _2, taskgraph] = taskgraph_t::make(graph, pls);

  return taskgraph;
}

int main(int argc, char** argv)
{
  set_seed(0);

  mpi_t mpi(argc, argv);

  int num_threads_per_node = 4;

  execute_taskgraph_settings_t execute_settings {
    .num_apply_runner = num_threads_per_node,
    .num_send_runner  = 1,
    .num_recv_runner  = 1
  };

  taskgraph_t taskgraph;

  // Assumption: all input tensors have the default dtype
  dtype_t dtype = default_dtype();

  if(mpi.this_rank == 0) {
    uint64_t giga = 1e9;

    int num_nodes = mpi.world_size;

    cluster_settings_t cluster_settings {
      .num_nodes = num_nodes,
      .num_threads_per_node = num_threads_per_node,
      .compute_per_thread = 5*giga,
      .bandwidth = 1*giga
    };

    autoplace_settings_t autoplace_settings {
      .num_steps = 1000,
      .betas = {10000.0},
      .do_balanced = true,
      .do_singleloc = false
    };

    graph_t graph = seven_matmul_graph(2000);

    taskgraph = solve(graph, cluster_settings, autoplace_settings);
    string taskgraph_str = taskgraph.to_wire();

    for(int dst = 1; dst != num_nodes; ++dst) {
      mpi.send_str(taskgraph_str, dst);
    }
  } else {
    string taskgraph_str = mpi.recv_str(0);
    taskgraph = taskgraph_t::from_wire(taskgraph_str);
  }

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

  kernel_manager_t kernel_manager = make_kernel_manager(taskgraph);

  {
    mpi.barrier();
    raii_print_time_elapsed_t gremlin("cpuexec time");
    execute_taskgraph(taskgraph, execute_settings, kernel_manager, &mpi, tensors);
    mpi.barrier();
  }
}
