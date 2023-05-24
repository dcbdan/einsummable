#include "../src/einsummable/forwardsim.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/timeplot.h"

#include "../src/matrixgraph/ff.h"

#include <fstream>

cluster_t make_cluster(int nlocs, uint64_t compute_score = 1, uint64_t communicate_score = 1) {
  using device_t = cluster_t::device_t;
  using connection_t = cluster_t::connection_t;

  uint64_t giga = 1e9;

  uint64_t compute_on_device = 100 * compute_score * giga;
  uint64_t bandwidth_between_device = communicate_score * giga;

  int capacity = 1; // all kernels have a utilization of 1 for now,
                    // so  give all devices a capacity of 1

  vector<device_t> devices;
  for(int loc = 0; loc != nlocs; ++loc) {
    devices.push_back(device_t {
      .compute = compute_on_device / capacity,
      .capacity = capacity
    });
  }

  vector<connection_t> connections;
  for(int i = 0; i != nlocs; ++i) {
  for(int j = 0; j != nlocs; ++j) {
    if(i != j) {
      connections.push_back(connection_t {
        .bandwidth = bandwidth_between_device,
        .src = i,
        .dst = j
      });
    }
  }}

  return cluster_t::make(devices, connections);
}

void random_walk_through(graph_t const& graph, cluster_t cluster) {
  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph);
    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  forward_state_t state(cluster, graph);

  using jid_t = forward_state_t::jid_t;
  using rid_t = forward_state_t::rid_t;

  std::function<partition_t(int)> get_partition = [&](int gid)
  {
    return graph.nodes[gid].placement.partition;
  };

  std::function<int(jid_t)> get_location = [&](jid_t jid)
  {
    auto const& [gid,bid] = jid;
    return graph.nodes[gid].placement.locations.get()[bid];
  };

  auto settings = state.random_step_settings(get_partition, get_location);

  DOUT("-----------------------------------------");
  vector<timeplot_ns::box_t> boxes;
  double makespan;
  uint64_t total_elems = 0;
  uint64_t total_flops = 0;
  while(!state.all_done()) {
    auto maybe_completed = state.random_step(settings);
    if(!maybe_completed) {
      continue;
    }

    auto const& completed = maybe_completed.value();
    // you could have (10.5, 11.5) and
    //                (10.9, 11.1)
    // be the last two objects but
    //   makespan = completed.finish
    // would have you believe that 11.1 is the makespan
    makespan = std::max(makespan, completed.finish);
    if(completed.did_apply()) {
      auto const& [loc,gid,bid,flops] = completed.get_apply_info();
      total_flops += flops;
      boxes.push_back(timeplot_ns::box_t {
        .row = loc,
        .start = completed.start,
        .stop  = completed.finish,
        .text = write_with_ss(jid_t { gid, bid })
      });
    } else {
      total_elems += completed.get_move_info().size;
    }
  }

  DOUT("Finished in " << makespan);
  {
    std::ofstream f("tp.svg");
    timeplot(f, boxes, 50, 50, makespan);
    DOUT("Printed to tp.svg");
  }

  std::cout << "Total elems:         " << total_elems << std::endl;
  std::cout << "Correct total elems: " << correct_total_elems << std::endl;
  std::cout << std::endl;
  std::cout << "Total flops:         " << total_flops << std::endl;
  std::cout << "Correct total flops: " << correct_total_flops << std::endl;
}

void main01() {
  // 1. create a 3d matmul graph
  // 2. add all the partitions and the corresponding locations
  // 3. while not done,
  //      enqueue all work
  //      pop work
  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 100, 1);

  auto graph = three_dimensional_matrix_multiplication(
    3,8,5,
    4000,4000,4000,
    nlocs);

  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph);
    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  forward_state_t state(cluster, graph);

  using jid_t = forward_state_t::jid_t;
  using rid_t = forward_state_t::rid_t;

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    state.assign_partition(gid, node.placement.partition);
    vector<int> const& locs = node.placement.locations.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      int const& loc = locs[bid];
      state.assign_location(jid_t{gid, bid}, loc);
    }
  }

  {
    std::ofstream f("tl.gv");
    state.print_twolayer_graphviz(f);
    DOUT("Printed to tl.gv");
  }
  DOUT("-----------------------------------------");
  vector<timeplot_ns::box_t> boxes;
  double makespan;
  uint64_t total_elems = 0;
  uint64_t total_flops = 0;
  while(!state.all_done()) {
    state.enqueue_all();
    auto completed = state.pop_work();
    makespan = std::max(makespan, completed.finish);

    if(completed.did_apply()) {
      auto const& [loc,gid,bid,flops] = completed.get_apply_info();
      total_flops += flops;
      boxes.push_back(timeplot_ns::box_t {
        .row = loc,
        .start = completed.start,
        .stop  = completed.finish,
        .text = write_with_ss(jid_t { gid, bid })
      });
    } else {
      total_elems += completed.get_move_info().size;
    }
  }
  DOUT("Finished in " << makespan);
  {
    std::ofstream f("tp.svg");
    timeplot(f, boxes, 50, 50, makespan);
    DOUT("Printed to tp.svg");
  }

  std::cout << "Total elems:         " << total_elems << std::endl;
  std::cout << "Correct total elems: " << correct_total_elems << std::endl;
  std::cout << std::endl;
  std::cout << "Total flops:         " << total_flops << std::endl;
  std::cout << "Correct total flops: " << correct_total_flops << std::endl;
}

int main() {
  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 100, 1);

  auto graph = three_dimensional_matrix_multiplication(
    2,2,2,
    4000,4000,4000,
    nlocs);

  set_seed(0);

  random_walk_through(graph, cluster);
}
