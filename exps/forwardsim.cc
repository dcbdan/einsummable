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

int main() {
  // 1. create a 3d matmul graph
  // 2. add all the partitions and the corresponding locations
  // 3. while not done,
  //      enqueue all work
  //      pop work
  // ?. compare to forwardsim1

  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 10, 1);

  auto graph = three_dimensional_matrix_multiplication(
    1,1,1,
    4000,4000,4000,
    nlocs);

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

  double makespan;
  while(!state.all_done()) {
    state.enqueue_all();
    makespan = state.pop_work().finish;
    DOUT(makespan);
  }
  DOUT("Finished in " << makespan);
}
