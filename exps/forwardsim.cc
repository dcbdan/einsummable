#include "../src/einsummable/forwardsim.h"

cluster_t make_cluster(int nlocs) {
  using device_t = cluster_t::device_t;
  using connection_t = cluster_t::connection_t;

  // nvidia tesla p100 9.3 Teraflops single precision
  uint64_t giga = 1e9;
  uint64_t tera = 1e12;
  uint64_t nvidia_tesla_p100 = (tera * 93) / 10;


  uint64_t compute_on_device = nvidia_tesla_p100;
  uint64_t bandwidth_between_device = 20 * giga;

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

int main(int argc, char** argv) {
  if(argc != 8) {
    throw std::runtime_error("usage: pi pj pk di dj dk nproc");
  }
  int pi = parse_with_ss<int>(argv[1]);
  int pj = parse_with_ss<int>(argv[2]);
  int pk = parse_with_ss<int>(argv[3]);

  uint64_t di = parse_with_ss<uint64_t>(argv[4]);
  uint64_t dj = parse_with_ss<uint64_t>(argv[5]);
  uint64_t dk = parse_with_ss<uint64_t>(argv[6]);

  int np = parse_with_ss<int>(argv[7]);

  cluster_t cluster = make_cluster(np);

  graph_t graph = three_dimensional_matrix_multiplication(
    pi,pj,pk,
    di,dj,dk,
    np);

  auto [g_to_tl, equal_items, twolayer] = twolayergraph_t::make(graph);

  twolayer.print_graphviz(std::cout);

  vector<int> locations = graph_locations_to_tasklayer(graph, g_to_tl);

  decision_interface_t interface = decision_interface_t::random(np);

  forward_state_t sim_state(cluster, twolayer, equal_items, locations);

  set_seed(0);

  while(!sim_state.all_done()) {
    auto const& [start,stop,work_unit] = sim_state.step(interface);
    std::cout << start << "," << stop << ": ";
    if(work_unit.did_move()) {
      auto const& [src,dst,_0,_1] = work_unit.get_move_info();
      std::cout << "move@" << src << "->" << dst;
    } else {
      auto const& [loc,jid] = work_unit.get_apply_info();
      std::cout << "apply J" << jid << "@" << loc;
    }
    std::cout << std::endl;
  }
}
