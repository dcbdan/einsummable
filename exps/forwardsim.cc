#include "../src/einsummable/forwardsim.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/einsummable/timeplot.h"

#include <fstream>

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

  vector<int> locations = graph_locations_to_tasklayer(graph, g_to_tl);

  vector<string> colors{
    "#61B292",
    "#AED09E",
    "#F1E8A7",
    "#A8896C",
    "#A8D8EA",
    "#AA96DA",
    "#FCBAD3",
    "#FFFFD2"
  };

  {
    std::ofstream f("twolayer.gv");
    twolayer.print_graphviz(f,
      [&](int jid) {
        int const& loc = locations[jid];
        if(loc < colors.size()) {
          return colors[loc];
        } else {
          return string();
        }
      }
    );
  }

  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph);
    std::ofstream f("taskgraph.gv");
    taskgraph.print_graphviz(f, colors);

    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  decision_interface_t interface = decision_interface_t::random(np);

  forward_state_t sim_state(cluster, twolayer, equal_items, locations);

  set_seed(0);

  using box_t = timeplot_ns::box_t;
  vector<box_t> boxes;

  uint64_t total_elems = 0;
  uint64_t total_flops = 0;

  while(!sim_state.all_done()) {
    auto const& [start,stop,work_unit] = sim_state.step(interface);
    std::cout << start << "," << stop << ": ";
    if(work_unit.did_move()) {
      auto const& [src,dst,rid,uid,size] = work_unit.get_move_info();

      total_elems += size;

      std::cout << "move@" << src << "->" << dst;

      boxes.push_back(box_t {
        .row = np + cluster.to_connection.at({src,dst}),
        .start = start,
        .stop = stop,
        .text = write_with_ss(rid) + "," + write_with_ss(uid)
      });
    } else {
      auto const& [loc,jid,flops] = work_unit.get_apply_info();

      total_flops += flops;

      std::cout << "apply J" << jid << "@" << loc;

      boxes.push_back(box_t {
        .row = loc,
        .start = start,
        .stop = stop,
        .text = write_with_ss(jid)
      });
    }
    std::cout << std::endl;
  }

  std::cout << "Total elems: " << total_elems << std::endl;
  std::cout << "Total flops: " << total_flops << std::endl;

  std::cout << "Correct total elems: " << correct_total_elems << std::endl;
  std::cout << "Correct total flops: " << correct_total_flops << std::endl;

  {
    std::ofstream f("timeplot.svg");
    int row_height = 50;
    int min_box_width = 30;
    timeplot(f, boxes, row_height, min_box_width);
  }
}
