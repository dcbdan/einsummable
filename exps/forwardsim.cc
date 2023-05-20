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

void main01(int argc, char** argv) {
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

  vector<int> locations = graph_locations_to_twolayer(graph, g_to_tl);

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

  std::cout << "Total elems:         " << total_elems << std::endl;
  std::cout << "Total flops:         " << total_flops << std::endl;
  std::cout << "Correct total elems: " << correct_total_elems << std::endl;
  std::cout << "Correct total flops: " << correct_total_flops << std::endl;

  {
    std::ofstream f("timeplot.svg");
    int row_height = 50;
    int min_box_width = 30;
    timeplot(f, boxes, row_height, min_box_width);
  }
}

void main02(int argc, char** argv) {
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

  vector<int> locations = graph_locations_to_twolayer(graph, g_to_tl);

  {
    forward_state_t sim_state(cluster, twolayer, equal_items, locations);
    decision_interface_t interface = decision_interface_t::random(np);
    double finish;
    while(!sim_state.all_done()) {
      auto [_0,finish_,_1] = sim_state.step(interface);
      finish = finish_;
    }
    std::cout << "3D Time: " << finish << std::endl;
  }

  forward_manager_t manager(cluster, twolayer, equal_items);

  manager.simulate(100, 1);
}

void main03(int argc, char** argv) {
  if(argc < 5) {
    std::cout << "usage: nlocs dn dp dd dws" << std::endl;
    return;
  }

  int nlocs;
  uint64_t dn, dp, dd;
  vector<uint64_t> dws;

  try {
    nlocs          = parse_with_ss<int>     (argv[1]);
    dn             = parse_with_ss<uint64_t>(argv[2]);
    dp             = parse_with_ss<uint64_t>(argv[3]);
    dd             = parse_with_ss<uint64_t>(argv[4]);
    for(int i = 5; i != argc; ++i) {
      dws.push_back( parse_with_ss<uint64_t>(argv[i]));
    }
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    std::cout << "usage: dn dp dd dws" << std::endl;
    return;
  }

  cluster_t cluster = make_cluster(nlocs, 10, 1);

  float learning_rate = 0.01;

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);

  auto [graph, _] = ff.mgraph.compile();
  //auto graph = three_dimensional_matrix_multiplication(
  //  4,4,4,
  //  4000,4000,4000,
  //  nlocs);

  {
    uint64_t mmlike_sizing = 1000u*1000u*1000u;
    uint64_t min_sizing = 800u*800u;
    vector<partition_t> new_partition = autopartition(
      graph,
      mmlike_sizing,
      min_sizing);
    graph.reset_annotations(new_partition);
  }

  auto [g_to_tl, equal_items, twolayer] = twolayergraph_t::make(graph);

  {
    /////////////vector<int> locations = graph_locations_to_twolayer(graph, g_to_tl);
    vector<int> locations(twolayer.joins.size(), 0);
    forward_state_t sim_state(cluster, twolayer, equal_items, locations);
    decision_interface_t interface = decision_interface_t::random(nlocs);
    double finish;
    while(!sim_state.all_done()) {
      auto [_0,finish_,_1] = sim_state.step(interface);
      finish = finish_;
    }
    std::cout << "Time all at loc 0: " << finish << std::endl;
    DOUT("Num locations to choose: " << locations.size());
  }

  forward_manager_t manager(cluster, twolayer, equal_items);

  manager.merge_line(manager.simulate_once());

  for(int i = 0; i != 10; ++i) {
    //manager.simulate(20, 1);
    manager.step(20, 0.95, 1.0);
    DOUT(manager.best_stats.makespan << "   " << manager.root->num_nodes());
    //vector<int> twolayer_locs = manager.get_best_locations();
    //DOUT(twolayer_locs);
    //set_locations_from_twolayer(graph, g_to_tl, twolayer_locs);
    //auto [_0, _1, taskgraph] = taskgraph_t::make(graph);

    //uint64_t correct_total_elems = taskgraph.total_elems_moved();
    //uint64_t correct_total_flops = taskgraph.total_flops();

    //DLINEOUT(correct_total_elems << " " << manager.best_stats.elems_total);
    //DLINEOUT(correct_total_flops << " " << manager.best_stats.flops_total);
  }
}

int main(int argc, char** argv) {
  if(argc < 5) {
    std::cout << "usage: nlocs dn dp dd dws" << std::endl;
    return 1;
  }

  int nlocs;
  uint64_t dn, dp, dd;
  vector<uint64_t> dws;

  try {
    nlocs          = parse_with_ss<int>     (argv[1]);
    dn             = parse_with_ss<uint64_t>(argv[2]);
    dp             = parse_with_ss<uint64_t>(argv[3]);
    dd             = parse_with_ss<uint64_t>(argv[4]);
    for(int i = 5; i != argc; ++i) {
      dws.push_back( parse_with_ss<uint64_t>(argv[i]));
    }
  } catch(...) {
    std::cout << "Parse error." << std::endl << std::endl;
    std::cout << "usage: dn dp dd dws" << std::endl;
    return 1;
  }

  cluster_t cluster = make_cluster(nlocs, 10, 1);

  float learning_rate = 0.01;

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);

  auto [graph, _] = ff.mgraph.compile();
  //auto graph = three_dimensional_matrix_multiplication(
  //  2,2,2,
  //  4000,4000,4000,
  //  nlocs);

  {
    uint64_t mmlike_sizing = 1000u*1000u*1000u;
    uint64_t min_sizing = 800u*800u;
    vector<partition_t> new_partition = autopartition(
      graph,
      mmlike_sizing,
      min_sizing);
    graph.reset_annotations(new_partition);
  }

  auto [g_to_tl, equal_items, twolayer] = twolayergraph_t::make(graph);
  DOUT("Number of input joins: " << twolayer.num_input_joins());

  {
    //vector<int> locations = graph_locations_to_twolayer(graph, g_to_tl);
    vector<int> locations(twolayer.joins.size(), 0);
    forward_state_t sim_state(cluster, twolayer, equal_items, locations);
    decision_interface_t interface = decision_interface_t::random(nlocs);
    double finish;
    while(!sim_state.all_done()) {
      auto [_0,finish_,_1] = sim_state.step(interface);
      finish = finish_;
    }
    std::cout << "Time all at loc 0: " << finish << std::endl;
    //std::cout << "Time 3D: " << finish << std::endl;
    DOUT("Num locations to choose: " << locations.size());
  }

  vector<int> fixed_locations(twolayer.joins.size(), -1);
  for(int jid = 0; jid != twolayer.joins.size(); ++jid) {
    auto const& join = twolayer.joins[jid];
    if(join.deps.size() == 0) {
      //fixed_locations[jid] = runif(nlocs);
    }
  }

  forward_mcts_tree_t mcts(cluster, twolayer, equal_items, fixed_locations);

  DOUT("............");
  bool fini = false;
  for(int i = 0; i != 10 && !fini; ++i) {
    for(int j = 0; j != 10 && !fini; ++j) {
      optional<int> mcts_leaf = mcts.selection();
      if(mcts_leaf) {
        mcts.expand_simulate_backprop(mcts_leaf.value());
      } else {
        fini = true;
        DOUT("fini!");
      }
    }
    DOUT(mcts.best.value().makespan);
  }
  for(int i = 0; i != 40 && !fini; ++i) {
    for(int j = 0; j != 1000 && !fini; ++j) {
      optional<int> mcts_leaf = mcts.selection();
      if(mcts_leaf) {
        mcts.expand_simulate_backprop(mcts_leaf.value());
      } else {
        fini = true;
        DOUT("fini!");
      }
    }
    DOUT(mcts.best.value().makespan);
  }

  google::protobuf::ShutdownProtobufLibrary();
}



