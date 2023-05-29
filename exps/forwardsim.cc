#include "../src/autoplace/forwardsim.h"
#include "../src/autoplace/autoplace.h"
#include "../src/autoplace/autopart.h"
#include "../src/einsummable/taskgraph.h"
#include "../src/base/timeplot.h"

#include "../src/matrixgraph/ff.h"

#include "../src/autoplace/mcts1.h"
#include "../src/einsummable/twolayergraph.h"

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

void random_walk_through(
  graph_t const& graph,
  vector<placement_t> const& placements,
  cluster_t cluster,
  bool random_loc)
{
  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);
    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  forward_state_t state(cluster, graph);
  int nloc = cluster.devices.size();

  using jid_t = forward_state_t::jid_t;
  using rid_t = forward_state_t::rid_t;

  std::function<partition_t(int)> get_partition = [&](int gid)
  {
    return placements[gid].partition;
  };

  std::function<int(jid_t)> get_location;
  if(random_loc) {
    get_location = [&](jid_t) { return runif(nloc); };
  } else {
    get_location = [&](jid_t jid)
    {
      auto const& [gid,bid] = jid;
      return placements[gid].locations.get()[bid];
    };
  }

  auto settings = state.random_step_settings(get_partition, get_location);
  //settings.priority_assign_partition = true;
  //settings.priority_assign_location = true;
  //settings.always_enqueue_all = true;

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
    std::ofstream f("tl.gv");
    state.print_twolayer_graphviz(f);
    DOUT("Printed to tl.gv");
  }
  {
    std::ofstream f("tp.svg");
    timeplot(f, boxes, 50, 50, makespan);
    DOUT("Printed to tp.svg");
  }

  if(!random_loc) {
    if(total_elems != correct_total_elems) {
      throw std::runtime_error("incorrect number of total_elems");
    }
  }
  if(total_flops != correct_total_flops) {
    throw std::runtime_error("incorrect number of flops");
  }
  //std::cout << "Total elems:         " << total_elems << std::endl;
  //std::cout << "Correct total elems: " << correct_total_elems << std::endl;
  //std::cout << std::endl;
  //std::cout << "Total flops:         " << total_flops << std::endl;
  //std::cout << "Correct total flops: " << correct_total_flops << std::endl;
}

void main01() {
  // 1. create a 3d matmul graph
  // 2. add all the partitions and the corresponding locations
  // 3. while not done,
  //      enqueue all work
  //      pop work
  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 100, 1);

  auto g = three_dimensional_matrix_multiplication(
    3,8,5,
    4000,4000,4000,
    nlocs);
  auto const& graph = g.graph;
  auto placements = g.get_placements();

  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(
      graph,
      placements);
    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  forward_state_t state(cluster, graph);

  using jid_t = forward_state_t::jid_t;
  using rid_t = forward_state_t::rid_t;

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    state.assign_partition(gid, placements[gid].partition);
    vector<int> const& locs = placements[gid].locations.get();
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

void main02() {
  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 10, 1);

  //auto graph = three_dimensional_matrix_multiplication(
  //  4,8,3,
  //  4000,4000,4000,
  //  nlocs);

  //bool random_loc = false;

  float learning_rate = 0.1;
  uint64_t dn = 3000;
  uint64_t dp = 1000;
  uint64_t dd = 100;
  vector<uint64_t> dws{3000,3000,3000,3000};

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);
  auto [graph, _] = ff.mgraph.compile();

  vector<placement_t> placements;
  {
    vector<partition_t> new_partition = autopartition(
      graph, nlocs, 8*nlocs);

    for(auto const& part: new_partition) {
      placements.emplace_back(part);
    }
  }

  bool random_loc = true;

  for(int i = 0; i != 100; ++i) {
    set_seed(i);
    random_walk_through(graph, placements, cluster, random_loc);
  }
}

void main03() {
  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 10, 1);

  //auto graph = three_dimensional_matrix_multiplication(
  //  4,4,4,
  //  4000,4000,4000,
  //  nlocs);

  float learning_rate = 0.1;
  uint64_t dn = 1500;
  uint64_t dp = 1500;
  uint64_t dd = 1500;
  vector<uint64_t> dws{2000,2000};

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);
  auto [graph, _] = ff.mgraph.compile();

  using namespace mcts1_ns;

  double c = 1.5;
  tree_t tree(graph, cluster, c);
  double base = tree.get_best_makespan();

  for(int i = 0; i != 4000; ++i) {
    auto [makespan,_] = tree.step();
    double speedup = base / tree.get_best_makespan();
    DOUT((tree.get_best_makespan() / base) << "             | " << (makespan / base) << "");
    //DOUT(speedup << "x");
    //DOUT(tree.size());
  }

  forward_state_t state = tree.construct_best();
  {
    {
      std::ofstream f("tl.gv");
      state.print_twolayer_graphviz(f);
      DOUT("Printed to tl.gv");
    }

    vector<timeplot_ns::box_t> boxes;
    double makespan = 0.0;
    using jid_t = forward_state_t::jid_t;
    while(!state.all_done()) {
      state.enqueue_all();
      auto completed = state.pop_work();
      makespan = std::max(makespan, completed.finish);
      if(completed.did_apply()) {
        auto const& [loc,gid,bid,flops] = completed.get_apply_info();
        boxes.push_back(timeplot_ns::box_t {
          .row = loc,
          .start = completed.start,
          .stop  = completed.finish,
          .text = write_with_ss(jid_t { gid, bid })
        });
      }
    }

    {
      std::ofstream f("tp.svg");
      timeplot(f, boxes, 50, 50, makespan);
      DOUT("Printed to tp.svg");
    }
  }

  //for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //  auto const& node = graph.nodes[gid];
  //  if(node.op.is_einsummable()) {
  //    DOUT(gid << ": " << node.op.get_einsummable());
  //  }
  //  DOUT(gid << ": " << state.get_ginfo(gid).partition.value());
  //}
  //for(int gid = 0; gid != graph.nodes.size(); ++gid) {
  //  auto const& node = graph.nodes[gid];
  //  if(node.op.is_einsummable()) {
  //    DOUT(gid << ": " << node.op.get_einsummable());
  //  } else if(node.op.is_input()) {
  //    DOUT(gid << ": input");
  //  } else if(node.op.is_formation()) {
  //    DOUT(gid << ": formation");
  //  }
  //}
}

void main05() {
  // Do markov chain monte carlo to search the space..

  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 3, 1);

  //auto graph = three_dimensional_matrix_multiplication(
  //  4,4,4,
  //  4000,4000,4000,
  //  nlocs);
  //equal_items_t<int> eqs = {};

  float learning_rate = 0.1;
  uint64_t dn = 1500;
  uint64_t dp = 1500;
  uint64_t dd = 1500;
  vector<uint64_t> dws{2000, 2000};

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);
  auto [graph, m_to_g] = ff.mgraph.compile();
  equal_items_t<int> eqs;
  for(int i = 0; i != ff.wsinn.size(); ++i) {
    auto const& inn = m_to_g.at(ff.wsinn[i]);
    auto const& out = m_to_g.at(ff.wsout[i]);
    eqs.insert(inn, out);
  }

  double base = simulate(cluster, graph, single_loc_placements(graph));

  //mcmc_t mcmc = mcmc_t::init_with_single_loc(cluster, graph, 600.1, eqs);
  mcmc_t mcmc = mcmc_t::init_balanced(cluster, graph, 600.1, eqs);

  for(int i = 0; i != 20000; ++i) {
    mcmc.step();
    if(i % 100 == 0) {
      double speedup = base / mcmc.best_makespan;
      DOUT((mcmc.best_makespan / base) << "             | "
            << (mcmc.current_makespan / base) << "");
      //DOUT( speedup << "         | " << ( base / mcmc.current_makespan ));
    }
  }

  {
    auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph, mcmc.best_placements);
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("Printed to tg.gv");
  }
}

int main() {
  main05();
}
