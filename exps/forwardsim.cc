#include "../src/autoplace/forwardsim.h"
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

void random_walk_through(graph_t const& graph, cluster_t cluster, bool random_loc) {
  uint64_t correct_total_elems;
  uint64_t correct_total_flops;
  {
    auto [_0, _1, taskgraph] = taskgraph_t::make(graph);
    correct_total_elems = taskgraph.total_elems_moved();
    correct_total_flops = taskgraph.total_flops();
  }

  forward_state_t state(cluster, graph);
  int nloc = cluster.devices.size();

  using jid_t = forward_state_t::jid_t;
  using rid_t = forward_state_t::rid_t;

  std::function<partition_t(int)> get_partition = [&](int gid)
  {
    return graph.nodes[gid].placement.partition;
  };

  std::function<int(jid_t)> get_location;
  if(random_loc) {
    get_location = [&](jid_t) { return runif(nloc); };
  } else {
    get_location = [&](jid_t jid)
    {
      auto const& [gid,bid] = jid;
      return graph.nodes[gid].placement.locations.get()[bid];
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

  {
    //uint64_t mmlike_sizing = 1000u*1000u*1000u;
    uint64_t mmlike_sizing = 10000u*10000u*10000u;

    //uint64_t min_sizing = 800u*800u;
    uint64_t min_sizing = 10000u*9000u;


    vector<partition_t> new_partition = autopartition(
      graph,
      mmlike_sizing,
      min_sizing);
    graph.reset_annotations(new_partition);
  }

  bool random_loc = true;

  for(int i = 0; i != 100; ++i) {
    set_seed(i);
    random_walk_through(graph, cluster, random_loc);
  }
}

void main04() {
  int nlocs = 1;

  cluster_t cluster = make_cluster(nlocs, 1, 1);

  graph_t graph;
  int inn = graph.insert_input({1000,1000});
  int out = graph.insert_formation(inn, true);

  forward_state_t state(cluster, graph);

  vector<partdim_t> pds {
    partdim_t::split(1000, 2),
    partdim_t::split(1000, 2)
  };
  state.assign_partition(inn, partition_t(pds));
  state.assign_partition(out, partition_t(pds));

  {
    std::ofstream f("tl.gv");
    state.print_twolayer_graphviz(f);
    DOUT("Printed to tl.gv");
  }

  {
    graph.nodes[inn].placement = placement_t(partition_t(pds));
    graph.nodes[out].placement = placement_t(partition_t(pds));
    auto [_0, _1, twolayer] = twolayergraph_t::make(graph);
    std::ofstream f("tl2.gv");
    twolayer.print_graphviz(f);
    DOUT("Printed to tl2.gv");
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

namespace mcmc_ns {

double simulate(
  cluster_t const& cluster,
  graph_t const& graph,
  vector<placement_t> const& pls)
{
  forward_state_t state(cluster, graph);

  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& pl = pls[gid];
    state.assign_partition(gid, pl.partition);
    vector<int> const& locs = pl.locations.get();
    for(int bid = 0; bid != locs.size(); ++bid) {
      state.assign_location({gid, bid}, locs[bid]);
    }
  }

  double makespan = 0.0;
  while(!state.all_done()) {
    state.enqueue_all();
    auto completed = state.pop_work();
    makespan = std::max(completed.finish, makespan);
  }
  return makespan;
}

vector<placement_t> single_loc_placements(graph_t const& graph) {
  vector<placement_t> pls;
  pls.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    pls.emplace_back(partition_t::singleton(node.op.shape()));
  }
  return pls;
}

struct mcmc_t {
  mcmc_t(
    cluster_t const& cl,
    graph_t const& gr,
    double bt,
    vector<placement_t> placements):
    cluster(cl), graph(gr), beta(bt), current_placements(placements)
  {
    current_makespan = simulate(cluster, graph, current_placements);

    best_makespan = current_makespan;
    best_placements = current_placements;
  }

  mcmc_t(
    cluster_t const& cl,
    graph_t const& gr,
    double bt)
    : mcmc_t(cl, gr, bt, single_loc_placements(gr))
  {}

  bool step() {
    vector<placement_t> pls = random_change();
    double makespan = simulate(cluster,  graph, pls);

    if(makespan < best_makespan) {
      best_makespan = makespan;
      best_placements = pls;
    }

    bool accept = makespan < current_makespan;
    if(!accept) {
      double diff = current_makespan - makespan;
      double prob_accept = std::exp(beta * diff);
      accept = runif({1-prob_accept, prob_accept});
    }

    if(accept) {
      current_makespan = makespan;
      current_placements = pls;
      return true;
    } else {
      return false;
    }
  }

  vector<placement_t> random_change() const {
    int prob_change_partition = 10;

    vector<placement_t> ret = current_placements;

    int n_locs = cluster.devices.size();
    int n_gids = graph.nodes.size();
    if(runif(100) < prob_change_partition) {
      // change the partition: either make it coarser or finer
      placement_t& pl = ret[runif(n_gids)];
      int n_parts = pl.partition.num_parts();
      if(n_parts == 1) {
        pl = make_finer(pl);
      } else if(n_parts > n_locs + n_locs/2) {
        pl = make_coarser(pl);
      } else {
        if(runif(2) < 1) {
          pl = make_coarser(pl);
        } else {
          pl = make_finer(pl);
        }
      }
    } else {
      placement_t& pl = ret[runif(n_gids)];
      vector<int>& locs = pl.locations.get();
      int bid = runif(locs.size());

      // pick any loc except locs[bid]
      int loc = runif(n_locs-1);
      if(loc >= locs[bid]) {
        loc += 1;
      }

      // pick an adjacent location
      //int loc;
      //if(locs[bid] == 0) {
      //  loc = 1;
      //} else if(locs[bid] == n_locs-1) {
      //  loc = locs[bid]-1;
      //} else {
      //  if(runif(3) < 2) {
      //    loc = locs[bid]-1;
      //  } else {
      //    loc = locs[bid]+1;
      //  }
      //}

      locs[bid] = loc;
    }

    return ret;
  }

  static placement_t make_finer(placement_t const& pl) {
    auto const& part = pl.partition;
    auto blk_shape = part.block_shape();
    vector<int> can_ds;
    for(int d = 0; d != blk_shape.size(); ++d) {
      auto const& pd = part.partdims[d];
      auto szs = pd.sizes();
      uint64_t sz = *std::min_element(szs.begin(), szs.end());
      if(sz >= 2) {
        can_ds.push_back(d);
      }
    }

    int d = can_ds[runif(can_ds.size())];
    auto new_partdims = part.partdims;
    new_partdims[d] = partdim_t::split_each(new_partdims[d], 2);
    partition_t new_partition(new_partdims);

    tensor_t<int> const& locs = pl.locations;
    tensor_t<int> new_locs(new_partition.block_shape());
    vector<int> index(blk_shape.size(), 0);
    do {
      int const& loc = locs.at(index);
      vector<int> new_index = index;

      new_index[d] *= 2;
      new_locs.at(new_index) = loc;
      new_index[d] += 1;
      new_locs.at(new_index) = loc;
    } while(increment_idxs(blk_shape, index));

    return placement_t(new_partition, new_locs);
  }

  static placement_t make_coarser(placement_t const& pl) {
    auto const& part = pl.partition;
    auto blk_shape = part.block_shape();
    vector<int> can_ds;
    for(int d = 0; d != blk_shape.size(); ++d) {
      auto const& pd = part.partdims[d];
      if(pd.num_parts() % 2 == 0) {
        can_ds.push_back(d);
      }
    }

    int d = can_ds[runif(can_ds.size())];
    auto new_partdims = part.partdims;
    {
      vector<uint64_t> new_sizes;
      auto const& pd = new_partdims[d];
      auto sizes = pd.sizes();
      int nr = pd.num_parts() / 2;
      for(int r = 0; r != nr; ++r) {
        new_sizes.push_back(sizes[2*r] + sizes[2*r+1]);
      }
      new_partdims[d] = partdim_t::from_sizes(new_sizes);
    }
    partition_t new_partition(new_partdims);

    auto new_blk_shape = new_partition.block_shape();
    tensor_t<int> new_locs(new_blk_shape);

    vector<int> index(new_blk_shape.size(), 0);

    tensor_t<int> const& locs = pl.locations;
    do {
      vector<int> old_index = index;

      old_index[d] *= 2;
      int loc1 = locs.at(old_index);
      old_index[d] += 1;
      int loc2 = locs.at(old_index);

      if(runif(2)){
        new_locs.at(index) = loc1;
      } else {
        new_locs.at(index) = loc2;
      }
    } while(increment_idxs(new_blk_shape, index));

    return placement_t(new_partition, new_locs);
  }

  cluster_t const& cluster;
  graph_t const& graph;
  double beta;

  double best_makespan;
  vector<placement_t> best_placements;

  double current_makespan;
  vector<placement_t> current_placements;
};

}

void main05() {
  // Do markov chain monte carlo to search the space..

  int nlocs = 4;

  cluster_t cluster = make_cluster(nlocs, 3, 1);

  //auto graph = three_dimensional_matrix_multiplication(
  //  4,4,4,
  //  4000,4000,4000,
  //  nlocs);

  float learning_rate = 0.1;
  uint64_t dn = 1500;
  uint64_t dp = 1500;
  uint64_t dd = 1500;
  vector<uint64_t> dws{2000, 2000};

  ff_sqdiff_t ff = ff_sqdiff_update(dn, dp, dd, dws, learning_rate);
  auto [graph, _] = ff.mgraph.compile();

  using namespace mcmc_ns;

  mcmc_t mcmc(cluster, graph, 600.1);

  double base = mcmc.current_makespan;

  for(int i = 0; i != 20000; ++i) {
    mcmc.step();
    if(i % 25 == 0) {
      double speedup = base / mcmc.best_makespan;
      DOUT((mcmc.best_makespan / base) << "             | "
            << (mcmc.current_makespan / base) << "");
      //DOUT( speedup << "         | " << ( base / mcmc.current_makespan ));
    }
  }

  {
    auto const& pls = mcmc.best_placements;
    for(int gid = 0; gid != graph.nodes.size(); ++gid) {
      graph.nodes[gid].placement = pls[gid];
    }

    auto const& [_0, _1, taskgraph] = taskgraph_t::make(graph);
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    DOUT("Printed to tg.gv");
  }
}

int main() {
  main05();
}
