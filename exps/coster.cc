#include "../src/coster.h"

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

void main_matmul111() {
  graph_t graph;

  uint64_t ni = 10000;
  uint64_t nj = 10000;
  uint64_t nk = 10000;

  int pi = 1;
  int pj = 1;
  int pk = 1;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_lhs = graph.insert_input(partition_t({pdi,pdj}));
  int id_rhs = graph.insert_input(partition_t({pdj,pdk}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    partition_t({pdi, pdk, pdj}),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  coster_t coster(graph, make_cluster(1));

  float cost = coster();
  std::cout << "cost: " << cost << std::endl;
}

void main_matmul222() {
  graph_t graph;

  uint64_t ni = 10000;
  uint64_t nj = 10000;
  uint64_t nk = 10000;

  int pi = 2;
  int pj = 2;
  int pk = 2;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_lhs = graph.insert_input(partition_t({pdi,pdj}));
  int id_rhs = graph.insert_input(partition_t({pdj,pdk}));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    partition_t({pdi, pdk, pdj}),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  coster_t coster(graph, make_cluster(1));

  float cost = coster();
  std::cout << "cost: " << cost << std::endl;
}

void main_matmul_pijk_randomloc(int pi, int pj, int pk, int nloc) {
  graph_t graph;

  uint64_t ni = 10000;
  uint64_t nj = 10000;
  uint64_t nk = 10000;

  partdim_t pdi = partdim_t::split(ni, pi);
  partdim_t pdj = partdim_t::split(nj, pj);
  partdim_t pdk = partdim_t::split(nk, pk);

  int id_lhs = graph.insert_input(placement_t::random({pdi,pdj}, nloc));
  int id_rhs = graph.insert_input(placement_t::random({pdj,pdk}, nloc));

  einsummable_t matmul = einsummable_t::from_matmul(ni, nj, nk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    placement_t::random({pdi, pdk, pdj}, nloc),
    matmul,
    {id_lhs, id_rhs});

  int id_save = graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  coster_t coster(graph, make_cluster(nloc));

  float cost = coster();
  std::cout << "cost: " << cost << std::endl;
}

void main_exp01() {
  int nloc = 8;
  for(int i = 0; i != 10; ++i) {
    set_seed(i);
    main_matmul_pijk_randomloc(2,2,2,nloc);
  }
  //
  main_matmul111();
  //
  graph_t matmul3d = three_dimensional_matrix_multiplication(
    2, 2, 2, 5000, 5000, 5000, nloc);
  coster_t coster3d(matmul3d, make_cluster(nloc));
  float cost3d = coster3d();
  std::cout << "cost: " << cost3d << std::endl;
}

struct random_change_t {
  struct gid_t {
    int nid;
    int bid;
  };

  random_change_t(graph_t& g, int nloc):
    graph(g), nloc(nloc)
  {
    int num_nodes = graph.nodes.size();
    for(int nid = 0; nid != num_nodes; ++nid) {
      auto const& vec = graph.nodes[nid].placement.locations.get();
      int num_blocks = vec.size();
      for(int bid = 0; bid != num_blocks; ++bid) {
        gids.push_back(gid_t { .nid = nid, .bid = bid });
      }
    }
  }

  void reset() {
    for(auto const& [nid, bid]: gids) {
      graph.nodes[nid].placement.locations.get()[bid] = runif(nloc);
    }
    prev = std::optional<tuple<gid_t, int>>();
  }

  void operator()() {
    auto const& gid = gids[runif(gids.size())];
    auto const& [nid, bid] = gid;

    int& current_loc = graph.nodes[nid].placement.locations.get()[bid];
    prev = {gid, current_loc};

    int new_loc = runif(nloc);
    current_loc = new_loc;
  }

  void undo() {
    if(prev) {
      auto [gid,loc] = prev.value();
      auto const& [nid,bid] = gid;
      graph.nodes[nid].placement.locations.get()[bid] = loc;
      prev = std::optional<tuple<gid_t, int>>();
    } else {
      throw std::runtime_error("cannot undo");
    }
  }

  graph_t& graph;
  int nloc;
  vector<gid_t> gids;
  std::optional<tuple<gid_t, int>> prev;
};

void main_random_changes() {
  set_seed(0);

  int pi = 2;
  int pj = 2;
  int pk = 2;
  int nloc = 8;
  graph_t matmul3d = three_dimensional_matrix_multiplication(
    2, 2, 2, 5000, 5000, 5000, nloc);
  coster_t coster(matmul3d, make_cluster(nloc));

  std::cout << "3d matmul cost: " << coster() << std::endl;

  random_change_t random_change(matmul3d, nloc);
  random_change.reset();

  for(int i = 0; i != 20; ++i) {
    for(int i = 0; i != 100; ++i) {
      random_change();
    }
    float cost = coster();
    std::cout << "cost: " << cost << std::endl;
  }
}

void main_while_better(int pi, int pj, int pk, int nloc, int seed,
  int nrep = 1,
  uint64_t di = 10000, uint64_t dj = 10000, uint64_t dk = 10000)
{
  set_seed(seed);

  graph_t matmul3d = three_dimensional_matrix_multiplication(
    pi, pj, pk, di / pi, dj / pj, dk / pk, nloc);
  coster_t coster(matmul3d, make_cluster(nloc));

  std::cout << "3d matmul cost: " << coster() << std::endl;

  random_change_t random_change(matmul3d, nloc);

  random_change.reset();

  float cost = coster();
  std::cout << "cost: " << cost << " (after reset)" << std::endl;
  for(int i = 0; i != 1000; ++i) {
    random_change();
    float new_cost = coster();
    if(new_cost > cost) {
      random_change.undo();
    } else {
      if(new_cost < cost) {
        cost = new_cost;
        std::cout << "cost: " << cost << std::endl;
      } else {
        // they're equal
      }
    }
  }

  //float cost_min = coster();
  //float cost_max = coster();
  //for(int i = 0; i != 1000; ++i) {
  //  float cc = coster();
  //  cost_min = std::min(cost_min, cc);
  //  cost_max = std::max(cost_max, cc);

  //}
  //std::cout << "? " << cost_min << ", " << cost_max << std::endl;
}

void main_while_better_straight(int pi, int pj, int pk, int nloc, int seed,
  int nrep = 1,
  uint64_t di = 10000, uint64_t dj = 10000, uint64_t dk = 10000)
{
  set_seed(seed);

  {
    graph_t matmul3d = three_dimensional_matrix_multiplication(
      pi, pj, pk, di / pi, dj / pj, dk / pk, nloc);
    coster_t coster(matmul3d, make_cluster(nloc));

    std::cout << "3d matmul cost: " << coster() << std::endl;
  }

  graph_t matmul = straight_matrix_multiplication(
    pi, pj, pk, di / pi, dj / pj, dk / pk);

  random_change_t random_change(matmul, nloc);
  random_change.reset();

  coster_t coster(matmul, make_cluster(nloc));
  float cost = coster();
  std::cout << "cost: " << cost << " (after init)" << std::endl;
  for(int i = 0; i != 1000; ++i) {
    random_change();
    float new_cost = coster();
    if(new_cost > cost) {
      random_change.undo();
    } else {
      if(new_cost < cost) {
        cost = new_cost;
        std::cout << "cost: " << cost << std::endl;
      } else {
        // they're equal
      }
    }
  }
}

void main_exp02() {
  int pi = 2;
  int pj = 2;
  int pk = 2;
  uint64_t di = 10000;
  uint64_t dj = 10000;
  uint64_t dk = 10000;
  int seed = 777;
  int nloc = 8;
  int nrep = 1; // TODO not used
  main_while_better_straight(pi, pj, pk, nloc, seed, nrep, di, dj, dk);
  std::cout << "---------------" << std::endl;
  main_while_better(pi, pj, pk, nloc, seed, nrep, di, dj, dk);
}

// Note: the costs should be the same whether or not
//   route 1:  graph -> taskgraph -> costgraph -> the cost
//   route 2:  graph -> twolayer -> costgraph -> the cost
// is taken. The benefit of route two is that locations can be
// changed without having to rematerializing twolayer. That is,
// if the locations are changed, if sufficies to use the existing
// taskgraph to go (-> costgraph -> the cost).
void main_compute_two_ways() {
  int pi = 4;
  int pj = 4;
  int pk = 4;
  uint64_t di = 10000;
  uint64_t dj = 10000;
  uint64_t dk = 10000;
  int nloc = 16;

  cluster_t cluster = make_cluster(nloc);

  graph_t graph = three_dimensional_matrix_multiplication(
    pi, pj, pk, di, dj, dk, nloc);

  auto [_0, _1, task] = taskgraph_t::make(graph);

  costgraph_t cost_from_task =
    costgraph_t::make_from_taskgraph(task);

  twolayergraph_t twolayer =
    twolayergraph_t::make(graph);

  costgraph_t cost_from_twolayer = costgraph_t::make(twolayer);

  float from_task, from_twolayer;
  from_task     = cost_from_task(cluster);
  from_twolayer = cost_from_twolayer(cluster);

  std::cout << "from task:     " << from_task     << std::endl;
  std::cout << "from twolayer: " << from_twolayer << std::endl;
}

void main_compute_two_ways_time() {
  int pi = 12;
  int pj = 12;
  int pk = 12;
  uint64_t di = 10000;
  uint64_t dj = 10000;
  uint64_t dk = 10000;
  int nloc = 24;

  cluster_t cluster = make_cluster(nloc);

  graph_t graph = three_dimensional_matrix_multiplication(
    pi, pj, pk, di, dj, dk, nloc);

  auto route_task = [&] {
    auto [_0, _1, task] = taskgraph_t::make(graph);
    costgraph_t cost_from_task =
      costgraph_t::make_from_taskgraph(task);
    return cost_from_task(cluster);
  };

  auto route_twolayer_full = [&] {
    twolayergraph_t twolayer =
      twolayergraph_t::make(graph);
    costgraph_t cost_from_twolayer = costgraph_t::make(twolayer);
    return cost_from_twolayer(cluster);
  };

  twolayergraph_t twolayer =
    twolayergraph_t::make(graph);
  auto route_twolayer_part = [&] {
    costgraph_t cost_from_twolayer = costgraph_t::make(twolayer);
    return cost_from_twolayer(cluster);
  };

  {
    raii_print_time_elapsed_t t("route_twolayer_full");
    std::cout << route_twolayer_full() << std::endl;
  }

  {
    raii_print_time_elapsed_t t("route_twolayer_part");
    std::cout << route_twolayer_part() << std::endl;
  }

  {
    raii_print_time_elapsed_t t("route_task         ");
    std::cout << route_task() << std::endl;
  }
}

int main() {
  main_compute_two_ways_time();
}

