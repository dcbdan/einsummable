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

int main() {
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

  graph.print();

  coster_t coster(graph, make_cluster(1));

  std::cout << "cost: " << coster() << std::endl;
}

