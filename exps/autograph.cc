#include "../src/matrixgraph/ff.h"
#include "../src/einsummable/coster.h"
#include "../src/einsummable/loadbalanceplace.h"

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

void update_graph_partitions(
  graph_t& graph,
  vector<partition_t> parts)
{
  int n_nodes = graph.nodes.size();
  for(int i = 0; i != n_nodes; ++i) {
    graph.nodes[i].placement = placement_t(parts[i]);
  }
}

void main01_autopart() {
  uint64_t dn = 2000;
  uint64_t dp = 30000;
  uint64_t dd = 4000;
  vector<uint64_t> dws = {3000, 3000, 3000};
  float learning_rate = 0.001;

  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,learning_rate);
  auto const& mgraph = ff_info.mgraph;

  vector<int> outs = ff_info.wsout;
  outs.push_back(ff_info.sqdiff);
  auto [graph, m_to_g] = mgraph.compile(outs);

  graph.print();

  vector<char> _line(40, '/');
  std::string line(_line.begin(), _line.end());
  std::cout << line << std::endl;

  uint64_t mmlike_sizing = 1000*1000*1000;
  uint64_t min_sizing = 950*950;

  {
    vector<partition_t> new_parts = autopartition(
      graph, mmlike_sizing, min_sizing);
    update_graph_partitions(graph, new_parts);
  }

  graph.print();
  std::cout << line << std::endl;

  {
    set<tuple<int,int>> same_parts;
    for(int i = 0; i != ff_info.wsinn.size(); ++i) {
      int const& winn = m_to_g.at(ff_info.wsinn[i]);
      int const& wout = m_to_g.at(ff_info.wsout[i]);
      DOUT(winn << ", " << wout);
      same_parts.insert({winn, wout});
    }

    vector<partition_t> new_parts = autopartition(
      graph,
      mmlike_sizing, min_sizing,
      same_parts, {}
    );

    update_graph_partitions(graph, new_parts);
  }

  graph.print();
  std::cout << line << std::endl;

  int nloc = 4;
  load_balanced_placement(graph, nloc);

  for(auto const& node: graph.nodes) {
    //std::cout << node.placement.locations.get() << std::endl;
    vector<int> counts(nloc, 0);
    for(auto const& loc: node.placement.locations.get()) {
      counts[loc] += 1;
    }
    std::cout << counts << std::endl;
  }

}

int main() {
  main01_autopart();
}


