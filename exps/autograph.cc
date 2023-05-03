#include "../src/matrixgraph/ff.h"

void update_graph_partitions(
  graph_t& graph,
  vector<partition_t> parts)
{
  int n_nodes = graph.nodes.size();
  for(int i = 0; i != n_nodes; ++i) {
    graph.nodes[i].placement = placement_t(parts[i]);
  }
}

void main01() {
  uint64_t dn = 1000;
  uint64_t dp = 100;
  uint64_t dd = 100;
  vector<uint64_t> dws = {105, 110, 115, 120, 125};
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

  uint64_t mmlike_sizing = 75*75*75;
  uint64_t min_sizing = 25*25;

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
}

void main02() {
  for(int i = 0; i != 30; ++i) {
    partdim_t p = partdim_t::split(1000, 5);
    DOUT(p);
    partdim_t w = partdim_t::split_each(p, 8);
    DOUT(w);
  }
}

int main() {
  main01();
}
