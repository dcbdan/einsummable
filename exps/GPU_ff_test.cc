#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/execution/gpu/execute.h"

#include <fstream>
#include <memory>


struct random_placement_t {
  placement_t operator()(vector<uint64_t> const& total_shape) {
    vector<partdim_t> partdims;
    for(uint64_t const& n: total_shape) {
      auto const& [beg_,end_] = part_size_rng;
      int p;
      if(end_ > n) {
        p = 1;
      } else {
        p = runif(beg_, end_);
      }
      partdims.push_back(partdim_t::split(n, p));
    }
    partition_t part(partdims);

    return placement_t::random(part, nloc);
  }

  tuple<int,int> part_size_rng;
  int nloc;
};

void main02() {
  std::unordered_map<einsummable_t, int> zzz;
  zzz.insert({einsummable_t::from_matmul(100,101,102), 9});
  zzz.insert({einsummable_t::from_matmul(100,101,103), 10});

  for(auto const& [e,v]: zzz) {
    DOUT(e << " " << v);
  }
}

int main() {
  using id_t = graph_writer_t::tensor_t;

  graph_writer_t writer;

  uint64_t nb = 100;
  uint64_t nw = 101;

  id_t x = writer.input({nb,nw});

  uint64_t nw_prev = nw;
  int nlayer = 4;
  for(int i = 0; i != nlayer; ++i) {
    uint64_t nw_next = runif(20,121);
    id_t w = writer.input({nw_prev, nw_next});
    x = writer.matmul(x, w);
    nw_prev = nw_next;
  }

  x.save();

  graph_t const& graph = writer.get_graph();

  random_placement_t random_placement {
    .part_size_rng = {2,4},
    .nloc = 1
  };

  vector<placement_t> placements;
  placements.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    placements.push_back(random_placement(node.op.shape()));
  }

  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);

  int np = taskgraph.num_locs();
  vector<int> compute_loc_to_cache(np, 0);

  auto [_2, _3, memgraph] = memgraph_t::make_without_evict(
    taskgraph, compute_loc_to_cache, { 100000 });

  // print the number of nodes in the graph
    std::cout << "Number of nodes in the graph: " << memgraph.nodes.size() << std::endl;
    // print the input and output of every node
    // for(int i = 0; i < memgraph.nodes.size(); ++i) {
    //   std::cout << "Node " << i << " has input: ";
    //   for(auto in: memgraph.nodes[i].inns) {
    //     std::cout << in << " ";
    //   }
    //   std::cout << "and output: ";
    //   for(auto out: memgraph.nodes[i].outs) {
    //     std::cout << out << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "Printing to mm3d_mem_lowest_dep.gv" << std::endl;
    // std::ofstream f("mm3d_mem_lowest_dep.gv");
    // memgraph.print_graphviz(f);
    uint64_t buffer_size = 0;
    execute(memgraph);
}
