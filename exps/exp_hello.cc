#include "../src/einsummable/graph.h"
#include "../src/base/args.h"

#include "../src/einsummable/taskgraph.h"

#include <fstream>

struct my_cluster_info_t {
  // this is a description of the cluster
  // TODO

  int get_world_size() const { return 1; /* TODO: implement */ }
};

vector<placement_t> my_autoplace(
  graph_t const& graph,
  my_cluster_info_t cluster_info);

graph_t my_make_graph(args_t const& args);

optional<string> check_placements(
  graph_t const& graph,
  vector<placement_t> const& pls,
  my_cluster_info_t const& info);

int main(int argc, char** argv) {
  // TODO
  // if executing anything,
  //   setup the server and the communicator here

  // TODO
  // setup a description of the cluster here
  my_cluster_info_t cluster_info;

  args_t args(argc, argv);
  graph_t graph = my_make_graph(args);

  {
    std::ofstream f("g.gv");
    graph.print_graphviz(f);
    std::cout << "printed g.gv" << std::endl;
  }

  // TODO
  // if executing anything:
  //   for each node in the graph, if it is input, get that input
  //   data and pass it in to the server

  auto placements = my_autoplace(graph, cluster_info);

  // Run some checks to detect any invalid placements for this graph
  optional<string> maybe_error = check_placements(graph, placements, cluster_info);
  if(maybe_error) {
    std::cout << "Invalid placements created: " << maybe_error.value() << std::endl;
    return 1;
  }

  // TODO
  // if executing,
  //   have the server execute the graph,
  //     server.execute_graph(graph, placements)
  //   maybe report timing information too
  //   maybe inspect results too

  // Note:
  // Executing the server would create a taskgraph and if the
  // server is executing with memgraphs, a memgraph too.
  // We'll create a taskgraph here just so we can print the graphviz file
  auto [_0, _1, taskgraph] = taskgraph_t::make(graph, placements);
  // Results _0 and _1 contain meta data mapping graph ids (gids) to
  // taskgraph ids (tids). We just want to print the taskgraph, so we'll ignore
  // them--the server would have to use _0 and _1.
  {
    std::ofstream f("tg.gv");
    taskgraph.print_graphviz(f);
    std::cout << "printed tg.gv" << std::endl;
  }
}

// Test that the provided placements are valid for the given graph.
// If an error is found, return a string. Return nullopt if no error is found.
optional<string> check_placements(
  graph_t const& graph,
  vector<placement_t> const& pls,
  my_cluster_info_t const& info)
{
  if(graph.nodes.size() != pls.size()) {
    return "number of nodes does not equal number of placements";
  }

  auto err = [](int gid, string ret) {
    return "on gid " + write_with_ss(gid) + ", " + ret;
  };

  int world_size = info.get_world_size();
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& node = graph.nodes[gid];
    auto const& op = node.op;

    auto const& pl = pls[gid];
    auto const& part = pl.partition;
    vtensor_t<int> const& locations = pl.locations;

    vector<uint64_t> op_shape = op.shape();

    if(op.shape() != part.total_shape()) {
      return err(gid, "op_shape != partitions total shape");
    }
    for(int const& loc: locations.get()) {
      if(loc < 0 || loc >= world_size) {
        return err(gid, "location is not in [0,world_size)");
      }
    }
  }

  return std::nullopt;
}

// A simple feedforward neural network
graph_t daniel_make_graph(args_t const& args) {
  uint64_t bsz     = args.get<uint64_t>("bsz");
  int      nhidden = args.get<int>("nhidden");
  uint64_t dhidden = args.get<uint64_t>("dhidden");

  vector<uint64_t> ds(nhidden + 1, dhidden);

  graph_writer_t writer;
  using tensor_t = graph_writer_t::tensor_t;

  scalarop_t relu = scalarop_t::make_relu();

  tensor_t X = writer.input({bsz, ds[0]});
  for(int i = 0; i != nhidden; ++i) {
    tensor_t W = writer.input({ds[i], ds[i+1]});

    X = writer.matmul(X, W);
    X = writer.ew(relu, X);
  }

  X = writer.softmax(X);
  X = X.save();

  return writer.get_graph();
}

graph_t my_make_graph(args_t const& args) {
  // TODO: implement for yourself
  return daniel_make_graph(args);
}

// Return placements without any partitioning and all on location zero
vector<placement_t> daniel_autoplace(
  graph_t const& graph,
  my_cluster_info_t cluster_info)
{
  return graph.make_singleton_placement();
}

vector<placement_t> my_autoplace(
  graph_t const& graph,
  my_cluster_info_t cluster_info)
{
  // TODO: implement for yourself
  return daniel_autoplace(graph, cluster_info);
}


