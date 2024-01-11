#include "../src/einsummable/graph.h"
#include "../src/base/args.h"
#include "../src/einsummable/reference.h"

#include "../src/einsummable/taskgraph.h"
#include "../src/autoplace/apart.h"

#include <fstream>

#include "../llama/misc.h"
#include "../llama/modules.h"


using tensor_t = graph_writer_t::tensor_t;

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


// create a graph with input matrices A,B,W all of shape (100,100) computes U = A+W, V = B+W
graph_t xin_make_graph() {

  graph_writer_t writer;
  using tensor_t = graph_writer_t::tensor_t;

  scalarop_t mul = scalarop_t::make_mul();

  tensor_t A = writer.input({100, 100});
  tensor_t B = writer.input({100, 100});
  tensor_t W = writer.input({100, 100});

  tensor_t U = writer.add(A,W);
  tensor_t V = writer.add(B,W);

  U = U.save();
  V = V.save();

  return writer.get_graph();
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

// graph_t llama_7B_make_graph(){
//   set_default_dtype(dtype_t::f16);

//   auto args = model_args_t::llama_7B();

//   // TODO: set vocab_size
//   args.vocab_size = 123;
//   args.n_layers = 4;

//   graph_writer_t writer;

//   uint64_t bsz = 3;
//   uint64_t seq_len = 17;

//   args.batch_size = bsz;

//   auto model = transformer_t(&writer, args, 0);

//   // read bsz and seq_len and input input tensor
//   auto input_input = [&]() {
//     full_shape_t shape({
//       full_dim_t::singleton(bsz),
//       full_dim_t::singleton(seq_len),
//       args.full_dim()
//     });

//     return writer.input(shape);
//   };

//   tensor_t x = input_input();
//   tensor_t y = model.forward(x);

//   y = y.save();
//   return writer.get_graph();
// }




// graph_t my_make_graph(args_t const& args) {
//   // TODO: implement for yourself
//   return xin_make_graph(args);
// }

// Return placements without any partitioning and all on location zero
vector<placement_t> daniel_autoplace(
  graph_t const& graph,
  my_cluster_info_t cluster_info)
{
  return graph.make_singleton_placement();
}



vector<placement_t> xin_placement(
  graph_t const& graph, 
  vector<partition_t> const& parts,
  int nlocs)
{
  
  vector<placement_t> placement;
  // for(int gid = 0; gid != graph.nodes.size(); ++gid){
  //   auto const& part = parts[gid];
  //   std::cout << part << std::endl;
  //   placement.emplace_back(part);
  // }
  // return placement;

  partdim_t pd_1 = partdim_t::split(100, 1);
  partdim_t pd_2 = partdim_t::split(100, 2);
  partition_t part1(vector<partdim_t>{pd_1, pd_2});
  partition_t part2(vector<partdim_t>{pd_2, pd_1});
  for(auto const& part: {part2, part1, part2, part2, part1}) {
    placement.emplace_back(part);
  }
  return placement;
}

vector<placement_t> create_loc0_placements(vector<partition_t> const& parts) {
  vector<placement_t> ret;
  ret.reserve(parts.size());
  for(auto const& part: parts) {
    std::cout << "part: " << part << std::endl;
    ret.emplace_back(part);
  }
  return ret;
}

vector<partition_t> part_by_100x100(graph_t graph){
  partdim_t pd_1 = partdim_t::split(100, 1);
  partition_t part1(vector<partdim_t>{pd_1, pd_1});
  vector<partition_t> partition({part1, part1, part1, part1, part1});
  return partition;
}

vector<partition_t> part_by_all50x100(graph_t graph){
  partdim_t pd_1 = partdim_t::split(100, 1);
  partdim_t pd_2 = partdim_t::split(100, 2);
  partition_t part1(vector<partdim_t>{pd_2, pd_1});
  vector<partition_t> partition({part1, part1, part1, part1, part1});
  return partition;
}

vector<partition_t> part_by_all50x50(graph_t graph){
  partdim_t pd_2 = partdim_t::split(100, 2);
  partition_t part1(vector<partdim_t>{pd_2, pd_2});
  vector<partition_t> partition({part1, part1, part1, part1, part1});
  return partition;
}

vector<partition_t> part_by_50x100x50(graph_t graph){
  partdim_t pd_1 = partdim_t::split(100, 1);
  partdim_t pd_2 = partdim_t::split(100, 2);
  partition_t part1(vector<partdim_t>{pd_1, pd_2});
  partition_t part2(vector<partdim_t>{pd_2, pd_1});
  vector<partition_t> partition({part2, part1, part1, part2, part1});
  return partition;
}

vector<partition_t> custom_part(graph_t graph){
  partdim_t pd_1 = partdim_t::split(100, 1);
  partdim_t pd_2 = partdim_t::split(100, 2);
  partition_t part1(vector<partdim_t>{pd_1, pd_2});
  partition_t part2(vector<partdim_t>{pd_2, pd_1});
  vector<partition_t> partition({part2, part1, part1});
  return partition;
}




vector<placement_t> create_random_loc_placements(vector<partition_t> const& parts, int num_locs = 2)
{
  vector<placement_t> ret;
  ret.reserve(parts.size());
  int count = 0;
  for(auto const& part: parts) {
    std::cout << "part: " << part << std::endl;
    ret.emplace_back(part);
    for(int& loc: ret.back().locations.get()) {
      loc = 0;
      // loc = runif(num_locs);
      std::cout << "loc: " << loc << std::endl; 
      count+=1;
    }
  }
  return ret;
}


vector<placement_t> my_autoplace(
  graph_t const& graph,
  my_cluster_info_t cluster_info)
{
  // vector<partition_t> partition = autopartition_for_bytes(graph, 2);
  // // TODO: implement for yourself
  // vector<placement_t> placement = xin_placement(graph, partition, 4);
  // return placement;
  // return graph.make_singleton_placement();

  // vector<partition_t> partition = autopartition_for_bytes(graph, 1);
  // vector<placement_t> placement = create_loc0_placements(partition);
  vector<partition_t> partition = part_by_all50x100(graph);
  vector<placement_t> placement = xin_placement(graph, partition, 4);
  return placement;
  
}









int main() {

  // TODO
  // setup a description of the cluster here
  my_cluster_info_t cluster_info;
  graph_t graph = xin_make_graph();

  {
    std::ofstream f("my_graph.gv");
    graph.print_graphviz(f);
    std::cout << "printed my_graph.gv" << std::endl;
  }

  auto placements = my_autoplace(graph, cluster_info);

  // Run some checks to detect any invalid placements for this graph
  optional<string> maybe_error = check_placements(graph, placements, cluster_info);
  if(maybe_error) {
    std::cout << "Invalid placements created: " << maybe_error.value() << std::endl;
    return 1;
  }

  auto [inn_gid_to_tids, out_gid_to_tids, taskgraph] = taskgraph_t::make(graph, placements);
  
  DOUT("");
  for(auto const& [inn_gid, tids]: inn_gid_to_tids) {
    DOUT("inn_gid " << inn_gid << " has tids " << tids);
  }
  DOUT("")
  for(auto const& [out_gid, tids]: out_gid_to_tids) {
    DOUT("out_gid " << out_gid << " has tids " << tids);
  }
  DOUT("")

  for(int id = 0; id != taskgraph.nodes.size(); ++id) {
    auto const& node = taskgraph.nodes[id];
    if(node.op.is_input()){
      auto const& [loc,size] = node.op.get_input();
      DOUT(id << ", with loc " << loc << ": " << size);
    }
    if(node.op.is_partialize()) {
      auto const& p = node.op.get_partialize();
      for(auto const& [inn, touch]: p.as_touches_from_flat()) {
        DOUT(id << ", with inn " << inn << ": " << touch);
      }
    }

  }
  // Results _0 and _1 contain meta data mapping graph ids (gids) to
  // taskgraph ids (tids). We just want to print the taskgraph, so we'll ignore
  // them--the server would have to use _0 and _1.
  {
    std::ofstream f("my_task_graph.gv");
    taskgraph.print_graphviz(f);
    std::cout << "printed my_task_graph.gv" << std::endl;
  }
}
