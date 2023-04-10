#include "taskgraph.h"

// The compilation from graph to taskgraph is designed to
// minimize the total number of bytes issued in moves.
//
// Their are three phases to worry about. The "compute" phase,
// the "access" phase, the "communicate" phase.
//
// In the compute phase, input and einsummable nodes are processed:
// for every (block,loc) in the placement, issue the computation to
// compute that tensor at that loc.
// For einsummable nodes that issue aggregations, the output tensor
// isn't formed but instead the joined object.
//
// The communicate phase forms a "refined" tensor from the compute phase output.
// A placement refinement is a placement that can have multiple locations
// per block _and_ has a partition formed by taking the union of multiple
// usage partitions. Given those n placements, the placement refinement is
// the union partition of the inputs, where every part of the hyper-rectangle
// index set has the union of all locations that it appears at.
// If the compute node was an einsummable with aggregations, then the refinement
// phase must do the aggregation.
//
// The access phase gets the input for the compute phase from the
// communicate phase.

struct placement_refinement_t {
  static placement_refinement_t make(vector<placement_t> const& ps) {
    if(ps.size() == 0) {
      throw std::runtime_error("make_placement_refinement: empty input");
    }
    if(ps.size() == 1) {
      auto const& p = ps[0];

      vector<set<int>> locs;
      locs.reserve(p.locations.get().size());
      for(int const& loc: p.locations.get()) {
        locs.push_back({loc});
      }

      return placement_refinement_t {
        .partition = p.partition,
        .locations = tensor_t<set<int>>(p.locations.get_shape(), locs)
      };
    }

    auto const& p0 = ps[0];
    int rank = p0.partition.block_shape().size();

    // Setup the refined partition
    vector<partdim_t> partdims;
    partdims.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      vector<partdim_t> xs;
      xs.reserve(ps.size());
      for(auto const& p: ps) {
        xs.push_back(p.partition.partdims[i]);
      }
      partdims.push_back(partdim_t::unions(xs));
    }
    partition_t partition(partdims);

    // Now set up the locations.
    // For each partition,
    //   for each block in the partition,
    //     get the refined block,
    //     add the current location to the refined block
    tensor_t<set<int>> locations(partition.block_shape());
    for(auto const& p: ps) {
      vector<int> p_shape = p.block_shape();
      vector<int> p_index(p_shape.size(), 0);
      do {
        int loc = p.locations.at(p_index);

        auto hrect = p.partition.get_hrect(p_index);
        vector<tuple<int,int>> region = partition.get_exact_region(hrect);

        vector<int> index = vector_mapfst(region);
        do {
          locations.at(index).insert(loc);
        } while(increment_idxs_region(region, index));
      } while(increment_idxs(p_shape, p_index));
    }

    return placement_refinement_t {
      .partition = std::move(partition),
      .locations = std::move(locations)
    };
  }

  partition_t const partition;
  tensor_t<set<int>> const locations;
};

struct locid_t {
  int loc;
  int id;
};

struct state_t {
  state_t(graph_t const& graph)
    : graph(graph)
  {}

  // the input compute graph
  graph_t const& graph;

  // the output task graph
  taskgraph_t taskgraph;

  // Map from gid to the refined tensor formed from that gid.
  // The
  map<int, tensor_t<vector<locid_t> > > refined_tensors;


  // get the which_input tensor for operation gid
  tensor_t<int> access(int gid, int which_input);

  // create a tensor to hold the compute phase results.
  // (this will call access)
  tensor_t<int> compute(int gid);

  // create the refined_tensor object from the compute result
  void communicate(int gid, tensor_t<int> compute_result);

};

tuple<
  map<int, tensor_t<int> >, // for each output id, the tids of the blocks
  taskgraph_t>              // the actual taskgraph
taskgraph_t::make(graph_t const& graph)
{
  state_t state(graph);

  // map from output gid to tensor
  map<int, tensor_t<int>> outputs;

  for(int gid: graph.get_order()) {
    graph_t::node_t const& node = graph.nodes[gid];

    if(node.op.is_output()) {
      outputs[gid] = state.access(gid, 0);
    } else {
      tensor_t<int> compute_result = state.compute(gid);
      state.communicate(gid, std::move(compute_result));
    }
  }

  return {std::move(outputs), std::move(state.taskgraph)};
}
