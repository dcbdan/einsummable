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

// TODO: better name for multiple_placement_t and multiple_tensor_t?

struct multiple_placement_t {
  static multiple_placement_t make_refinement(vector<placement_t> const& ps);

  static multiple_placement_t make_einsummable_input(
    placement_t const& join_placement,
    einsummable_t const& einsummable,
    int which_input);

  partition_t const partition;
  tensor_t<set<int>> const locations;
};

struct multiple_tensor_t {
  struct locid_t {
    int loc;
    int id;
  };

  int vec_at(int vec_index, int desired_loc) const {
    for(auto const& [loc,id]: tensor.get()[vec_index]) {
      if(desired_loc == loc) {
        return id;
      }
    }
    throw std::runtime_error("multiple_tensor_t::vec_at could not get");
  }

  int at(vector<int> const& index, int desired_loc) const {
    for(auto const& [loc,id]: tensor.at(index)) {
      if(desired_loc == loc) {
        return id;
      }
    }
    throw std::runtime_error("multiple_tensor_t::at could not get");
  }

  partition_t partition;
  tensor_t<vector<locid_t>> tensor;

  tensor_t<int> to_tensor(placement_t const& placement);
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
  map<int, multiple_tensor_t> refined_tensors;

  // Get the which_input tensor for operation gid.
  // Since a join may require multiple uses of an input block,
  // the return type is multiple_tensor_t
  multiple_tensor_t access(int gid, int which_input);

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
      outputs[gid] = state.access(gid, 0).to_tensor(node.placement);
    } else {
      tensor_t<int> compute_result = state.compute(gid);
      state.communicate(gid, std::move(compute_result));
    }
  }

  return {std::move(outputs), std::move(state.taskgraph)};
}

// TODO: this could be better if it stored previous results and detected
//       similar placement usages...
//       Consider y = x + x where x (lhs input) and x(rhs input) are placed
//       in the same way. Then if x (previous computation) was formed differently
//       then either of the lhs or rhs x inputs, this does a bunch of duplicate work.
// TODO TODO TODO
multiple_tensor_t
state_t::access(int join_gid, int which_input)
{
  graph_t::node_t const& join_node = graph.nodes[join_gid];
  einsummable_t const& einsummable = std::get<einsummable_t>(join_node.op.op);

  // get the multiple placement of the input necessary for the join
  multiple_placement_t inn_placement =
    multiple_placement_t::make_einsummable_input(
      join_node.placement, einsummable, which_input);

  // get the refined tensor of the relevant input tensor
  int gid = join_node.inns[which_input];
  multiple_tensor_t const& refine_tensor = refined_tensors.at(gid);

  // When the refinement is a no-op, just return the refined tensor
  if(inn_placement.partition == refine_tensor.partition) {
    return refine_tensor;
  }

  // At this point we have refine_tensor, which contains a hyper-rectangular
  // grid of sub-tensors, each sub-tensor at all the locations it will be used
  // across all operations.
  //
  // For this particular operation, it will be used according to the partition
  // and locations in inn_placement. Each sub-tensor in the refine_tensor maps
  // to exactly one subtensor in the inn_placement partition.
  //
  // For each sub-tensor in refine_tensor, write it into the locations it'll
  // be used at for inn_placement.
  tensor_t<vector<multiple_tensor_t::locid_t>> ret;

  auto out_shape = inn_placement.partition.block_shape();
  vector<int> out_index(out_shape.size(), 0);
  do {
    // At this index and all locations it'll be used at, we need
    // to write the new id into the return tensor and get the
    // partialize builders
    auto const& locs = inn_placement.locations.at(out_index);
    vector<taskgraph_t::partialize_builder_t> builders;
    builders.reserve(locs.size());

    auto write_shape = inn_placement.partition.tensor_shape_at(out_index);
    for(auto const& loc: locs) {
      auto builder = taskgraph.new_partialize(write_shape, loc);
      ret.at(out_index).push_back(
        multiple_tensor_t::locid_t {
          .loc = builder.loc,
          .id  = builder.id
        });
      builders.push_back(builder);
    }

    // Now iterate through all the refined subtensors writing
    // into the output
    auto hrect = inn_placement.partition.get_hrect(out_index);
    auto refine_region = refine_tensor.partition.get_exact_region(hrect);
    vector<int> refine_index = vector_mapfst(refine_region);

    // Get the out offset here
    do {
      auto refine_hrect = refine_tensor.partition.get_hrect(refine_index);
      for(auto& builder: builders) {
        int refine_tensor_id = refine_tensor.at(refine_index, builder.loc);
        builder.region_write(hrect, refine_hrect, refine_tensor_id);
      }
    } while(increment_idxs_region(refine_region, refine_index));
  } while(increment_idxs(out_shape, out_index));

  return multiple_tensor_t {
    .partition = inn_placement.partition,
    .tensor = std::move(ret)
  };
};

tensor_t<int>
state_t::compute(int gid)
{
  graph_t::node_t const& node = graph.nodes[gid];

  if(node.op.is_input()) {
    auto shape = node.placement.block_shape();
    tensor_t<int> ret(shape);
    vector<int> index(shape.size(), 0);
    do {
      int const& loc = node.placement.locations.at(index);
      auto subtensor_shape = node.placement.partition.tensor_shape_at(index);
      ret.at(index) = taskgraph.insert_input(loc, subtensor_shape);
    } while(increment_idxs(shape, index));

    return ret;
  }

  // Get the inputs
  vector<multiple_tensor_t> inputs;
  inputs.reserve(node.inns.size());
  for(int i = 0; i != inputs.size(); ++i) {
    inputs.push_back(this->access(gid, i));
  }

  einsummable_t const& base_einsummable = std::get<einsummable_t>(node.op.op);

  auto shape = node.placement.block_shape();
  tensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);

  do {
    int const& loc = node.placement.locations.at(index);

    vector<int> inns;
    inns.reserve(inputs.size());
    auto inn_idxs = base_einsummable.input_idxs(index);
    for(int i = 0; i != inputs.size(); ++i) {
      auto const& inn_tensor = inputs[i];
      auto const& inn_idx = inn_idxs[i];
      inns.push_back(inn_tensor.at(inn_idx, loc));
    }

    auto subtensor_shape = node.placement.partition.tensor_shape_at(index);
    ret.at(index) = taskgraph.insert_einsummable(
      loc,
      einsummable_t::with_new_shape(base_einsummable, subtensor_shape),
      inns);
  } while(increment_idxs(shape, index));

  return ret;
}

void
state_t::communicate(int join_gid, tensor_t<int> join_result)
{
  // TODO
  // 1. where does this tensor get used? get those placements and call make_refinement
  // 2. form the multiple_tensor_t object by doing the computation
}

multiple_placement_t multiple_placement_t::make_refinement(vector<placement_t> const& ps) {
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

    return multiple_placement_t {
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

  return multiple_placement_t {
    .partition = std::move(partition),
    .locations = std::move(locations)
  };
}

multiple_placement_t multiple_placement_t::make_einsummable_input(
  placement_t const& join_placement,
  einsummable_t const& einsummable,
  int which_input)
{
  partition_t partition(
    einsummable.get_input_from_join(
      join_placement.partition.partdims,
      which_input));

  auto join_shape = join_placement.block_shape();
  vector<int> join_index(join_shape.size(), 0);

  auto inn_shape = partition.block_shape();
  tensor_t<set<int>> locations(inn_shape);

  do {
    auto inn_index = einsummable.get_input_from_join(join_index, which_input);
    locations.at(inn_index).insert(join_placement.locations.at(join_index));
  } while(increment_idxs(join_shape, join_index));

  return multiple_placement_t {
    .partition = std::move(partition),
    .locations = std::move(locations)
  };
}

tensor_t<int> multiple_tensor_t::to_tensor(placement_t const& placement) {
  if(!vector_equal(placement.block_shape(), tensor.get_shape())) {
    throw std::runtime_error("multiple_tensor_t::to_tensor");
  }

  vector<int> const& locs = placement.locations.get();
  int sz = locs.size();

  vector<int> ret;
  ret.reserve(sz);

  for(int i = 0; i != sz; ++i) {
    ret.push_back(vec_at(i, locs[i]));
  }

  return ret;
}
