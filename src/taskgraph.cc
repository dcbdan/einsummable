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
  static multiple_placement_t from_single_placement(placement_t const& p);

  static multiple_placement_t make_refinement(vector<placement_t> const& ps);

  static multiple_placement_t make_refinement(vector<multiple_placement_t> const& ps);

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
          .loc = builder.loc(),
          .id  = builder.id
        });
      builders.push_back(builder);
    }

    // Now iterate through all the refined subtensors writing
    // into the output
    auto hrect_wrt_full = inn_placement.partition.get_hrect(out_index);
    auto refine_region = refine_tensor.partition.get_exact_region(hrect_wrt_full);
    vector<int> refine_index = vector_mapfst(refine_region);

    // Welcome to the tedious world of hyper-rectangles...
    //
    // The full hrect of the entire tensor
    // -------------------------
    // |          |            |
    // |          |            |
    // |----------|------------|
    // |          |A    |A     |
    // |          |-----|------|
    // |          |A    |AB    |
    // -------------------------
    //
    // --------------
    // |C    |C     |
    // |-----|------|
    // |C    |CD    |
    // --------------
    //
    // A := hrect_wrt_full             (the hrect of this output)
    // B := refined_hrect_wrt_full     (one of the refined hrects)
    // C := output region for op       (from the perspective of the partialize op)
    // D := one input to output region (from the perspective of the partialize op)

    do {
      auto refine_hrect_wrt_full = refine_tensor.partition.get_hrect(refine_index);
      auto centered_hrect = center_hrect(hrect_wrt_full, refine_hrect_wrt_full);

      for(auto& builder: builders) {
        int refine_tensor_id = refine_tensor.at(refine_index, builder.loc());
        builder.region_write_full_input(centered_hrect, refine_tensor_id);
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
  using locid_t = multiple_tensor_t::locid_t;

  auto const& join_node = graph.nodes[join_gid];
  auto const& join_placement = join_node.placement;

  vector<multiple_placement_t> usage_placements;
  usage_placements.reserve(2*join_node.outs.size());
  for(auto const& out_gid: join_node.outs) {
    auto const& out_node = graph.nodes[out_gid];
    if(out_node.op.is_output()) {
      usage_placements.push_back(
        multiple_placement_t::from_single_placement(out_node.placement));
    } else if(out_node.op.is_einsummable()) {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage placements to collect
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          usage_placements.push_back(
            multiple_placement_t::make_einsummable_input(
              out_node.placement,
              out_node.op.get_einsummable(),
              which_input));
        }
      }
    } else {
      throw std::runtime_error("state_t::communicate: should not happen");
    }
  }

  auto refinement = multiple_placement_t::make_refinement(usage_placements);

  // TODO:
  // 1. if the refinement is equivalent to a placement the same as the join_result,
  //    we're actually done
  // 2. if the agg block size is one and the refinement is equivalent to a placement
  //    the same as the out join_result, we're actually done
  // TODO:
  // If some but not all join outputs are already correct, what do you do?
  // Want to avoid copying
  // (can change buyilds to a map<int, vector<builder>> instead where
  //  all missing ones have already been filled)

  auto block_shape = refinement.partition.block_shape();

  // the output object
  tensor_t<vector<locid_t>> ret(block_shape);

  // a build partial for every object in the output
  tensor_t<vector<taskgraph_t::partialize_builder_t> > builds(block_shape);

  // init builds and fill out ret: go through each index and
  // call constructor to set the write shape and id,
  // then copy id into ret
  {
    vector<int> index(block_shape.size(), 0);
    do {
      auto const& locs = refinement.locations.at(index);

      auto& ret_at = ret.at(index);
      auto& builds_at = builds.at(index);

      ret_at.reserve(locs.size());
      builds_at.reserve(locs.size());

      vector<uint64_t> write_shape = refinement.partition.tensor_shape_at(index);

      for(auto const& loc: locs) {
        builds_at.push_back(taskgraph.new_partialize(write_shape, loc));

        ret_at.push_back(locid_t {
          .loc = loc,
          .id = builds_at.back().id
        });
      }
    } while(increment_idxs(block_shape, index));
  }
  // at this point, ret is correct, but the all the partials in the task graph
  // are not properly filled out

  auto const& _join_partdims = join_placement.partition.partdims;

  int out_rank = join_node.op.out_rank();
  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  int join_rank = join_node.op.rank() - out_rank;
  std::optional<partition_t> maybe_agg_partition;
  if(join_rank > 0) {
    maybe_agg_partition = vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end());
  }

  // TODO:
  //   for every out index:
  //     1. for each loc, do local aggregations
  //     2. for each loc, partial:
  //          write directly into any local outputs
  //        otherwise
  //          a. create the subset
  //          b. move the subset
  //          c. write the subset to the out loc partial

  vector<int> out_shape = out_partition.block_shape();

  vector<int> join_shape;
  if(maybe_agg_partition) {
    join_shape = maybe_agg_partition.value().block_shape();
  }

  vector<int> out_index(out_rank, 0);
  do {
    // Form the partials that will need to be aggregated across locations.
    map<int, int> partials; // loc to id map
    if(maybe_agg_partition) {
      auto const& agg_partition = maybe_agg_partition.value();
      map<int, vector<int>> loc_to_ids;
      vector<int> join_index(join_rank, 0);
      do {
        auto index = vector_concatenate(out_index, join_index);
        int loc = join_placement.locations.at(index);
        int id = join_result.at(index);
        loc_to_ids[index].push_back(id);
      } while(increment_idxs(join_shape, join_index));

      for(auto const& [loc, ids]: loc_to_ids.items()) {
        if(ids.size() == 1) {
          partials[loc] = id;
        } else {
          int local_aggd_id = taskgraph.insert_consumed_aggregate(loc, castable, ids);
          partials[loc] = local_aggd_id;
        }
      }
    } else {
      // here the join placement is the out placement
      int loc = join_placement.locations.at(index);
      int id  = join_result.at(index);
      partials[loc] = id;
    }

    for(auto const& [partial_loc, id]: partials) {
    //
    }

  } while(increment_idxs(out_shape, out_rank));

  multiple_tensor_t refined_tensor {
    .partition = refinement.partition,
    .tensor    = std::move(ret)
  };

  refined_tensors.insert({join_gid, refined_tensor});
}

multiple_placement_t multiple_placement_t::from_single_placement(placement_t const& p)
{
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

// Here T must have member T::partition of type partition_t
// Assumption: ps is non-empty
template <typename T>
partition_t union_partition_holders(vector<T> const& ps)
{
  vector<partdim_t> partdims;
  int rank = ps[0].partition.block_shape().size();
  partdims.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    vector<partdim_t> xs;
    xs.reserve(ps.size());
    for(auto const& p: ps) {
      xs.push_back(p.partition.partdims[i]);
    }
    partdims.push_back(partdim_t::unions(xs));
  }
  return partition_t(partdims);
}

multiple_placement_t
multiple_placement_t::make_refinement(
  vector<placement_t> const& ps)
{
  if(ps.size() == 0) {
    throw std::runtime_error("make_placement_refinement: empty input");
  }
  if(ps.size() == 1) {
    return from_single_placement(ps[0]);
  }

  auto const& p0 = ps[0];

  // Setup the refined partition
  partition_t partition = union_partition_holders(ps);

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

multiple_placement_t
multiple_placement_t::make_refinement(
  vector<multiple_placement_t> const& ps)
{
  if(ps.size() == 0) {
    throw std::runtime_error("make_placement_refinement: empty input_");
  }
  if(ps.size() == 1) {
    return ps[0];
  }

  auto const& p0 = ps[0];

  // Setup the refined partition
  partition_t partition = union_partition_holders(ps);

  // Now set up the locations.
  // For each partition,
  //   for each block in the partition,
  //     get the refined block,
  //     add the current location to the refined block
  tensor_t<set<int>> locations(partition.block_shape());
  for(auto const& p: ps) {
    vector<int> p_shape = p.partition.block_shape();
    vector<int> p_index(p_shape.size(), 0);
    do {
      set<int> const& locs = p.locations.at(p_index);

      auto hrect = p.partition.get_hrect(p_index);
      vector<tuple<int,int>> region = partition.get_exact_region(hrect);

      vector<int> index = vector_mapfst(region);
      do {
        locations.at(index).insert(locs.begin(), locs.end());
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

taskgraph_t::partialize_builder_t::partialize_builder_t(
  taskgraph_t* self,
  vector<uint64_t> write_shape,
  int loc): self(self)
{
  // insert a new partialize_t node into the graph
  // and save the id
  partialize_t partialize{
    .loc = loc,
    .write_shape = write_shape,
    .units = vector<partialize_t::partial_unit_t>()
  };

  id = self->nodes.size();

  self->nodes.push_back(node_t {
    .op = op_t(partialize),
    .outs = set<int>()
  });
}

taskgraph_t::partialize_builder_t::~partialize_builder_t()
{
  // TODO: check that the partialize object is valid
  // 1. make sure that units.size() > 0
  // 2. make sure each partial_unit_t is over a disjoint out region
}

void
taskgraph_t::partialize_builder_t::region_write_full_input(
  vector<tuple<uint64_t, uint64_t>> hrect_out,
  int id_inn)
{
  using inn_regiondim_t = partialize_t::inn_regiondim_t;
  using out_regiondim_t = partialize_t::out_regiondim_t;
  using input_op_t      = partialize_t::input_op_t;
  using partial_unit_t  = partialize_t::partial_unit_t;

  vector<inn_regiondim_t> inn_regiondims;
  inn_regiondims.reserve(hrect_out.size());

  vector<out_regiondim_t> out_regiondims;
  out_regiondims.reserve(hrect_out.size());

  for(auto const& [b,e]: hrect_out) {
    inn_regiondims.push_back(inn_regiondim_t {
      .dim    = e-b,
      .offset = 0
    });
    out_regiondims.push_back(out_regiondim_t {
      .offset = b,
      .size   = e-b
    });
  }

  input_op_t input {
    .id         = id_inn,
    .consumable = false,
    .region     = inn_regiondims
  };

  partial_unit_t partial_unit {
    .castable = castable_t::add,
    .out_region = out_regiondims,
    .inputs = {input}
  };

  insert_partial_unit(partial_unit);
  get().units.push_back(partial_unit);
}

void
taskgraph_t::partialize_builder_t::insert_partial_unit(
  taskgraph_t::partialize_t::partial_unit_t const& unit)
{
  // Make sure that this will indeed be a new write into the partial
  // (really, all the out regions should be disjoint as well)
  for(auto const& other_unit: get().units) {
    if(vector_equal(other_unit.out_region, unit.out_region)) {
      throw std::runtime_error("region_write_full_input should have different regions");
    }
  }

  get().units.push_back(unit);

  // now all the inputs need to be registered on the other end
  for(auto const& unit_input: unit.inputs) {
    self->nodes[unit_input.id].outs.insert(id);
  }
}

taskgraph_t::partialize_builder_t taskgraph_t::new_partialize(
  vector<uint64_t> write_shape,
  int loc)
{
  return partialize_builder_t(this, write_shape, loc);
}

int taskgraph_t::insert_input(
  int loc,
  vector<uint64_t> shape)
{
  input_t input {
    .loc = loc,
    .size = product(shape)
  };

  return insert(node_t {
    .op = op_t(input),
    .outs = set<int>()
  });
}

int taskgraph_t::insert_einsummable(
  int loc,
  einsummable_t e,
  vector<int> inns)
{
  apply_t apply {
    .loc = loc,
    .inns = inns,
    .einsummable = e
  };

  node_t node {
    .op = op_t(apply),
    .outs = set<int>()
  };

  if(e.inns.size() != inns.size()) {
    throw std::runtime_error("insert_einsummable: incorrect number of inputs");
  }
  auto inn_shapes = e.inn_shapes();
  for(int i = 0; i != inns.size(); ++i) {
    if(nodes[i].op.tensor_size() != product(inn_shapes[i])) {
      throw std::runtime_error("insert_einsummable: input has wrong size");
    }
  }

  return insert(node);
}

int taskgraph_t::insert_move(
  int src,
  int dst,
  int inn)
{
  move_t move {
    .src = src,
    .dst = dst,
    .inn = inn,
    .size = nodes[inn].op.tensor_size()
  };

  return insert(node_t {
    .op = op_t(move),
    .outs = set<int>()
  });
}

int taskgraph_t::insert_consumed_aggregate(
  int loc,
  castable_t castable,
  vector<int> inns)
{
  using inn_regiondim_t = partialize_t::inn_regiondim_t;
  using out_regiondim_t = partialize_t::out_regiondim_t;
  using input_op_t      = partialize_t::input_op_t;
  using partial_unit_t  = partialize_t::partial_unit_t;

  if(inns.size() == 0) {
    throw std::runtime_error("invalid insert_consumed_aggregate argument");
  }

  uint64_t sz = get_size_at(inns[0]);

  int ret = nodes.size();

  vector<input_op_t> inputs;
  inputs.reserve(inns.size());
  for(auto const& inn: inns) {
    inputs.push_back(input_op_t {
      .id = id,
      .consumable = true,
      .region = { inn_regiondim_t { .dim = sz, .offset = 0 } }
    });
  }

  auto unit = partial_unit_t {
    .castable = castable,
    .out_region = { out_regiondim_t { .offset = 0, .size = sz } },
    .inputs = inputs
  };

  nodes.push_back(partialize_t {
    .loc = loc,
    .write_shape = {sz},
    .units = {unit}
  });

  return ret;
}

uint64_t taskgraph_t::get_size_at(int id) const
{
  return nodes[id].op.tensor_size();
}

int taskgraph_t::insert(node_t node) {
  int ret = nodes.size();

  for(auto inn: node.op.inputs()) {
    nodes[inn].outs.insert(ret);
  }

  nodes.push_back(node);
  return ret;
};

bool operator==(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs)
{
  return lhs.offset == rhs.offset && lhs.size == rhs.size;
}
bool operator!=(
  taskgraph_t::partialize_t::out_regiondim_t const& lhs,
  taskgraph_t::partialize_t::out_regiondim_t const& rhs)
{
  return !(lhs == rhs);
}

uint64_t taskgraph_t::op_t::tensor_size() const
{
  if(std::holds_alternative<input_t>(op)) {
    return std::get<input_t>(op).size;
  } else if(std::holds_alternative<apply_t>(op)) {
    return product(std::get<apply_t>(op).einsummable.out_shape());
  } else if(std::holds_alternative<move_t>(op)) {
    return std::get<move_t>(op).size;
  } else if(std::holds_alternative<partialize_t>(op)) {
    return product(std::get<partialize_t>(op).write_shape);
  } else {
    throw std::runtime_error("should not reach");
    return 0;
  }
}

set<int> taskgraph_t::op_t::inputs() const
{
  if(std::holds_alternative<input_t>(op)) {
    return {};
  } else if(std::holds_alternative<apply_t>(op)) {
    auto const& inns = std::get<apply_t>(op).inns;
    return set<int>(inns.begin(), inns.end());
  } else if(std::holds_alternative<move_t>(op)) {
    return {std::get<move_t>(op).inn};
  } else if(std::holds_alternative<partialize_t>(op)) {
    set<int> ret;
    for(auto const& partial_unit: std::get<partialize_t>(op).units) {
      for(auto const& input: partial_unit.inputs) {
        ret.insert(input.id);
      }
    }
    return ret;
  } else {
    throw std::runtime_error("should not reach");
    return {};
  }
}
