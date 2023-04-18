#include "taskgraph.h"
#include "copyregion.h"

// The compilation from graph to taskgraph is designed to
// automatically split up tensors so as to only move
// the specified bytes.
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

  multiple_tensor_t(
    partition_t p,
    tensor_t<vector<locid_t>> && t)
    : partition(p), tensor(std::move(t))
  {
    if(tensor.get_shape() != p.block_shape()){
      throw std::runtime_error("multiple_tensor_t incorrect shape");
    }
  }

  multiple_tensor_t(multiple_placement_t p, int init_value = 0)
    : partition(p.partition), tensor(p.partition.block_shape())
  {
    auto shape = partition.block_shape();
    vector<int> index(shape.size(), 0);
    do {
      auto const& locs = p.locations.at(index);
      auto& tensor_at = tensor.at(index);
      tensor_at.reserve(locs.size());
      for(auto const& loc: locs) {
        tensor_at.push_back(locid_t {
          .loc = loc,
          .id = init_value
        });
      }
    } while(increment_idxs(shape, index));
  }

  int vec_at(int vec_index, int desired_loc) const {
    for(auto const& [loc,id]: tensor.get()[vec_index]) {
      if(desired_loc == loc) {
        return id;
      }
    }
    throw std::runtime_error("multiple_tensor_t::vec_at could not get");
  }

  int const& at(vector<int> const& index, int desired_loc) const {
    for(auto const& [loc,id]: tensor.at(index)) {
      if(desired_loc == loc) {
        return id;
      }
    }
    throw std::runtime_error("multiple_tensor_t::at could not get");
  }

  int& at(vector<int> const& index, int desired_loc) {
    for(auto& [loc,id]: tensor.at(index)) {
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

std::ostream& operator<<(std::ostream& out, multiple_tensor_t::locid_t const& x)
{
  auto const& [loc,id] = x;
  out << "loc" << loc << "id" << id;
  return out;
}

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

  // create the refined_tensor object from the compute result;
  // save into refined_tensors
  void communicate(int gid, tensor_t<int> compute_result);

};

tuple<
  map<int, tensor_t<int> >, // for each input, the tids of the blocks
  map<int, tensor_t<int> >, // for each save id, the tids of the blocks
  taskgraph_t>              // the actual taskgraph
taskgraph_t::make(graph_t const& graph)
{
  state_t state(graph);

  // maps from gid to tensor
  map<int, tensor_t<int>> inns;
  map<int, tensor_t<int>> saves;

  for(int gid: graph.get_order()) {
    graph_t::node_t const& node = graph.nodes[gid];

    if(node.op.is_formation()) {
      tensor_t<int> access_result = state.access(gid, 0).to_tensor(node.placement);
      if(node.op.is_save()) {
        saves[gid] = access_result;
        // Set the things to be saved on the taskgraph
        for(int const& taskgraph_tid: access_result.get()) {
          state.taskgraph.nodes[taskgraph_tid].is_save = true;
        }
      }
      if(node.outs.size() > 0) {
        state.communicate(gid, std::move(access_result));
      }
    } else {
      // the node is an input or a einsummable
      tensor_t<int> compute_result = state.compute(gid);
      if(node.op.is_input()) {
        inns[gid] = compute_result;
      }
      state.communicate(gid, std::move(compute_result));
    }
  }

  if(!state.taskgraph.all_zero_outs_is_save()) {
    throw std::runtime_error("In taskgraph_t::make: non-saved outputs");
  }

  return {std::move(inns), std::move(saves), std::move(state.taskgraph)};
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
  DOUT("ACCESS " << join_gid << " WITH INPUT " << which_input);
  graph_t::node_t const& join_node = graph.nodes[join_gid];

  // get the multiple placement of the input necessary for the join
  multiple_placement_t inn_placement = [&]
  {
    if(join_node.op.is_einsummable()) {
      einsummable_t const& einsummable = std::get<einsummable_t>(join_node.op.op);
      return multiple_placement_t::make_einsummable_input(
        join_node.placement, einsummable, which_input);
    } else {
      // If it isn't an einsummable node, then it is a formation node,
      // in which case the required inn placement is the join_node.placement.
      return multiple_placement_t::from_single_placement(join_node.placement);
    }
  }();

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
  auto out_shape = inn_placement.partition.block_shape();
  tensor_t<vector<multiple_tensor_t::locid_t>> ret(out_shape);

  vector<int> out_index(out_shape.size(), 0);
  do {
    // At this index and all locations it'll be used at, we need
    // to write the new id into the return tensor and get the
    // partialize builders
    auto const& locs = inn_placement.locations.at(out_index);
    auto write_shape = inn_placement.partition.tensor_shape_at(out_index);

    // initialize the partials to be written
    auto& _ret_at = ret.at(out_index);
    _ret_at.reserve(locs.size());
    for(auto const& loc: locs) {
      int builder_id = taskgraph.new_partial(loc, write_shape);
      _ret_at.push_back(
        multiple_tensor_t::locid_t {
          .loc = loc,
          .id  = builder_id
        });
    }

    // now ret_at is filled out with uninitialized partials
    auto const& ret_at = _ret_at;

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
    // B := refine_hrect_wrt_full      (one of the refined hrects)
    // C := output region for op       (from the perspective of the partialize op)
    // D := one input to output region (from the perspective of the partialize op)

    do {
      auto refine_hrect_wrt_full = refine_tensor.partition.get_hrect(refine_index);
      auto centered_hrect = center_hrect(hrect_wrt_full, refine_hrect_wrt_full);

      for(auto const& [loc, builder_id]: ret_at) {
        int refine_tensor_id = refine_tensor.at(refine_index, loc);
        taskgraph.add_to_partial_the_full_input(
          builder_id, refine_tensor_id, centered_hrect);
      }
    } while(increment_idxs_region(refine_region, refine_index));
  } while(increment_idxs(out_shape, out_index));

  return multiple_tensor_t(inn_placement.partition, std::move(ret));
};

tensor_t<int>
state_t::compute(int gid)
{
  DOUT("COMPUTE " << gid);
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
  for(int i = 0; i != node.inns.size(); ++i) {
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
  DOUT("COMMUNICATE " << join_gid);
  using locid_t = multiple_tensor_t::locid_t;

  auto const& join_node = graph.nodes[join_gid];
  auto const& join_placement = join_node.placement;

  vector<multiple_placement_t> usage_placements;
  usage_placements.reserve(2*join_node.outs.size());
  for(auto const& out_gid: join_node.outs) {
    auto const& out_node = graph.nodes[out_gid];
    if(out_node.op.is_formation()) {
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
  auto refinement_shape = refinement.partition.block_shape();

  // the output
  // initialize every id to -1 initially; the locs
  // will track with the refinement locs
  multiple_tensor_t refined_tensor(refinement, -1);

  auto const& _join_partdims = join_placement.partition.partdims;

  int join_rank = join_node.op.rank();
  int out_rank = join_node.op.out_rank();
  int agg_rank = join_rank - out_rank;

  castable_t castable;

  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  std::optional<partition_t> maybe_agg_partition;
  if(agg_rank > 0) {
    maybe_agg_partition = vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end());

    // an agg will only happen if this is an einsummable node;
    // so set the castable value
    castable = graph.nodes[join_gid].op.get_einsummable().castable;
  }

  // There are out index, agg index, join index
  // that refer to the acutal join_gid and join_result.
  //
  // Then there is the refinement_index that is being created.
  //
  // for every out index:
  //   1. vector<loc,int> partials
  //      for each loc:
  //        do local aggregations and write to partial
  //   2. for each loc, partial pair:
  //        aggregate into local output refinement blocks
  //        and move data if necessary
  //
  // Note that all this is rather smart.
  //
  // If there are no aggregations
  // and the join_result is not refined, then join_result will be
  // copied directly into refined_tensor.
  //
  // Only the specified communication happens: if necessary,
  // output partials are split into subsets and moved to the correct
  // location.
  //
  // Since all moved data is only used once, it is always consumed
  // when aggregated into a partial. Unless the selection shape
  // does not equal the refinement shape, in which case the
  // moved data can't be consumed.
  //
  // Moreover, this all occurs on an output index basis.

  vector<int> out_shape = out_partition.block_shape();

  vector<int> agg_shape;
  if(maybe_agg_partition) {
    agg_shape = maybe_agg_partition.value().block_shape();
  }

  vector<int> out_index(out_rank, 0);
  do {
    // Form the partials that will need to be aggregated across locations.
    map<int, int> partials; // loc to id map
    if(maybe_agg_partition) {
      auto const& agg_partition = maybe_agg_partition.value();
      map<int, vector<int>> loc_to_ids;
      vector<int> agg_index(agg_rank, 0);
      do {
        auto join_index = vector_concatenate(out_index, agg_index);
        int loc = join_placement.locations.at(join_index);
        int id = join_result.at(join_index);
        loc_to_ids[loc].push_back(id);
      } while(increment_idxs(agg_shape, agg_index));

      for(auto const& [loc, ids]: loc_to_ids) {
        if(ids.size() == 1) {
          partials[loc] = ids[0];
        } else {
          int local_aggd_id = taskgraph.insert_consumed_aggregate(loc, castable, ids);
          partials[loc] = local_aggd_id;
        }
      }
    } else {
      // here the join placement is the out placement
      auto const& join_index = out_index;
      int loc = join_placement.locations.at(join_index);
      int id  = join_result.at(join_index);
      partials[loc] = id;
    }

    vector<uint64_t> out_tensor_shape = out_partition.tensor_shape_at(out_index);

    copyregion_t get_regions(refinement.partition, out_partition, out_index);
    do {
      auto const& subset_info = get_regions.info;
      // r for refinement
      vector<int> r_index = vector_from_each_member(subset_info, int, idx);
      set<int> const& required_locs = refinement.locations.at(r_index);

      vector<uint64_t> selection_shape = vector_from_each_member(
        subset_info, uint64_t, size);
      vector<uint64_t> refinement_shape =
        refinement.partition.tensor_shape_at(r_index);

      // {{{
      // Initializing lambdas:
      //   insert_out_to_selection
      //   add_selection_to_refinement
      //   add_out_to_refinement
      //
      vector<regiondim_t> _out_to_selection_rs;
      {
        _out_to_selection_rs.reserve(subset_info.size());
        for(int i = 0; i != subset_info.size(); ++i) {
          _out_to_selection_rs.push_back(regiondim_t {
            .dim    = out_tensor_shape[i],
            .offset = subset_info[i].offset_inn,
            .size   = selection_shape[i]
          });
        }
      }
      auto insert_out_to_selection = [&](int partial_loc, int partial_id) {
        return taskgraph.insert_select_subset(
          partial_loc, _out_to_selection_rs, partial_id);
      };

      // A constructor to modify ids of refined_tensor
      auto maybe_init_ref_id_as_partial = [&](int& ref_id, int loc) {
        if(ref_id == -1) {
          ref_id = taskgraph.new_partial(loc, refinement_shape);
        } else {
          auto const& node = taskgraph.nodes[ref_id];
          if(!node.op.is_partialize()) {
            throw std::runtime_error("got a ref id that isn't a partial");
          }
        }
      };

      vector<touchdim_t> _selection_to_refinement_ts;
      {
        _selection_to_refinement_ts.reserve(subset_info.size());
        for(int i = 0; i != subset_info.size(); ++i) {
          _selection_to_refinement_ts.push_back(touchdim_t {
            .d_inn      = selection_shape[i],
            .d_out      = refinement_shape[i],
            .offset_inn = 0,
            .offset_out = subset_info[i].offset_out,
            .size       = selection_shape[i]
          });
        }
      }
      touch_t _selection_to_refinement_touch {
        .selection = _selection_to_refinement_ts,
        .castable = optional<castable_t>(castable)
      };
      // ref_id by reference!
      auto add_selection_to_refinement = [&](int loc, int& ref_id, int sel_id) {
        maybe_init_ref_id_as_partial(ref_id, loc);
        taskgraph.add_to_partial(ref_id, sel_id, _selection_to_refinement_touch, false);
      };

      vector<touchdim_t> _out_to_refinement_ts;
      {
        _out_to_refinement_ts.reserve(subset_info.size());
        for(int i = 0; i != subset_info.size(); ++i) {
          _out_to_refinement_ts.push_back(touchdim_t {
            .d_inn      = out_tensor_shape[i],
            .d_out      = refinement_shape[i],
            .offset_inn = subset_info[i].offset_inn,
            .offset_out = subset_info[i].offset_out,
            .size       = subset_info[i].size
          });
        }
      }
      touch_t _out_to_refinement_touch {
        .selection = _out_to_refinement_ts,
        .castable = optional<castable_t>(castable)
      };
      // ref_id by reference!
      auto add_out_to_refinement = [&](int loc, int& ref_id, int partial_id) {
        maybe_init_ref_id_as_partial(ref_id, loc);
        taskgraph.add_to_partial(ref_id, partial_id, _out_to_refinement_touch, false);
      };
      // }}}

      // case 1: out_tensor_shape == selection_shape == refinement_shape
      // case 2: out_tensor_shape >> (selection_shape == refinement_shape)
      // case 3: out_tensor_shape >> selection_shape,
      //         selection_shape != refinement_shape

      if(partials.size() == 1) {
        auto const& [partial_loc, partial_id] = *partials.begin();
        if(vector_equal(selection_shape, out_tensor_shape)) {
          if(vector_equal(out_tensor_shape, selection_shape)) {
            // case 1: copy right into the refined_tensor
            for(int loc: required_locs) {
              if(loc == partial_loc) {
                refined_tensor.at(r_index, loc) = partial_id;
              } else {
                refined_tensor.at(r_index, loc) =
                  taskgraph.insert_move(partial_loc, loc, partial_id);
              }
            }
          } else {
            // case 2: subset then copy right into the refined tensor
            int subset_id = insert_out_to_selection(partial_loc, partial_id);
            for(int loc: required_locs) {
              if(loc == partial_loc) {
                refined_tensor.at(r_index, loc) = subset_id;
              } else {
                refined_tensor.at(r_index, loc) =
                  taskgraph.insert_move(partial_loc, loc, subset_id);
              }
            }
          }
        } else {
          // case 3:
          // 1. Will a subset selection need to be made?
          //      If so, create the subset and use that
          // 2. Otherwise, copy straight from the output to the refinement
          //      This only happens when partial_id is used exclusively at
          //      partial_loc
          if(required_locs.size() == 1 && *required_locs.begin() == partial_loc)
          {
            // ref_id by reference!
            int& ref_id = refined_tensor.at(r_index, partial_loc);
            add_out_to_refinement(partial_loc, ref_id, partial_id);
          } else {
            int subset_id = insert_out_to_selection(partial_loc, partial_id);
            for(int loc: required_locs) {
              if(loc == partial_loc) {
                int& ref_id = refined_tensor.at(r_index, partial_loc);
                add_selection_to_refinement(partial_loc, ref_id, subset_id);
              } else {
                int moved_subset_id =
                  taskgraph.insert_move(partial_loc, loc, subset_id);
                int& ref_id = refined_tensor.at(r_index, loc);
                add_selection_to_refinement(loc, ref_id, moved_subset_id);
              }
            }
          }
        }
      } else {
        // Here there are multiple partials which means partialize_t
        // objects may need to be created, but the same cases holds

        for(int loc: required_locs) {
          // intialize as partials if still uninitialized
          int& ref_id = refined_tensor.at(r_index, loc);
          maybe_init_ref_id_as_partial(ref_id, loc);
        }

        if(vector_equal(selection_shape, refinement_shape)) {
          if(vector_equal(out_tensor_shape, selection_shape)) {
            // case 1: aggregate the full input
            for(auto const& [partial_loc, partial_id]: partials) {
              for(int loc: required_locs) {
                int builder = refined_tensor.at(r_index, loc);
                if(loc == partial_loc) {
                  // partial_id will also be moved since required_locs.size() > 1.
                  // So don't consume the output!
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, partial_id, castable, false);
                } else {
                  // Here, the moved data can be consumed
                  int moved_partial_id =
                    taskgraph.insert_move(partial_loc, loc, partial_id);
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, moved_partial_id, castable, true);
                }
              }
            }
          } else {
            // case 2: subset then aggregate into the refined tensor
            for(auto const& [partial_loc, partial_id]: partials) {
              int subset_id = insert_out_to_selection(partial_loc, partial_id);
              for(int loc: required_locs) {
                int builder = refined_tensor.at(r_index, loc);
                if(loc == partial_loc) {
                  // subset_id will also be moved since required_locs.size() > 1.
                  // So don't consume the output!
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, subset_id, castable, false);
                } else {
                  // Here, the moved data can be consumed
                  int moved_subset_id =
                    taskgraph.insert_move(partial_loc, loc, subset_id);
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, moved_subset_id, castable, true);
                }
              }
            }
          }
        } else {
          // case 3:
          for(auto const& [partial_loc, partial_id]: partials) {
            // 1. If a subset selection needs to be made,
            //    create it and use that
            // 2. otherwise, copy straight into the builder
            if(required_locs.size() == 1 && *required_locs.begin() == partial_loc)
            {
              int builder = refined_tensor.at(r_index, partial_loc);
              add_out_to_refinement(partial_loc, builder, partial_id);
            } else {
              int subset_id = insert_out_to_selection(partial_loc, partial_id);
              for(int loc: required_locs) {
                int builder = refined_tensor.at(r_index, loc);
                if(loc == partial_loc) {
                  add_selection_to_refinement(partial_loc, builder, subset_id);
                } else {
                  int moved_subset_id =
                    taskgraph.insert_move(partial_loc, loc, subset_id);
                  add_selection_to_refinement(loc, builder, moved_subset_id);
                }
              }
            }
          }
        }
      }
    } while(get_regions.increment());
  } while(increment_idxs(out_shape, out_index));

  // for every id in the refined tensor,
  //   1. make sure it is initialized
  //   2. if it is a partial, verify it is valid
  for(auto const& locids: refined_tensor.tensor.get()) {
    for(auto const& [loc,id]: locids) {
      if(id == -1) {
        throw std::runtime_error("uninitialized id in refined tensor");
      }

      auto const& node = taskgraph.nodes[id];
      if(!node.op.is_valid_if_partialize()) {
        throw std::runtime_error("invalid partialize in communicate");
      }
    }
  }

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

  return tensor_t<int>(tensor.get_shape(), ret);
}

int taskgraph_t::insert_input(
  int loc,
  vector<uint64_t> shape,
  bool is_save)
{
  input_t input {
    .loc = loc,
    .size = product(shape)
  };

  return insert(input, is_save);
}

int taskgraph_t::insert_einsummable(
  int loc,
  einsummable_t e,
  vector<int> inns,
  bool is_save)
{
  apply_t apply {
    .loc = loc,
    .inns = inns,
    .einsummable = e
  };

  if(e.inns.size() != inns.size()) {
    throw std::runtime_error("insert_einsummable: incorrect number of inputs");
  }
  auto inn_shapes = e.inn_shapes();
  for(int i = 0; i != inns.size(); ++i) {
    int const& inn = inns[i];
    auto const& inn_shape = inn_shapes[i];
    if(nodes[inn].op.tensor_size() != product(inn_shape)) {
      throw std::runtime_error("insert_einsummable: input has wrong size");
    }
 }

  return insert(apply, is_save);
}

int taskgraph_t::insert_move(
  int src,
  int dst,
  int inn,
  bool is_save)
{
  move_t move {
    .src = src,
    .dst = dst,
    .inn = inn,
    .size = nodes[inn].op.tensor_size()
  };

  return insert(move, is_save);
}

int taskgraph_t::insert_consumed_aggregate(
  int loc,
  castable_t castable,
  vector<int> inns,
  bool is_save)
{
  using inn_regiondim_t = partialize_t::inn_regiondim_t;
  using out_regiondim_t = partialize_t::out_regiondim_t;
  using input_op_t      = partialize_t::input_op_t;
  using partial_unit_t  = partialize_t::partial_unit_t;

  if(inns.size() == 0) {
    throw std::runtime_error("invalid insert_consumed_aggregate argument");
  }

  uint64_t sz = get_size_at(inns[0]);

  vector<input_op_t> inputs;
  inputs.reserve(inns.size());
  for(auto const& inn: inns) {
    inputs.push_back(input_op_t {
      .id = inn,
      .consumable = true,
      .region = { inn_regiondim_t { .dim = sz, .offset = 0 } }
    });
  }

  auto unit = partial_unit_t {
    .castable = castable,
    .out_region = { out_regiondim_t { .offset = 0, .size = sz } },
    .inputs = inputs
  };

  return insert(partialize_t {
      .loc = loc,
      .write_shape = {sz},
      .units = {unit}
    },
    is_save);
}

int taskgraph_t::insert_select_subset(
  int loc,
  vector<regiondim_t> selection,
  int inn,
  bool is_save)
{
  using inn_regiondim_t = partialize_t::inn_regiondim_t;
  using out_regiondim_t = partialize_t::out_regiondim_t;
  using input_op_t      = partialize_t::input_op_t;
  using partial_unit_t  = partialize_t::partial_unit_t;

  vector<uint64_t> write_shape;
  write_shape.reserve(selection.size());

  vector<touchdim_t> ts;
  ts.reserve(selection.size());
  for(auto const& [dim_inn, offset_inn, size]: selection)
  {
    write_shape.push_back(size);

    ts.push_back(touchdim_t {
      .d_inn = dim_inn,
      .d_out = size,
      .offset_inn = offset_inn,
      .offset_out = 0,
      .size = size
    });
  }

  touch_t touch {
    .selection = ts,
    .castable = optional<castable_t>()
  };

  int ret = new_partial(loc, write_shape, is_save);
  add_to_partial(ret, inn, touch);
  return ret;
}

int taskgraph_t::new_partial(
  int loc,
  vector<uint64_t> write_shape,
  bool is_save)
{
  return insert(partialize_t {
      .loc = loc,
      .write_shape = write_shape,
      .units = {}
    },
    is_save);
}

void taskgraph_t::add_to_partial(
  int id_out,
  int id_inn,
  touch_t touch,
  bool consume)
{
  using inn_regiondim_t = partialize_t::inn_regiondim_t;
  using out_regiondim_t = partialize_t::out_regiondim_t;
  using input_op_t      = partialize_t::input_op_t;
  using partial_unit_t  = partialize_t::partial_unit_t;

  partialize_t& partialize = nodes[id_out].op.get_partialize();

  {
    vector<uint64_t> shape_out = vector_from_each_member(
      touch.selection, uint64_t, d_out);
    if(!vector_equal(partialize.write_shape, shape_out)) {
      throw std::runtime_error("add_to_partial: incorrect output shape");
    }
  }

  vector<out_regiondim_t> region_out;
  region_out.reserve(touch.selection.size());

  vector<inn_regiondim_t> region_inn;
  region_inn.reserve(touch.selection.size());

  for(auto const& [d_inn, _, offset_inn, offset_out, size]: touch.selection)
  {
    region_out.push_back(out_regiondim_t {
      .offset = offset_out,
      .size = size
    });
    region_inn.push_back(inn_regiondim_t {
      .dim = d_inn,
      .offset = offset_inn
    });
  }

  input_op_t input {
    .id = id_inn,
    .consumable = consume,
    .region = region_inn
  };

  bool found = false;
  for(partial_unit_t& unit: partialize.units) {
    if(vector_equal(unit.out_region, region_out)) {
      if(!touch.castable || !unit.castable) {
        throw std::runtime_error("optional castable must be something");
      }
      if(unit.castable.value() != touch.castable.value()) {
        throw std::runtime_error("cannot use different castables");
      }

      unit.inputs.push_back(input);

      found = true;
      break;
    } else {
      // TODO: assert that unit.out_region and region_out are disjoint
    }
  }

  if(!found) {
    partialize.units.push_back(partial_unit_t {
      .castable = touch.castable,
      .out_region = region_out,
      .inputs = {input}
    });
  }

  // make sure to tell nodes this id_inn gets used at id_out
  nodes[id_inn].outs.insert(id_out);
}

void taskgraph_t::add_to_partial_the_full_input(
  int id_out,
  int id_inn,
  vector<tuple<uint64_t, uint64_t>> hrect_out,
  bool consume)
{
  // this is a little goofy: we're gonna get the write shape,
  // put it into touches, the add_to_partial is gonna extract
  // the write shape and verify that it is the same shape as
  // the partialize we're adding to.
  //
  // Oh well.
  auto const& write_shape = nodes[id_out].op.get_partialize().write_shape;

  if(hrect_out.size() != write_shape.size()) {
    throw std::runtime_error("incorrect sizing");
  }

  vector<touchdim_t> ts;
  ts.reserve(hrect_out.size());
  for(int i = 0; i != hrect_out.size(); ++i) {
    auto const& [b,e] = hrect_out[i];
    auto const& d_out = write_shape[i];

    ts.push_back(touchdim_t {
      .d_inn = e-b,
      .d_out = d_out,
      .offset_inn = 0,
      .offset_out = b,
      .size = e-b
    });
  }

  touch_t touch {
    .selection = ts,
    .castable = optional<castable_t>()
  };

  add_to_partial(id_out, id_inn, touch, consume);
}

void taskgraph_t::add_to_partial_the_full_aggregate(
  int id_out,
  int id_inn,
  castable_t castable,
  bool consume)
{
  auto const& write_shape = nodes[id_out].op.get_partialize().write_shape;
  if(nodes[id_inn].op.tensor_size() != product(write_shape)) {
    throw std::runtime_error("invalid input size when adding aggregate");
  }

  vector<touchdim_t> ts;
  ts.reserve(write_shape.size());
  for(int i = 0; i != write_shape.size(); ++i) {
    ts.push_back(touchdim_t {
      .d_inn = write_shape[i],
      .d_out = write_shape[i],
      .offset_inn = 0,
      .offset_out = 0,
      .size = write_shape[i]
    });
  }

  touch_t touch {
    .selection = ts,
    .castable = optional<castable_t>(castable)
  };

  add_to_partial(id_out, id_inn, touch, consume);
}

uint64_t taskgraph_t::get_size_at(int id) const
{
  return nodes[id].op.tensor_size();
}

vector<int> taskgraph_t::get_order() const {
  vector<int> ready;
  ready.reserve(nodes.size() / 4);

  vector<int> counts(nodes.size(), -1);

  for(int i = 0; i != nodes.size(); ++i) {
    auto const& node = nodes[i];
    counts[i] = node.op.inputs().size();
    if(counts[i] == 0) {
      ready.push_back(i);
    }
  }

  vector<int> ret;
  ret.reserve(nodes.size());
  while(ready.size() > 0) {
    ret.push_back(ready.back());
    ready.pop_back();

    int const& id = ret.back();
    auto const& node = nodes[id];
    for(auto const& out: node.outs) {
      counts[out] -= 1;
      if(counts[out] == 0) {
        ready.push_back(out);
      }
    }
  }

  for(auto const& cnt: counts) {
    if(cnt != 0) {
      throw std::runtime_error("all counts should be zero");
    }
  }

  return ret;
}

bool taskgraph_t::all_zero_outs_is_save() const {
  for(auto const& node: nodes) {
    if(node.outs.size() == 0) {
      if(!node.is_save) {
        return false;
      }
    }
  }

  return true;
}

bool taskgraph_t::all_valid_partialize() const {
  for(auto const& node: nodes) {
    if(!node.op.is_valid_if_partialize()) {
      return false;
    }
  }
  return true;
}


int taskgraph_t::insert(op_t op, bool is_save) {
  int ret = nodes.size();

  for(auto inn: op.inputs()) {
    nodes[inn].outs.insert(ret);
  }

  nodes.emplace_back(op, is_save);

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

vector<vector<tuple<int, touch_t>>> taskgraph_t::partialize_t::as_touches_from() const {
  vector<vector<tuple<int, touch_t>>> rets;
  for(auto const& unit: units) {
    rets.emplace_back();
    auto& ret = rets.back();
    for(auto const& input: unit.inputs) {
      vector<touchdim_t> ts;
      ts.reserve(write_shape.size());
      for(int i = 0; i != write_shape.size(); ++i) {
        ts.push_back(touchdim_t {
          .d_inn = input.region[i].dim,
          .d_out = write_shape[i],
          .offset_inn = input.region[i].offset,
          .offset_out = unit.out_region[i].offset,
          .size = unit.out_region[i].size
        });
      }
      ret.emplace_back(
        input.id,
        touch_t {
          .selection = ts,
          .castable = unit.castable
        }
      );
    }
  }
  return rets;
}

bool taskgraph_t::partialize_t::valid() const
{
  int rank = write_shape.size();

  // Cut the write_shape hrect into a refined set of blocks
  // based on the partial units
  partition_t refinement = [&] {
    vector<partdim_t> partdims;
    partdims.reserve(rank);
    {
      vector<vector<uint64_t>> spans(rank);
      for(int i = 0; i != rank; ++i) {
        spans[i].push_back(write_shape[i]);
      }
      for(auto const& unit: units) {
        for(int i = 0; i != rank; ++i) {
          auto const& [offset, size] = unit.out_region[i];
          auto& ss = spans[i];
          if(offset != 0) {
            ss.push_back(offset);
          }
          ss.push_back(offset + size);
        }
      }
      for(vector<uint64_t>& ss: spans) {
        std::sort(ss.begin(), ss.end());
        vector_remove_duplicates(ss);
        partdims.push_back(partdim_t::from_spans(ss));
      }
    }
    return partition_t(partdims);
  }();

  // for each touch, increment the relevant write regions
  auto refinement_shape = refinement.block_shape();
  tensor_t<int> counts(
    refinement_shape,
    vector<int>(product(refinement_shape), 0));

  for(auto const& unit: units) {
    vector<tuple<uint64_t, uint64_t>> hrect;
    hrect.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      auto const& [offset, size] = unit.out_region[i];
      hrect.emplace_back(offset, offset + size);
    }

    vector<tuple<int,int>> region = refinement.get_exact_region(hrect);
    vector<int> index = vector_mapfst(region);
    do {
      counts.at(index) += 1;
    } while(increment_idxs_region(region, index));
  }

  // Check that the entire write shape is partitioned.
  //
  // If the out regions are not disjoint, then
  //   some num_touch will be bigger than one.
  // If some of the write_shape is not written to,
  //   then some nume_touch will be zero.

  for(auto const& num_touch: counts.get()) {
    if(num_touch != 1) {
      return false;
    }
  }
  return true;
}

void taskgraph_t::print() const {
  std::cout << "taskgraph[num nodes = " << nodes.size() << "]" << std::endl;
  std::cout << std::endl;

  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];

    std::cout << "node " << id;
    if(node.is_save) {
      std::cout << " (save)";
    }
    std::cout << std::endl;

    auto inputs = node.op.inputs();
    std::cout << "inputs: " << vector<int>(inputs.begin(), inputs.end()) << std::endl;
    std::cout << "tensor size: " << node.op.tensor_size() << std::endl;

    if(node.op.is_input()) {
      std::cout << "input" << std::endl;
    } else if(node.op.is_apply()) {
      std::cout << "apply" << std::endl;
    } else if(node.op.is_move()) {
      std::cout << "move" << std::endl;
    } else if(node.op.is_partialize()) {
      std::cout << "partialize" << std::endl;
    }

    std::cout << std::endl;
  }
}

std::ostream& operator<<(std::ostream& out, touchdim_t const& td) {
  out << "td[d_inn:" << td.d_inn << ",d_out:" << td.d_out;
  out << ",o_inn:" << td.offset_inn << ",o_out:" << td.offset_out;
  out << ",size:" << td.size << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, touch_t const& t) {
  out << t.castable << " " << t.selection;
  return out;
}

