#include "taskgraph.h"
#include "../base/copyregion.h"
#include "../base/hrect.h"

#include "einsummable.pb.h"

touch_t touch_t::simplify() const {
  vector<touchdim_t> new_selection;
  new_selection.push_back(selection[0]);

  auto is_dummy_dim = [](touchdim_t const& td) {
    auto const& [d_inn, d_out, o_inn, o_out, sz] = td;
    return d_inn == d_out && o_inn == 0 && o_out == 0 && d_inn == sz;
  };

  for(int i = 1; i != selection.size(); ++i) {
    if(is_dummy_dim(selection[i])) {
      int const& d = selection[i].d_inn;
      auto& [d_inn, d_out, o_inn, o_out, sz] = new_selection.back();
      d_inn *= d;
      d_out *= d;
      o_inn *= d;
      o_out *= d;
      sz    *= d;
    } else {
      new_selection.push_back(selection[i]);
    }
  }

  return touch_t {
    .selection = new_selection,
    .castable = castable
  };
}

// The compilation from graph to taskgraph is designed to
// automatically split up tensors so as to only move
// the specified elements.
//
// Their are three phases to worry about. The "compute" phase,
// the "form" phase, the "communicate" phase.
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
// The form phase grabs the input from the refinement.

struct multiple_placement_t {
  static multiple_placement_t from_single_placement(placement_t const& p);

  static multiple_placement_t make_refinement(vector<placement_t> const& ps);

  static multiple_placement_t make_refinement(vector<multiple_placement_t> const& ps);

  // deduce the required multiple placement of an einsummable's
  // input at which_input given that the einsummable is placed with
  // with join_placement
  static multiple_placement_t make_einsummable_input(
    placement_t const& join_placement,
    einsummable_t const& einsummable,
    int which_input);

  // deduce the required multiple_placement of a concat's
  // input at which_input given that the concat is placed with
  // join_placement
  static multiple_placement_t make_concat_input(
    placement_t const& join_placement,
    int concat_dim,
    vector<uint64_t> concat_parts,
    int which_input);
  static placement_t make_concat_input_placement(
    placement_t const& join_placement,
    int concat_dim,
    vector<uint64_t> concat_parts,
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
    tensor_t<vector<locid_t>> && t);

  multiple_tensor_t(multiple_placement_t p, int init_value = 0);

  int vec_at(int vec_index, int desired_loc) const;

  int const& at(vector<int> const& index, int desired_loc) const;

  int& at(vector<int> const& index, int desired_loc);

  partition_t partition;
  tensor_t<vector<locid_t>> tensor;

  tensor_t<int> to_tensor(placement_t const& placement);
};

std::ostream& operator<<(std::ostream& out, multiple_tensor_t::locid_t const& x);

struct taskgraph_make_state_t {
  taskgraph_make_state_t(graph_t const& graph, vector<placement_t> const& placements)
    : graph(graph), placements(placements)
  {}

  // the input compute graph
  graph_t const& graph;
  vector<placement_t> const& placements;

  // the output task graph
  taskgraph_t taskgraph;

  // Map from gid to the refined tensor formed from that gid.
  map<int, multiple_tensor_t> refined_tensors;

  // Partialize the data into the partial ids at outs.
  void relational_touch(
    vector<uint64_t>   const& offset_inn,
    vector<uint64_t>   const& offset_out,
    vector<uint64_t>   const& size,
    multiple_tensor_t  const& inns,
    multiple_tensor_t  const& outs);

  multiple_tensor_t initialize_partials(
    multiple_placement_t const& placement);

  // Grab gid from it's refined tensor in the form of the
  // given multiple placement
  multiple_tensor_t form_from_refinement(
    int gid,
    multiple_placement_t const& placement);
  // this dispatches to the multiple tensor form_from_refinement
  tensor_t<int> form_from_refinement(
    int gid,
    placement_t const& placement);
  // TODO: have a cache for this method

  tensor_t<int> form_concat(int gid);

  // this will form the inputs as required
  tensor_t<int> compute_einsummable(int gid);

  tensor_t<int> compute_input(int gid);

  tensor_t<int> form_relation(int gid);

  // create the refined_tensor object from the compute result;
  // save into refined_tensors
  void communicate(int gid, tensor_t<int> compute_result);

  multiple_placement_t construct_refinement_placement(int gid);

  // Note: the join_result can include agg'd dimensions, in which case
  //       a castable must be given
  multiple_tensor_t construct_refinement_tensor(
    placement_t const& join_placement,
    tensor_t<int> join_result,
    multiple_placement_t const& refinement,
    optional<castable_t> castable);
};

tuple<
  map<int, tensor_t<int> >, // for each input, the tids of the blocks
  map<int, tensor_t<int> >, // for each save id, the tids of the blocks
  taskgraph_t>              // the actual taskgraph
taskgraph_t::make(
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  taskgraph_make_state_t state(graph, placements);

  // maps from gid to tensor
  map<int, tensor_t<int>> inns;
  map<int, tensor_t<int>> saves;

  for(int gid: graph.get_order()) {
    graph_t::node_t const& node = graph.nodes[gid];

    tensor_t<int> relation = state.form_relation(gid);

    if(node.op.is_input()) {
      inns[gid] = relation;
    }

    if(node.op.is_save()) {
      saves[gid] = relation;
      // Set the taskgraph nodes to be saved
      for(int const& taskgraph_tid: relation.get()) {
        state.taskgraph.nodes[taskgraph_tid].is_save = true;
      }
    }

    if(node.outs.size() > 0) {
      state.communicate(gid, std::move(relation));
    }
  }

  if(!state.taskgraph.all_zero_outs_is_save()) {
    throw std::runtime_error("In taskgraph_t::make: non-saved outputs");
  }

  return {std::move(inns), std::move(saves), std::move(state.taskgraph)};
}

void taskgraph_make_state_t::relational_touch(
  vector<uint64_t>   const& offset_inn,
  vector<uint64_t>   const& offset_out,
  vector<uint64_t>   const& size,
  multiple_tensor_t  const& inn_locids,
  multiple_tensor_t  const& out_locids)
{
  // Some naming conventions:
  //   hrect:       a uint64_t hyper rectangular indexing actual tensor indices
  //   region:      a int hyper rectangular indexing blocks
  //   inn/out:     inn = read only, out = being written
  //                also: inn means with respect to the input indexing,
  //                      out means with respect to the output indexing
  //   big/med/sml: big = the entire output,
  //                med: a smaller output part or bigger input part,
  //                sml: a smaller input part
  //                (big/medium/small)
  //   exa/uxa:     exact: all of this is the region being copied
  //                unexact: not all of this is the region being copied

  // Form hrect_out_big_exa and get region_out_big_uxa
  // For every out block in region_out_big_uxa:
  //   get hrect_out_mid_uxa from the corr index
  //   get hrect_out_mid_exa by taking the intersection
  //   get hrect_inn_mid_uxa by fixing the offsets
  //   get region_inn_mid_uxa
  //   For every input block in region_inn_mid_uxa:
  //     get hrect_inn_sml_exa from the corr index
  //
  //     get the partial id at the out index
  //     form the touch
  //     for each loc,
  //       copy hrect_inn_sml_exa into hrect_out_mid_uxa
  partition_t const& part_inn = inn_locids.partition;
  partition_t const& part_out = out_locids.partition;

  using hrect_t  = vector<tuple<uint64_t, uint64_t>>;
  using shape_t  = vector<uint64_t>;

  using region_t = vector<tuple<int, int>>;
  using index_t  = vector<int>;

  using locid_t = multiple_tensor_t::locid_t;

  hrect_t hrect_out_big_exa;
  {
    hrect_out_big_exa.reserve(size.size());
    for(int i = 0; i != size.size(); ++i) {
      auto const& off_out = offset_out[i];
      auto const& sz = size[i];
      hrect_out_big_exa.emplace_back(off_out, off_out + sz);
    }
  }

  region_t region_out_big_uxa = part_out.get_region(hrect_out_big_exa);
  index_t out_index = vector_mapfst(region_out_big_uxa);
  do {
    vector<locid_t> const& out_lis = out_locids.tensor.at(out_index);

    hrect_t hrect_out_mid_uxa = part_out.get_hrect(out_index);
    hrect_t hrect_out_mid_exa = hrect_intersect(hrect_out_mid_uxa, hrect_out_big_exa);

    shape_t shape_out_mid_uxa = hrect_shape(hrect_out_mid_uxa);

    shape_t out_mid_offset;
    out_mid_offset.reserve(size.size());
    for(int i = 0; i != size.size(); ++i) {
      out_mid_offset.push_back(
        std::get<0>(hrect_out_mid_exa[i]) - std::get<0>(hrect_out_mid_uxa[i]));
    }

    // convert hrect_out_mid_exa into something with respect to the inn indicess
    hrect_t hrect_inn_mid;
    hrect_inn_mid.reserve(size.size());
    for(int i = 0; i != size.size(); ++i) {
      auto [b,e] = hrect_out_mid_exa[i];
      uint64_t const& off_inn = offset_inn[i];
      uint64_t const& off_out = offset_out[i];
      b = (b - off_out) + off_inn;
      e = (e - off_out) + off_inn;
      hrect_inn_mid.emplace_back(b,e);
    }

    region_t region_inn_mid = part_inn.get_exact_region(hrect_inn_mid);
    index_t inn_index = vector_mapfst(region_inn_mid);
    do {
      hrect_t hrect_inn_sml_uxa = part_inn.get_hrect(inn_index);
      hrect_t hrect_inn_sml_exa = hrect_intersect(hrect_inn_sml_uxa, hrect_inn_mid);

      shape_t shape_inn_sml_uxa = hrect_shape(hrect_inn_sml_uxa);
      shape_t shape_inn_sml_exa = hrect_shape(hrect_inn_sml_exa);

      shape_t inn_sml_offset;
      inn_sml_offset.reserve(size.size());
      for(int i = 0; i != size.size(); ++i) {
        inn_sml_offset.push_back(
          std::get<0>(hrect_inn_sml_exa[i]) - std::get<0>(hrect_inn_sml_uxa[i]));
      }

      // copy from hrect_inn_sml_uxa to
      //           hrect_out_mid_uxa
      vector<touchdim_t> selection;
      selection.reserve(size.size());
      for(int i = 0; i != size.size(); ++i) {
        selection.push_back(touchdim_t {
          .d_inn      = shape_inn_sml_uxa[i],
          .d_out      = shape_out_mid_uxa[i],
          .offset_inn = inn_sml_offset[i],
          .offset_out = out_mid_offset[i],
          .size       = shape_inn_sml_exa[i]
        });
      }
      touch_t touch {
        .selection = selection,
        .castable = std::nullopt
      };
      // Note: one could put a castable argument here

      for(auto const& [loc,out_id]: out_lis) {
        int const& inn_id = inn_locids.at(inn_index, loc);
        // TODO: when to consume in add_to_partial
        taskgraph.add_to_partial(out_id, inn_id, touch);
      }
    } while(increment_idxs_region(region_inn_mid, inn_index));
  } while(increment_idxs_region(region_out_big_uxa, out_index));
}

multiple_tensor_t taskgraph_make_state_t::initialize_partials(
  multiple_placement_t const& placement)
{
  auto out_shape = placement.partition.block_shape();
  multiple_tensor_t packaged_ret(
    placement.partition,
    tensor_t<vector<multiple_tensor_t::locid_t>>(out_shape));

  auto& ret = packaged_ret.tensor;
  vector<int> out_index(out_shape.size(), 0);
  do {
    auto const& locs = placement.locations.at(out_index);
    auto write_shape = placement.partition.tensor_shape_at(out_index);

    // initialize the partials to be written
    auto& ret_at = ret.at(out_index);
    ret_at.reserve(locs.size());
    for(auto const& loc: locs) {
      int builder_id = taskgraph.new_partial(loc, write_shape);
      ret_at.push_back(
        multiple_tensor_t::locid_t {
          .loc = loc,
          .id  = builder_id
        });
    }
  } while(increment_idxs(out_shape, out_index));

  return packaged_ret;
}

multiple_tensor_t
taskgraph_make_state_t::form_from_refinement(
  int gid,
  multiple_placement_t const& placement)
{
  multiple_tensor_t const& refine_tensor = refined_tensors.at(gid);

  // verify that the refinement partition
  // is actually a refinement of the given placement
  if(!refine_tensor.partition.refines(placement.partition)) {
    throw std::runtime_error("refine partition not actually a refinement");
  }

  if(placement.partition == refine_tensor.partition) {
    return refine_tensor;
  }

  // initialize multiple_tensor_t for the outs by creating new partializes
  multiple_tensor_t ret = initialize_partials(placement);

  // now add to all the partial builders inside ret
  vector<uint64_t> total_shape = placement.partition.total_shape();
  vector<uint64_t> offsets(total_shape.size(), 0);
  relational_touch(
    offsets, offsets, placement.partition.total_shape(),
    refine_tensor,
    ret);

  return ret;
}

tensor_t<int>
taskgraph_make_state_t::form_from_refinement(
  int gid,
  placement_t const& pl)
{
  multiple_placement_t mpl =
    multiple_placement_t::from_single_placement(pl);
  return form_from_refinement(gid, mpl).to_tensor(pl);
}

// Example
//   big { 10, 10, 10 }
//   sml { 5, 2, 3, 10, 3, 3, 3, 1}
//   rs  { 3, 4, 8 }
vector<tuple<int,int>> _get_sum_breaks(
  vector<uint64_t> big,
  vector<uint64_t> sml)
{
  vector<int> rs;
  int j = 0;
  for(int i = 0; i != big.size(); ++i) {
    auto sz = big[i];
    for(; sz != 0; ++j) {
      sz -= sml[j];
    }
    rs.push_back(j);
  }

  vector<tuple<int,int>> ret;
  int b = 0;
  for(int i = 0; i != rs.size(); ++i) {
    ret.emplace_back(b, rs[i]);
    b = rs[i];
  }

  return ret;
}

tensor_t<int>
taskgraph_make_state_t::form_concat(int gid) {
  graph_t::node_t const& node = graph.nodes.at(gid);
  placement_t const& pl = placements.at(gid);

  if(!node.op.is_concat()) {
    throw std::runtime_error("from_concat needs concat node");
  }

  int rank = node.op.out_rank();

  auto const& concat = node.op.get_concat();
  int const& dim = concat.dim;
  auto const dim_parts = concat.dim_parts();

  // split the partition of this node along the concat dimension
  vector<partdim_t> split_partdims = pl.partition.partdims;
  split_partdims[dim] = partdim_t::unions({
    split_partdims[dim],
    partdim_t::from_sizes(dim_parts)
  });
  partition_t split_partition(split_partdims);

  // TODO: calling form_from_refinement could be done on a part by
  //       part basis:
  //       (in1|in2  |in3    )   <- the inputs being concated
  //       (x x|x x x|x x x x)   <- split_partition
  //       (x x|x x x|x x|x x)   <- pl partition
  //        ^yes^yes  ^no ^no

  if(pl.partition == split_partition) {
    // Note: if form_from_refinement had cache support,
    //       this would dip into that cache.

    vector<tensor_t<int>> ts;

    for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
      int const& inn_gid = node.inns[which_inn];
      placement_t inn_pl = multiple_placement_t::make_concat_input_placement(
        pl,
        dim,
        dim_parts,
        which_inn);

      ts.push_back(form_from_refinement(inn_gid, inn_pl));
    }

    return tensor_t<int>::concat(dim, ts);
  }

  // form the output partials
  multiple_placement_t mpl =
    multiple_placement_t::from_single_placement(pl);
  multiple_tensor_t ret = initialize_partials(mpl);

  // store 0, dim_parts[0], dim_parts[0] + dim_parts[1], ...
  vector<uint64_t> dim_offset = concat.get_offsets();

  // copying the entirety of each input so
  // the offset is the same for all inns
  vector<uint64_t> offset_inn(rank, 0);

  // for each input, call relational_touch to copy all of the input
  // into the corresponding part of ret
  for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
    int const& inn_gid = node.inns[which_inn];
    multiple_tensor_t const& inn = refined_tensors.at(inn_gid);

    vector<uint64_t> offset_out(rank, 0);
    offset_out[dim] = dim_offset[which_inn];

    relational_touch(
      offset_inn,
      offset_out,
      inn.partition.total_shape(),
      inn,
      ret);
  }

  return ret.to_tensor(pl);
}

tensor_t<int>
taskgraph_make_state_t::compute_input(int gid) {
  graph_t::node_t const& node = graph.nodes[gid];
  placement_t const& pl = placements[gid];

  if(!node.op.is_input()) {
    throw std::runtime_error("compute_input must have input node");
  }

  auto shape = pl.block_shape();
  tensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);
  do {
    int const& loc = pl.locations.at(index);
    auto subtensor_shape = pl.partition.tensor_shape_at(index);
    ret.at(index) = taskgraph.insert_input(loc, subtensor_shape);
  } while(increment_idxs(shape, index));

  return ret;
}

tensor_t<int>
taskgraph_make_state_t::compute_einsummable(int gid)
{
  graph_t::node_t const& node = graph.nodes[gid];
  placement_t const& pl = placements[gid];

  if(!node.op.is_einsummable()) {
    throw std::runtime_error("compute_einsummable must have einsummable node");
  }

  einsummable_t const& base_einsummable = std::get<einsummable_t>(node.op.op);

  // Get the inputs
  vector<multiple_tensor_t> inputs;
  inputs.reserve(node.inns.size());
  for(int which_input = 0; which_input != node.inns.size(); ++which_input) {
    int const& inn_gid = node.inns[which_input];

    auto inn_placement = multiple_placement_t::make_einsummable_input(
      pl, base_einsummable, which_input);

    inputs.push_back(this->form_from_refinement(inn_gid, inn_placement));
  }

  auto shape = pl.block_shape();
  tensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);

  do {
    int const& loc = pl.locations.at(index);

    vector<int> inns;
    inns.reserve(inputs.size());
    auto inn_idxs = base_einsummable.input_idxs(index);
    for(int i = 0; i != inputs.size(); ++i) {
      auto const& inn_tensor = inputs[i];
      auto const& inn_idx = inn_idxs[i];
      inns.push_back(inn_tensor.at(inn_idx, loc));
    }

    auto subtensor_shape = pl.partition.tensor_shape_at(index);
    ret.at(index) = taskgraph.insert_einsummable(
      loc,
      einsummable_t::with_new_shape(base_einsummable, subtensor_shape),
      inns);
  } while(increment_idxs(shape, index));

  return ret;
}

tensor_t<int>
taskgraph_make_state_t::form_relation(int gid)
{
  graph_t::node_t const& node = graph.nodes.at(gid);

  if(node.op.is_input()) {
    return compute_input(gid);
  }
  if(node.op.is_formation()) {
    int inn_gid = node.inns[0];
    return form_from_refinement(inn_gid, placements.at(gid));
  }
  if(node.op.is_concat()) {
    return form_concat(gid);
  }
  if(node.op.is_einsummable()) {
    return compute_einsummable(gid);
  }

  throw std::runtime_error("state form relation should not reach");
}

void
taskgraph_make_state_t::communicate(int join_gid, tensor_t<int> join_result)
{
  multiple_placement_t usage_placement = construct_refinement_placement(join_gid);

  optional<castable_t> maybe_castable;
  auto const& node = graph.nodes[join_gid];
  if(node.op.has_aggregation()) {
    maybe_castable = node.op.get_castable();
  }

  multiple_tensor_t ret = construct_refinement_tensor(
    placements[join_gid],
    join_result,
    usage_placement,
    maybe_castable);

  refined_tensors.insert({join_gid, ret});
}

multiple_placement_t
taskgraph_make_state_t::construct_refinement_placement(int join_gid)
{
  auto const& join_node = graph.nodes[join_gid];

  vector<multiple_placement_t> usage_placements;
  usage_placements.reserve(2*join_node.outs.size());
  for(auto const& out_gid: join_node.outs) {
    auto const& out_node = graph.nodes[out_gid];
    auto const& out_pl   = placements[out_gid];
    if(out_node.op.is_formation()) {
      usage_placements.push_back(
        multiple_placement_t::from_single_placement(out_pl));
    } else if(out_node.op.is_einsummable()) {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage placements to collect
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          usage_placements.push_back(
            multiple_placement_t::make_einsummable_input(
              out_pl,
              out_node.op.get_einsummable(),
              which_input));
        }
      }
    } else if(out_node.op.is_concat()) {
      auto concat = out_node.op.get_concat();
      auto dim_parts = concat.dim_parts();
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          usage_placements.push_back(
            multiple_placement_t::make_concat_input(
              out_pl,
              concat.dim,
              dim_parts,
              which_input));
        }
      }
    } else {
      throw std::runtime_error(
          "taskgraph state construct refinement placement: "
          "should not happen");
    }
  }

  return multiple_placement_t::make_refinement(usage_placements);
}

multiple_tensor_t
taskgraph_make_state_t::construct_refinement_tensor(
  placement_t const& join_placement,
  tensor_t<int> join_result,
  multiple_placement_t const& refinement,
  optional<castable_t> maybe_castable)
{
  auto const& _join_partdims = join_placement.partition.partdims;

  int join_rank = join_placement.partition.block_shape().size();
  int out_rank = refinement.partition.block_shape().size();
  int agg_rank = join_rank - out_rank;

  auto refinement_shape = refinement.partition.block_shape();

  // the output (of this method) to be added to refined_tensors
  // initialize every id to -1 initially;
  // the locs will track with the refinement locs
  multiple_tensor_t refined_tensor(refinement, -1);

  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  optional<partition_t> maybe_agg_partition;
  if(agg_rank > 0) {
    maybe_agg_partition = vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end());

    if(!maybe_castable) {
      throw std::runtime_error("ann agg is happening but no castable");
    }
  } else {
    if(maybe_castable) {
      throw std::runtime_error("no agg but castable given");
    }
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
          castable_t& castable = maybe_castable.value();
          int local_aggd_id =
            taskgraph.insert_consumed_aggregate(loc, castable, ids);
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
        .castable = maybe_castable
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
        .castable = maybe_castable
      };
      // ref_id by reference!
      auto add_out_to_refinement = [&](int loc, int& ref_id, int partial_id) {
        maybe_init_ref_id_as_partial(ref_id, loc);
        taskgraph.add_to_partial(ref_id, partial_id, _out_to_refinement_touch, false);
      };
      // }}}

      // out_tensor_shape  (o) what was created by the join node
      // selection_shape   (s) intersection of out and refinement
      // refinement_shape  (r) what is being formed here at all the required locs
      //
      // Note: the selection shape is the finest
      //
      // s == o == r  case 0
      // s == o << r  case 1
      // s == r << o  case 2
      // s << o == r  (can't happen: if o == r, s == o and s == r)
      // s << o != r  case 3
      if(partials.size() == 1) {
        auto const& [partial_loc, partial_id] = *partials.begin();
        if(vector_equal(selection_shape, out_tensor_shape)) {
          if(vector_equal(out_tensor_shape, refinement_shape)) {
            // case 0: copy right into the refined tensor
            for(int loc: required_locs) {
              if(loc == partial_loc) {
                refined_tensor.at(r_index, loc) = partial_id;
              } else {
                refined_tensor.at(r_index, loc) =
                  taskgraph.insert_move(partial_loc, loc, partial_id);
              }
            }
          } else {
            // case 1: copy the partial into the refinement
            for(int loc: required_locs) {
              int const& subset_id = partial_id;
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
        } else if(vector_equal(selection_shape, refinement_shape)) {
          // case 2: subset then copy right into the refined tensor
          int subset_id = insert_out_to_selection(partial_loc, partial_id);
          // (subsert_id = ref_id)
          for(int loc: required_locs) {
            if(loc == partial_loc) {
              refined_tensor.at(r_index, loc) = subset_id;
            } else {
              refined_tensor.at(r_index, loc) =
                taskgraph.insert_move(partial_loc, loc, subset_id);
            }
          }
        } else {
          // case 3
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
          // (this way it doesn't matter if helper functions
          //  take ref_id by reference or not)
          int& ref_id = refined_tensor.at(r_index, loc);
          maybe_init_ref_id_as_partial(ref_id, loc);
        }

        if(vector_equal(selection_shape, out_tensor_shape)) {
          if(vector_equal(out_tensor_shape, refinement_shape)) {
            // case 0: aggregate the full input
            for(auto const& [partial_loc, partial_id]: partials) {
              for(int loc: required_locs) {
                int builder = refined_tensor.at(r_index, loc);
                if(loc == partial_loc) {
                  // partial_id will also be moved since required_locs.size() > 1.
                  // So don't consume the output!
                  castable_t castable = maybe_castable.value();
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, partial_id, castable, false);
                } else {
                  // Here, the moved data can be consumed
                  int moved_partial_id =
                    taskgraph.insert_move(partial_loc, loc, partial_id);
                  castable_t castable = maybe_castable.value();
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, moved_partial_id, castable, true);
                }
              }
            }
          } else {
            // case 1
            for(auto const& [partial_loc, partial_id]: partials) {
              int const& subset_id = partial_id;
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
        } else if(vector_equal(selection_shape, refinement_shape)) {
          // case 2: subset then aggregate into the refined tensor
          for(auto const& [partial_loc, partial_id]: partials) {
            if(required_locs.size() == 1) {
              int loc = *required_locs.begin();
              int builder = refined_tensor.at(r_index, loc);
              if(loc == partial_loc) {
                add_out_to_refinement(loc, builder, partial_id);
              } else {
                int subset_id = insert_out_to_selection(partial_loc, partial_id);
                int moved_subset_id =
                  taskgraph.insert_move(partial_loc, loc, subset_id);
                castable_t castable = maybe_castable.value();
                // The moved data can be consumed
                taskgraph.add_to_partial_the_full_aggregate(
                  builder, moved_subset_id, castable, true);
              }
            } else {
              // Since the subset will be used multiple times:
              // 1. Form the subset
              // 2. Move it to where it needs to be
              // 3. If it needs to be at partial_loc, agg the subset into that loc
              int subset_id = insert_out_to_selection(partial_loc, partial_id);
              for(int loc: required_locs) {
                int builder = refined_tensor.at(r_index, loc);
                if(loc == partial_loc) {
                  // subset_id will also be moved since required_locs.size() > 1.
                  // So don't consume the output!
                  castable_t castable = maybe_castable.value();
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, subset_id, castable, false);
                } else {
                  // Here, the moved data can be consumed
                  int moved_subset_id =
                    taskgraph.insert_move(partial_loc, loc, subset_id);
                  castable_t castable = maybe_castable.value();
                  taskgraph.add_to_partial_the_full_aggregate(
                    builder, moved_subset_id, castable, true);
                }
              }
            }
          }
        } else {
          // case 3
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

  return refined_tensor;
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

placement_t
multiple_placement_t::make_concat_input_placement(
  placement_t const& join_placement,
  int concat_dim,
  vector<uint64_t> concat_parts,
  int which_input)
{
  auto const& partdim = join_placement.partition.partdims[concat_dim];

  vector<tuple<int,int>> breaks = _get_sum_breaks(
    partdim.sizes(),
    concat_parts);

  auto block_shape = join_placement.block_shape();
  vector<tuple<int,int>> region;
  region.reserve(block_shape.size());
  for(auto const& d: block_shape) {
    region.emplace_back(0, d);
  }

  region[concat_dim] = breaks[which_input];

  return join_placement.subset(region);
}

multiple_placement_t multiple_placement_t::make_concat_input(
  placement_t const& join_placement,
  int concat_dim,
  vector<uint64_t> concat_parts,
  int which_input)
{
  return multiple_placement_t::from_single_placement(
    make_concat_input_placement(
      join_placement, concat_dim, concat_parts, which_input));
}

multiple_tensor_t::multiple_tensor_t(
  partition_t p,
  tensor_t<vector<locid_t>> && t)
  : partition(p), tensor(std::move(t))
{
  if(tensor.get_shape() != p.block_shape()){
    throw std::runtime_error("multiple_tensor_t incorrect shape");
  }
}

multiple_tensor_t::multiple_tensor_t(multiple_placement_t p, int init_value)
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

int multiple_tensor_t::vec_at(int vec_index, int desired_loc) const {
  for(auto const& [loc,id]: tensor.get()[vec_index]) {
    if(desired_loc == loc) {
      return id;
    }
  }
  throw std::runtime_error("multiple_tensor_t::vec_at could not get");
}

int const& multiple_tensor_t::at(vector<int> const& index, int desired_loc) const
{
  for(auto const& [loc,id]: tensor.at(index)) {
    if(desired_loc == loc) {
      return id;
    }
  }
  throw std::runtime_error("multiple_tensor_t::at could not get");
}

int& multiple_tensor_t::at(vector<int> const& index, int desired_loc) {
  for(auto& [loc,id]: tensor.at(index)) {
    if(desired_loc == loc) {
      return id;
    }
  }
  throw std::runtime_error("multiple_tensor_t::at could not get");
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

std::ostream& operator<<(std::ostream& out, multiple_tensor_t::locid_t const& x)
{
  auto const& [loc,id] = x;
  out << "loc" << loc << "id" << id;
  return out;
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

int taskgraph_t::num_locs() const {
  int ret = 0;
  for(auto const& node: nodes) {
    ret = std::max(ret, 1 + node.op.output_loc());
    if(node.op.is_move()) {
      ret = std::max(ret, 1 + node.op.get_move().src);
    }
  }
  return ret;
}

uint64_t taskgraph_t::total_elems_moved() const {
  uint64_t ret = 0;
  for(auto const& node: nodes) {
    if(node.op.is_move()) {
      ret += node.op.get_move().size;
    }
  }
  return ret;
}

uint64_t taskgraph_t::total_flops() const {
  uint64_t ret = 0;
  for(auto const& node: nodes) {
    if(node.op.is_apply()) {
      ret += product(node.op.get_apply().einsummable.join_shape);
    }
  }
  return ret;
}

string taskgraph_t::to_wire() const {
  es_proto::TaskGraph tg;

  for(auto const& node: nodes) {
    es_proto::TaskGraphNode* n = tg.add_nodes();

    if(node.op.is_input()) {
      auto const& [loc,size] = node.op.get_input();

      es_proto::TGInput* i = n->mutable_input();
      i->set_loc(loc);
      i->set_size(size);
    } else if(node.op.is_apply()) {
      auto const& [loc, inns, einsummable] = node.op.get_apply();

      es_proto::TGApply* a = n->mutable_apply();

      a->set_loc(loc);

      for(int const& inn: inns) {
        a->add_inns(inn);
      }

      es_proto::Einsummable* e = a->mutable_einsummable();
      einsummable.to_proto(*e);
    } else if(node.op.is_move()) {
      auto const& [src,dst,inn,size] = node.op.get_move();

      es_proto::TGMove* m = n->mutable_move();
      m->set_src(src);
      m->set_dst(dst);
      m->set_inn(inn);
      m->set_size(size);
    } else if(node.op.is_partialize()) {
      auto const& [loc, write_shape, units] = node.op.get_partialize();

      es_proto::TGPartialize* t = n->mutable_partialize();
      t->set_loc(loc);

      for(uint64_t const& d: write_shape){
        t->add_write_shape(d);
      }

      for(auto const& [castable, out_region, inputs]: units) {
        es_proto::TGPartialUnit* u = t->add_units();

        if(castable) {
          u->set_castable(write_with_ss(castable.value()));
        }

        for(auto const& [offset, size]: out_region) {
          es_proto::OutRegionDim* s = u->add_out_region();
          s->set_offset(offset);
          s->set_size(size);
        }

        for(auto const& [id, consumable, region]: inputs) {
          es_proto::TGPartialInn* ii = u->add_inputs();
          ii->set_id(id);
          ii->set_consumable(consumable);
          for(auto const& [dim, offset]: region) {
            es_proto::InnRegionDim* i = ii->add_region();
            i->set_dim(dim);
            i->set_offset(offset);
          }
        }
      }
    } else {
      throw std::runtime_error("should not reach");
    }

    n->set_is_save(node.is_save);
  }

  string ret;
  tg.SerializeToString(&ret);
  return ret;
}

taskgraph_t taskgraph_t::from_wire(string const& str) {
  es_proto::TaskGraph tg;
  if(!tg.ParseFromString(str)) {
    throw std::runtime_error("could not parse taskgraph!");
  }

  taskgraph_t ret;

  for(int id = 0; id != tg.nodes_size(); ++id) {
    es_proto::TaskGraphNode const& n = tg.nodes(id);

    bool is_save = n.is_save();

    if(n.has_input()) {
      auto const& i = n.input();
      ret.nodes.emplace_back(
        op_t(input_t { i.loc(), i.size() }),
        is_save);
    } else if(n.has_apply()) {
      auto const& a = n.apply();

      auto rv = a.inns();
      vector<int> inns(rv.begin(), rv.end());

      einsummable_t e = einsummable_t::from_proto(a.einsummable());

      ret.nodes.emplace_back(
        op_t(apply_t { a.loc(), inns, e }),
        is_save);
    } else if(n.has_move()) {
      auto const& m = n.move();
      ret.nodes.emplace_back(
        op_t(move_t { m.src(), m.dst(), m.inn(), m.size() }),
        is_save);
    } else if(n.has_partialize()) {
      auto const& p = n.partialize();

      auto const& _ws = p.write_shape();
      vector<uint64_t> write_shape(_ws.begin(), _ws.end());

      vector<partialize_t::partial_unit_t> units;
      for(auto const& u: p.units()) {
        optional<castable_t> castable = std::nullopt;
        if(u.has_castable()) {
          castable = parse_with_ss<castable_t>(u.castable());
        }

        vector<partialize_t::out_regiondim_t> out_region;
        for(auto const& o: u.out_region()) {
          out_region.push_back(partialize_t::out_regiondim_t {
            .offset = o.offset(),
            .size = o.size()
          });
        }

        vector<partialize_t::input_op_t> inputs;
        for(auto const& ii: u.inputs()) {
          vector<partialize_t::inn_regiondim_t> inn_region;
          for(auto const& i: ii.region()) {
            inn_region.push_back(partialize_t::inn_regiondim_t {
              .dim = i.dim(),
              .offset = i.offset()
            });
          }

          inputs.push_back(partialize_t::input_op_t {
            .id = ii.id(),
            .consumable = ii.consumable(),
            .region = inn_region
          });
        }

        units.push_back(partialize_t::partial_unit_t {
          .castable = castable,
          .out_region = out_region,
          .inputs = inputs
        });
      }

      partialize_t partialize {
        .loc = p.loc(),
        .write_shape = write_shape,
        .units = units
      };
      ret.nodes.emplace_back(op_t(partialize), is_save);
    } else {
      throw std::runtime_error("should not happen");
    }
  }

  // Note: The nodes are sent over the wire in order,
  //       but that order is not necc a graph order.
  //       Therefore, don't call insert.
  //       Instead, directly put in all the nodes and then
  //       add all outgoing edges

  for(int id = 0; id != ret.nodes.size(); ++id) {
    auto const& op = ret.nodes[id].op;
    for(auto inn: op.inputs()) {
      ret.nodes[inn].outs.insert(id);
    }
  }

  return ret;
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

vector<vector<tuple<int, touch_t>>>
taskgraph_t::partialize_t::as_touches_from() const {
  vector<vector<tuple<int, touch_t>>> rets;
  for(int u = 0; u != units.size(); ++u) {
    auto const& unit = units[u];

    rets.emplace_back();
    auto& ret = rets.back();

    for(int i = 0; i != unit.inputs.size(); ++i) {
      ret.push_back(get_touch(u, i));
    }
  }
  return rets;
}

tuple<int, touch_t>
taskgraph_t::partialize_t::get_touch(int which_unit, int which_touch) const
{
  auto const& unit = units[which_unit];
  auto const& input = unit.inputs[which_touch];
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
  return {
    input.id,
    touch_t { .selection = ts, .castable = unit.castable }
  };
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

bool taskgraph_t::partialize_t::is_straight_copy() const {
  if(units.size() == 1 && units[0].inputs.size() == 1)
  {
    auto [_, touch] = get_touch(0, 0);
    for(auto const& t: touch.selection) {
      if(t.d_inn == t.d_out &&
         t.offset_inn == t.offset_out &&
         t.size == t.d_inn)
      {
        // copying the whole dimension exatly
      } else {
        return false;
      }
    }
    // All dimensions are being copied exactly,
    // this whole partialize is just a copy
    return true;
  }

  return false;
}

void taskgraph_t::print() const {
  std::cout << "taskgraph[num nodes = " << nodes.size() << "]" << std::endl;
  std::cout << std::endl;

  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];

    std::cout << "node " << id;
    if(node.is_save) {
      std::cout << " (save to loc " << node.op.output_loc() << ")";
    }
    std::cout << std::endl;

    auto inputs = node.op.inputs();
    std::cout << "inputs: " << vector<int>(inputs.begin(), inputs.end()) << std::endl;
    std::cout << "tensor size: " << node.op.tensor_size() << std::endl;

    if(node.op.is_input()) {
      auto const& [loc, _0] = node.op.get_input();
      std::cout << "input | loc[" << loc << "]" << std::endl;
    } else if(node.op.is_apply()) {
      auto const& [loc, _0, _1] = node.op.get_apply();
      std::cout << "apply | loc[" << loc << "]" << std::endl;
    } else if(node.op.is_move()) {
      auto const& [src, dst, _0, _1] = node.op.get_move();
      std::cout << "move | loc[" << src << "] -> loc[" << dst << "]" << std::endl;
    } else if(node.op.is_partialize()) {
      int loc = node.op.output_loc();
      std::cout << "partialize | loc[" << loc << "]" << std::endl;
    }

    std::cout << std::endl;
  }
}

void taskgraph_t::print_graphviz(std::ostream& out) const {
  vector<string> colors{
    "#61B292",
    "#AED09E",
    "#F1E8A7",
    "#A8896C",
    "#A8D8EA",
    "#AA96DA",
    "#FCBAD3",
    "#FFFFD2"
  };

  print_graphviz(out, colors);
}
void taskgraph_t::print_graphviz(
  std::ostream& out,
  vector<string> const& colors) const
{
  using std::endl;

  string tab = "  ";
  out << "digraph {" << endl;

  for(int id = 0; id != nodes.size(); ++id) {
    node_t const& node = nodes[id];
    op_t const& op = node.op;

    string label;
    string color = "";

    // set label and color
    if(op.is_input()) {
      auto const& [loc, _] = node.op.get_input();
      if(loc < colors.size()) {
        color = colors[loc];
      }
      label = "input" + write_with_ss(id) + "@loc" + write_with_ss(loc);
    } else if(op.is_apply()) {
      auto const& [loc, _, e] = node.op.get_apply();
      if(loc < colors.size()) {
        color = colors[loc];
      }
      label = "apply" + write_with_ss(id) + "@loc["
        + write_with_ss(loc) + "]" + write_with_ss(e);
    } else if(op.is_move()) {
      auto const& [src, dst, _0, _1] = node.op.get_move();
      string src_ = write_with_ss(src);
      string dst_ = write_with_ss(dst);
      label = "move" + write_with_ss(id) + "@loc" + src_ + "->" + dst_;
    } else if(op.is_partialize()) {
      int loc = node.op.output_loc();
      if(loc < colors.size()) {
        color = colors[loc];
      }
      label = "partialize" + write_with_ss(id) + "@loc" + write_with_ss(loc);
    }

    out << tab
      << "n" << id
      << " [style=filled,label=\"" << label << "\"";
    if(color != "") {
      out << ",color=\"" << color << "\"";
    }
    out << "]" << endl;

    // print out the edges
    if(op.is_input()) {
      // no edges to id
    } else if(op.is_apply()) {
      auto const& inns = op.get_apply().inns;
      for(int i = 0; i != inns.size(); ++i) {
        int const& inn_id = inns[i];
        out << tab << "n" << inn_id << " -> " << "n" << id;
        out << "[label=\"" << write_with_ss(i) << "\"]" << endl;
      }
    } else if(op.is_move()) {
      int const& inn_id = op.get_move().inn;
      out << tab << "n" << inn_id << " -> " << "n" << id;
    } else if(op.is_partialize()) {
      for(auto const& inn_id: op.inputs()) {
        out << tab << "n" << inn_id << " -> " << "n" << id;
      }
    }
  }
  out << "}" << endl;
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

