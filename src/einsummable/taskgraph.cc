#include "taskgraph.h"
#include "../base/copyregion.h"
#include "../base/hrect.h"

#include "einsummable.pb.h"

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

struct multiple_tensor_t {
  struct locid_t {
    int loc;
    int id;
  };

  multiple_tensor_t(
    partition_t p,
    vtensor_t<vector<locid_t>> && t);

  multiple_tensor_t(multiple_placement_t p, int init_value = 0);

  int vec_at(int vec_index, int desired_loc) const;

  int const& at(vector<int> const& index, int desired_loc) const;

  int& at(vector<int> const& index, int desired_loc);

  partition_t partition;
  vtensor_t<vector<locid_t>> tensor;
  // Note: it is possble to have empty locid sets

  vtensor_t<int> to_tensor(placement_t const& placement);
};

std::ostream& operator<<(std::ostream& out, multiple_tensor_t::locid_t const& x);

struct tg_region_t {
  vector<tuple<int,int>> region;

  std::size_t hash() const;

  vector<int> shape() const;
};

template <> struct std::hash<tg_region_t> {
  inline std::size_t operator()(tg_region_t const& t) const
  {
    return t.hash();
  }
};

bool operator==(tg_region_t const& lhs, tg_region_t const& rhs);
bool operator!=(tg_region_t const& lhs, tg_region_t const& rhs);

partition_t convert_squeezer_partition(
  vector<uint64_t> new_shape,
  partition_t const& part)
{
  vector<partdim_t> core_pds;
  for(partdim_t const& pd: part.partdims) {
    if(pd.total() != 1) {
      core_pds.push_back(pd);
    }
  }
  auto iter = core_pds.begin();
  vector<partdim_t> ret_pds;
  ret_pds.reserve(new_shape.size());
  for(uint64_t sz: new_shape) {
    if(sz == 1) {
      ret_pds.push_back(partdim_t::singleton(1));
    } else {
      ret_pds.push_back(*iter);
      iter++;
    }
  }
  if(iter != core_pds.end()) {
    throw std::runtime_error("invalid squeezer in convert_squeezer_partition");
  }
  return partition_t(ret_pds);
}

// Note that to_complex and to_real graph_t operations should
// become no ops at the taskgraph level when the partitions
// line up.
//
// Example:
//   graph_constructor_t g;
//   inn = insert input into g of shape {8}, f32, with partition {4,4}
//   out = have g convert inn to complex with partition {2,2}
// Should become
//   taskgraph tg
//   id0 = insert input of shape {4}, f32
//   id1 = insert input of shape {4}, f32
// where id0 refers to the first  partition of inn and out,
//       id1 refers to the second partition of inn and out
//
// As the interface between graph and taskgraph, taskgraph_make_state_t
// has to be careful with how complex dtypes. First of all, all the
// given placements are typed with respect to a dtype. In the example,
// out would have a {2,2} partition. However, from the perspective of
// the taskgraph, N complex is just 2*N reals = 2*N*sizeof(that real)
// bytes.
//
// taskgraph_make_state_t has this notion of wrt_graph and wrt_real.
// With respect to graph: the associated partitioning and sizing
//   is corresponding to the dtype in the graph object (out = {2,2})
// With respect to real: all complex dtypes have the last dimension
//   flattened. (out partitition = {4,4})
//
// Example:
//   inn =                           shape {8}, dtype f32, partition {5,3},
//   out = inn converted to complex; shape {4}, dtype c64, partition {2,2}
// For the taskgraph:
//   (inn = (id0, id1))
//     where
//      id0 = inn[0:5]
//      id1 = inn[5:8];
//
//   id2[0:4] = id0[0:4]
//   id3[0:1] = id0[4:5]
//   id3[1:4] = id1[0:3]
//
//   (out = (id2,id3))
//
struct taskgraph_make_state_t {
  taskgraph_make_state_t(graph_t const& g, vector<placement_t> const& placements)
    : graph(g), placements(placements), access_cache(g.nodes.size())
  {}

  // the input compute graph
  graph_t const& graph;
  vector<placement_t> const& placements;

  // the output task graph
  taskgraph_t taskgraph;

  // Map from gid to the refined tensor formed from that gid.
  map<int, multiple_tensor_t> refined_tensors;
  // ^ wrt real

  // (tg_region_t is just vector<tuple<int,int>> with a hash
  //  for unordered_map)
  //
  // gid -> region -> list of (loc, tid) pairs
  vector<
    std::unordered_map<
      tg_region_t,
      vector<multiple_tensor_t::locid_t>
    >
  > access_cache;

  int access(
    int gid,
    vector<tuple<uint64_t, uint64_t>> const& hrect,
    int loc);
  // ^ wrt real

  void copy_from_refinement(
    vector<uint64_t> const& offset_inn_rel,
    vector<uint64_t> const& offset_out,
    vector<uint64_t> const& size,
    int inn_gid,
    int out_partial);
  // ^ wrt real

  // Grab gid from it's refined tensor in the form of the
  // given multiple placement
  multiple_tensor_t form_from_refinement(
    int gid,
    multiple_placement_t const& placement,
    bool wrt_graph = true);
  // ^ wrt graph if wrt_graph else wrt_real (including the returned value!)

  // this dispatches to the multiple tensor form_from_refinement
  vtensor_t<int> form_from_refinement(
    int gid,
    placement_t const& placement,
    bool wrt_graph = true);

  vtensor_t<int> form_select(int gid);

  // this will form the inputs as required
  vtensor_t<int> compute_einsummable(int gid);

  vtensor_t<int> compute_fill(int gid);

  vtensor_t<int> compute_input(int gid);

  vtensor_t<int> form_relation(int gid);

  // create the refined_tensor object from the compute result;
  // save into refined_tensors
  void communicate(int gid, vtensor_t<int> compute_result);

  multiple_placement_t construct_refinement_placement_(int gid) const {
    return construct_refinement_placement(
      graph, gid, [this](int other_gid) -> placement_t const&
      {
        return placements[other_gid];
      }
    );
  }
  // ^ wrt real

  // Note: the join_result can include agg'd dimensions, in which case
  //       a castable must be given
  multiple_tensor_t construct_refinement_tensor(
    dtype_t dtype,
    placement_t const& join_placement,
    vtensor_t<int> join_result,
    multiple_placement_t const& refinement,
    optional<castable_t> castable);
  // ^ wrt dtype

  static dtype_t complex_to_real(dtype_t);
};

void double_last_dim_inplace(multiple_placement_t& p);
void double_last_dim_inplace(multiple_tensor_t&    p);

#define _DOUBLE_LAST_DIM_(type) \
type double_last_dim(type const& p) { \
  type ret = p; \
  double_last_dim_inplace(ret); \
  return ret; \
}

_DOUBLE_LAST_DIM_(partition_t)
_DOUBLE_LAST_DIM_(placement_t)
_DOUBLE_LAST_DIM_(multiple_placement_t)
_DOUBLE_LAST_DIM_(multiple_tensor_t)

void halve_last_dim_inplace(multiple_placement_t& p);
void halve_last_dim_inplace(multiple_tensor_t&    p);

#define _HALVE_LAST_DIM_(type) \
type halve_last_dim(type const& p) { \
  type ret = p; \
  halve_last_dim_inplace(ret); \
  return ret; \
}

_HALVE_LAST_DIM_(partition_t)
_HALVE_LAST_DIM_(placement_t)
_HALVE_LAST_DIM_(multiple_placement_t)
_HALVE_LAST_DIM_(multiple_tensor_t)

tuple<
  map<int, vtensor_t<int> >, // for each input, the tids of the blocks
  map<int, vtensor_t<int> >, // for each save id, the tids of the blocks
  taskgraph_t>              // the actual taskgraph
taskgraph_t::make(
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  taskgraph_make_state_t state(graph, placements);

  // maps from gid to tensor
  map<int, vtensor_t<int>> inns;
  map<int, vtensor_t<int>> saves;

  for(int gid: graph.get_order()) {
    graph_t::node_t const& node = graph.nodes[gid];

    vtensor_t<int> relation = state.form_relation(gid);

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

  // Note: It used to be the cast that if there were any non-saved outputs,
  //       an error would be thrown. However, there is a valid use case with
  //       the subset operator that not all outs end up as saved.
  //       Example: X = norm(X); Y = X[:,-1,:]

  // Note: Passthrough partials could serve a purpose. To go from columns to
  //       rows partitioning, it may be better to go columns -> singleton -> rows.
  //       If the partials are removed, it will go column part -> row part, which
  //       means there will be more inputs in each non-pass through partial.
  //
  //       In any case, this optimization can be turned off easily enough. I
  //       predict it's better to have than to not. The only time dummy
  //       passthrough partials are really formed is from concat ops (I think).
  optional<tuple<map<int,int>, taskgraph_t>> maybe_simplified =
    state.taskgraph.remove_passthrough_partials();

  if(maybe_simplified) {
    auto const& to_new_tg = std::get<0>(maybe_simplified.value());
    auto const& new_tg = std::get<1>(maybe_simplified.value());

    auto correct = [&](vtensor_t<int>& tids) {
      for(auto& tid: tids.get()) {
        tid = to_new_tg.at(tid);
      }
    };

    for(auto& [_, tids]: inns) {
      correct(tids);
    }
    for(auto& [_, tids]: saves) {
      correct(tids);
      for(auto const& tid: tids.get()) {
        if(!new_tg.nodes[tid].is_save) {
          throw std::runtime_error("unsaved node should be saved");
        }
      }
    }

    return {std::move(inns), std::move(saves), std::move(new_tg)};
  } else {
    return {std::move(inns), std::move(saves), std::move(state.taskgraph)};
  }
}

int taskgraph_make_state_t::access(
  int gid,
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  int loc)
{
  auto const& refined_tensor = refined_tensors.at(gid);
  auto const& refined_part = refined_tensor.partition;

  tg_region_t key = {
    refined_part.get_exact_region(hrect)
  };
  auto const& region = key.region;

  if(product(key.shape()) == 1) {
    // This is just a tid in refined_tensor
    return refined_tensor.at(vector_mapfst(region), loc);
  }

  // See if the cache already has this (gid, key, loc) pair
  auto& locids = access_cache[gid][key];

  for(auto const& [a_loc,a_id]: locids) {
    if(a_loc == loc) {
      return a_id;
    }
  }

  // Otherwise, create it from refined_tensor and insert into the cache

  dtype_t dtype = graph.out_dtype(gid);
  if(dtype_is_complex(dtype)) {
    dtype = this->complex_to_real(dtype);
  }

  int builder_id = taskgraph.new_partial(loc, dtype, hrect_shape(hrect));

  auto inn_idx = vector_mapfst(region);
  do {
    int const& inn_id = refined_tensor.at(inn_idx, loc);
    auto inn_hrect = refined_part.get_hrect(inn_idx);

    // form the touch
    touch_t touch {
      .selection = make_touch_selection_from_full_small(hrect, inn_hrect),
      .castable = std::nullopt,
      .dtype = dtype
    };

    taskgraph.add_to_partial(builder_id, inn_id, touch);
  } while(increment_idxs_region(region, inn_idx));

  locids.push_back(multiple_tensor_t::locid_t {
    .loc = loc,
    .id = builder_id
  });

  return builder_id;
}

void taskgraph_make_state_t::copy_from_refinement(
  vector<uint64_t> const& offset_inn,
  vector<uint64_t> const& offset_out,
  vector<uint64_t> const& size,
  int inn_gid,
  int out_partial)
{
  int rank = size.size();
  auto refined_tensor = refined_tensors.at(inn_gid);
  auto const& partition = refined_tensor.partition;

  int const& loc = taskgraph.out_loc(out_partial);
  auto const& partialize = taskgraph.nodes[out_partial].op.get_partialize();
  auto const& out_shape = partialize.write_shape;
  auto const& dtype = partialize.dtype;

  if(dtype_is_complex(dtype)) {
    throw std::runtime_error("copy_from_refinement wrong type");
  }

  dtype_t dtype_ = graph.out_dtype(inn_gid);
  if(dtype_is_complex(dtype_)) {
    dtype_ = this->complex_to_real(dtype_);
  }

  if(dtype != dtype_) {
    throw std::runtime_error("copy_from_refinement");
  }

  vector<tuple<uint64_t, uint64_t>> hrect_inn_big;
  hrect_inn_big.reserve(offset_inn.size());
  for(int i = 0; i != rank; ++i) {
    auto const& b = offset_inn[i];
    uint64_t e = b + size[i];
    hrect_inn_big.emplace_back(b,e);
  }

  auto region_inn = partition.get_exact_region(hrect_inn_big);
  vector<int> offset_inn_idx = vector_mapfst(region_inn);

  vector<partdim_t> base_pds;
  base_pds.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto sizes = partition.partdims[i].sizes();
    auto const& [b,e] = region_inn[i];
    base_pds.push_back(partdim_t::from_sizes(vector<uint64_t>(
      sizes.begin() + b,
      sizes.begin() + e)));
  }
  partition_t base_part(base_pds);

  auto base_block_shape = base_part.block_shape();
  vector<int> idx(rank, 0);
  do {
    auto base_hrect = base_part.get_hrect(idx);
    auto offset_base = vector_mapfst(base_hrect);
    auto inn_shape = hrect_shape(base_hrect);

    vector<uint64_t> actual_offset_out = vector_add(
      offset_out,
      offset_base);

    touch_t touch {
      .selection = {},
      .castable = std::nullopt,
      .dtype = dtype
    };
    auto& selection = touch.selection;
    selection.reserve(size.size());
    for(int i = 0; i != size.size(); ++i) {
      selection.push_back(touchdim_t {
        .d_inn = inn_shape[i],
        .d_out = out_shape[i],
        .offset_inn = 0,
        .offset_out = actual_offset_out[i],
        .size = inn_shape[i]
      });
    }

    vector<int> inn_idx = vector_add(idx, offset_inn_idx);
    int const& inn = refined_tensor.at(inn_idx, loc);
    taskgraph.add_to_partial(out_partial, inn, touch);

  } while(increment_idxs(base_block_shape, idx));
}

multiple_tensor_t
taskgraph_make_state_t::form_from_refinement(
  int gid,
  multiple_placement_t const& placement,
  bool wrt_graph)
{
  dtype_t dtype = graph.out_dtype(gid);

  if(dtype_is_complex(dtype)) {
    if(wrt_graph) {
      auto ret = form_from_refinement(
        gid,
        double_last_dim(placement),
        false);
      return halve_last_dim(ret);
    } else {
      dtype = this->complex_to_real(dtype);
    }
  }

  multiple_tensor_t const& refine_tensor = refined_tensors.at(gid);

  // verify that the refinement partition
  // is actually a refinement of the given placement
  if(!refine_tensor.partition.refines(placement.partition)) {
    throw std::runtime_error("refine partition not actually a refinement");
  }

  if(placement.partition == refine_tensor.partition) {
    return refine_tensor;
  }

  multiple_tensor_t ret(placement);

  // Just call access a lot of times:
  //   For each block in placement partition,
  //     For each loc,
  //       Get the id and set it in ret
  vector<int> block_shape = placement.partition.block_shape();
  vector<int> idx(block_shape.size(), 0);
  do {
    set<int> const& locs = placement.locations.at(idx);
    auto hrect = placement.partition.get_hrect(idx);
    for(auto const& loc: locs) {
      ret.at(idx, loc) = access(gid, hrect, loc);
    }
  } while(increment_idxs(block_shape, idx));

  return ret;
}

vtensor_t<int>
taskgraph_make_state_t::form_from_refinement(
  int gid,
  placement_t const& pl,
  bool wrt_graph)
{
  multiple_placement_t mpl =
    multiple_placement_t::from_single_placement(pl);
  return form_from_refinement(gid, mpl, wrt_graph).to_tensor(pl);
}

// Example
//   big = coarse { 10, 10, 10 }
//   sml = fine   { 5, 2, 3, 10, 3, 3, 3, 1}
//   rs  { 3, 4, 8 }
vector<tuple<int,int>> _get_sum_breaks(
  vector<uint64_t> coarse,
  vector<uint64_t> fine)
{
  auto const& big = coarse;
  auto const& sml = fine;

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

vtensor_t<int>
taskgraph_make_state_t::form_select(int gid) {
  graph_t::node_t const& node = graph.nodes[gid];

  placement_t const& pl_wrt_join_dtype = placements[gid];

  if(!node.op.is_select()) {
    throw std::runtime_error("form_select must have select node");
  }

  auto select = node.op.get_select();
  bool join_is_complex = dtype_is_complex(select.dtype);
  if(join_is_complex) {
    select.dtype = complex_to_real(select.dtype);

    select.out_shape.back() *= 2;
    for(auto& inn_region: select.inn_regions) {
      auto& selectdim = inn_region.back();
      selectdim.d_inn      *= 2;
      selectdim.offset_inn *= 2;
      selectdim.offset_out *= 2;
      selectdim.size       *= 2;
    }
  }

  auto shape = pl_wrt_join_dtype.block_shape();
  vtensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);

  int rank = shape.size();

  // For every block, loc in pl,
  //   if this block has a single graph input?
  //     call access
  //   otherwise
  //     create a builder,
  //     for each of the inputs,
  //       call add_from_refinement

  // Some naming:
  //   rel means relational with respect to gid
  //   out means with respect to the hrect
  //   inn means with respect to which_inn

  using hrect_t = vector<tuple<uint64_t, uint64_t>>;
  do {
    int const& loc = pl_wrt_join_dtype.locations.at(index);
    hrect_t rel_out_hrect = pl_wrt_join_dtype.partition.get_hrect(index);
    if(join_is_complex) {
      auto& [b,e] = rel_out_hrect.back();
      b *= 2;
      e *= 2;
    }

    vector<tuple<hrect_t, int>> rel_inn_hrects = select.collect(rel_out_hrect);

    if(rel_inn_hrects.size() == 1) {
      auto const& [rel_inn_hrect, which_inn] = rel_inn_hrects[0];
      int const& inn_gid = node.inns[which_inn];
      ret.at(index) = access(inn_gid, rel_inn_hrect, loc);
    } else {
      int builder_id = taskgraph.new_partial(loc, select.dtype, hrect_shape(rel_out_hrect));

      for(auto const& [rel_inn_hrect, which_inn]: rel_inn_hrects) {
        vector<uint64_t> rel_inn_offset = vector_mapfst(rel_inn_hrect);
        vector<uint64_t> rel_out_offset = select.wrt_output_point(rel_inn_offset, which_inn);

        vector<uint64_t> out_offset;
        out_offset.reserve(rank);
        for(int i = 0; i != rank; ++i) {
          auto const& [rel_out_begin, _] = rel_out_hrect[i];
          out_offset.push_back(rel_out_offset[i] - rel_out_begin);
        }

        int const& inn_gid = node.inns[which_inn];

        // A: with respect to the partialize output shape of builder_id
        // B: with respect to the inn tensor shape (relation)
        copy_from_refinement(
          vector_mapfst(rel_inn_hrect), // B
          out_offset,                   // A
          hrect_shape(rel_inn_hrect),   // A
          inn_gid,                      // B
          builder_id                    // A
        );
      }

      ret.at(index) = builder_id;
    }

  } while(increment_idxs(shape, index));

  return ret;
}

vtensor_t<int>
taskgraph_make_state_t::compute_input(int gid) {
  graph_t::node_t const& node = graph.nodes[gid];
  placement_t const& pl = placements[gid];

  if(!node.op.is_input()) {
    throw std::runtime_error("compute_input must have input node");
  }

  auto shape = pl.block_shape();
  vtensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);
  do {
    int const& loc = pl.locations.at(index);
    auto subtensor_shape = pl.partition.tensor_shape_at(index);
    ret.at(index) = taskgraph.insert_input(loc, node.op.out_dtype(), subtensor_shape);
  } while(increment_idxs(shape, index));

  return ret;
}

vtensor_t<int>
taskgraph_make_state_t::compute_einsummable(int gid)
{
  graph_t::node_t const& node = graph.nodes[gid];
  placement_t const& pl = placements[gid];

  if(!node.op.is_einsummable()) {
    throw std::runtime_error("compute_einsummable must have einsummable node");
  }

  einsummable_t const& base_einsummable = node.op.get_einsummable();

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
  vtensor_t<int> ret(shape);
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

vtensor_t<int>
taskgraph_make_state_t::compute_fill(int gid)
{
  graph_t::node_t const& node = graph.nodes[gid];
  placement_t const& pl = placements[gid];

  if(!node.op.is_fill()) {
    throw std::runtime_error("compute_fill must have fill node");
  }

  auto const& full_fill = node.op.get_fill();

  auto shape = pl.block_shape();
  vtensor_t<int> ret(shape);
  vector<int> index(shape.size(), 0);
  do {
    int const& loc = pl.locations.at(index);
    auto subtensor_shape = pl.partition.tensor_shape_at(index);
    ret.at(index) = taskgraph.insert_constant(
      loc,
      fill_t {
        .value = full_fill.value,
        .shape = subtensor_shape
      }
    );
  } while(increment_idxs(shape, index));

  return ret;
}

vtensor_t<int>
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
  if(node.op.is_complexer()) {
    int inn_gid = node.inns[0];
    auto const& complexer = node.op.get_complexer();
    if(complexer.is_to_real()) {
      // complex -> real
      return form_from_refinement(inn_gid, placements.at(gid), false);
    } else {
      // real -> complex
      return form_from_refinement(inn_gid, double_last_dim(placements.at(gid)));
    }
  }
  if(node.op.is_squeezer()) {
    int inn_gid = node.inns[0];
    auto const& squeezer = node.op.get_squeezer();
    auto const& pl = placements.at(gid);
    auto const& part = pl.partition;

    if(dtype_is_complex(squeezer.dtype)) {
      throw std::runtime_error("form_relation: cannot have complex squeezer");
    }

    partition_t inn_part = convert_squeezer_partition(squeezer.inn_shape, part);
    placement_t inn_pl(
      inn_part,
      vtensor_t<int>(inn_part.block_shape(), pl.locations.get()));

    vtensor_t<int> ret = form_from_refinement(inn_gid, inn_pl);
    ret.reshape(part.block_shape());

    return ret;
  }
  if(node.op.is_select()) {
    return form_select(gid);
  }
  if(node.op.is_einsummable()) {
    return compute_einsummable(gid);
  }
  if(node.op.is_fill()) {
    return compute_fill(gid);
  }

  throw std::runtime_error("state form relation should not reach");
}

void
taskgraph_make_state_t::communicate(int join_gid, vtensor_t<int> join_result)
{
  multiple_placement_t usage_placement = construct_refinement_placement_(join_gid);

  optional<castable_t> maybe_castable;
  auto const& node = graph.nodes[join_gid];
  if(node.op.has_aggregation()) {
    maybe_castable = node.op.get_castable();
  }

  dtype_t dtype = node.op.out_dtype();

  if(dtype_is_complex(dtype)) {
    if(maybe_castable && maybe_castable.value() != castable_t::add) {
      // In this case, an aggregation is happening across a complex datatype
      // and it isn't just a simple addition, so usage placements will
      // have to be halved along the last dimension.
      //
      // If usage_placement can't be halved, than the input placements
      // are invalid.
      //
      // All that being said: I don't know what the use case is
      // for aggregating complex values with castable_t::mul...
      auto szs = usage_placement.partition.partdims.back().sizes();
      for(auto const& sz: szs) {
        if(sz % 2 == 1) {
          throw std::runtime_error(
            "cannot halve usage placment; try another partitioning");
        }
      }

      // First halve usage placement
      multiple_tensor_t ret = construct_refinement_tensor(
        dtype,
        placements[join_gid],
        join_result,
        halve_last_dim(usage_placement),
        maybe_castable);

      // then double the result
      refined_tensors.insert({join_gid, double_last_dim(ret)});

      return;
    } else {
      // here, we call construct_refinement_partition, which
      // may be doing an addition aggregation, but only as if we
      // were doing it with reals
      //
      // This is fine beacuse
      //   a + b = (a.real + b.real) + (a.imag + b.imag)*i

      placement_t join_placement = placements[join_gid];

      // modify the last output partition dimension to be doubled
      int out_rank = node.op.out_shape().size();
      int last_dim = out_rank - 1;
      auto& last_partdim = join_placement.partition.partdims[last_dim];
      last_partdim = partdim_t::from_sizes(vector_double(last_partdim.sizes()));

      multiple_tensor_t ret = construct_refinement_tensor(
        complex_to_real(dtype),
        join_placement,
        join_result,
        usage_placement,
        maybe_castable);

      refined_tensors.insert({join_gid, ret});

      return;
    }
  } else {
    multiple_tensor_t ret = construct_refinement_tensor(
      dtype,
      placements[join_gid],
      join_result,
      usage_placement,
      maybe_castable);

    refined_tensors.insert({join_gid, ret});

    return;
  }
  throw std::runtime_error("should not reach");
}

multiple_placement_t
construct_refinement_placement(
  graph_t const& graph,
  int join_gid,
  std::function<placement_t const&(int)> get_placement)
{
  auto const& join_node = graph.nodes[join_gid];
  auto const& join_dtype = join_node.op.out_dtype();

  bool join_is_complex = dtype_is_complex(join_dtype);

  vector<multiple_placement_t> usage_placements;
  usage_placements.reserve(2*join_node.outs.size());
  auto insert_usage = [&](multiple_placement_t p) {
    if(join_is_complex) {
      double_last_dim_inplace(p);
    }
    usage_placements.push_back(p);
  };

  for(auto const& out_gid: join_node.outs) {
    auto const& out_node = graph.nodes[out_gid];
    auto const& out_pl   = get_placement(out_gid);
    if(out_node.op.is_formation()) {
      insert_usage(
        multiple_placement_t::from_single_placement(out_pl));
    } else if(out_node.op.is_complexer()) {
      if(join_is_complex) {
        // complex -> real
        usage_placements.push_back(
          multiple_placement_t::from_single_placement(out_pl));
      } else {
        // real -> complex
        usage_placements.push_back(
          multiple_placement_t::from_single_placement(double_last_dim(out_pl)));
      }
    } else if(out_node.op.is_squeezer()) {
      auto const& squeezer = out_node.op.get_squeezer();
      auto const& inn_shape = squeezer.inn_shape;
      partition_t fix_partition =
        convert_squeezer_partition(inn_shape, out_pl.partition);
      insert_usage(multiple_placement_t::from_single_placement(
        placement_t(
          fix_partition,
          vtensor_t<int>(
            fix_partition.block_shape(),
            out_pl.locations.get()))));
    } else if(out_node.op.is_einsummable()) {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage placements to collect
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          insert_usage(
            multiple_placement_t::make_einsummable_input(
              out_pl,
              out_node.op.get_einsummable(),
              which_input));
        }
      }
    } else if(out_node.op.is_select()) {
      // Note that a select node can use an input multiple times
      // and therefore there may be multiple usage placements to collect
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_gid) {
          insert_usage(
            multiple_placement_t::make_select_input(
              out_pl,
              out_node.op.get_select(),
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
  dtype_t dtype,
  placement_t const& join_placement,
  vtensor_t<int> join_result,
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
  // moved data can't be consumed. (TODO: remove this comment when
  // consumed partials are removed)
  //
  // Moreover, this all occurs one output index at a time

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
            taskgraph.insert_consumed_aggregate(loc, dtype, castable, ids);
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

      if(required_locs.size() == 0) {
        // TODO: there could be dangling partials (unlikely) when there
        //       are not output locs
        continue;
      }

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
          partial_loc, _out_to_selection_rs, partial_id, dtype);
      };

      // A constructor to modify ids of refined_tensor
      auto maybe_init_ref_id_as_partial = [&](int& ref_id, int loc) {
        if(ref_id == -1) {
          ref_id = taskgraph.new_partial(loc, dtype, refinement_shape);
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
        .castable = maybe_castable,
        .dtype = dtype
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
        .castable = maybe_castable,
        .dtype = dtype
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
        // objects may need to be created, but the same cases hold

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

dtype_t taskgraph_make_state_t::complex_to_real(dtype_t dtype) {
  if(dtype == dtype_t::c64) {
    return dtype_t::f32;
  }
  throw std::runtime_error("should not reach");
}

void double_last_dim_inplace(partition_t& p) {
  partdim_t& partdim = p.partdims.back();
  partdim = partdim_t::from_sizes(vector_double(partdim.sizes()));
}
void double_last_dim_inplace(placement_t& p) {
  double_last_dim_inplace(p.partition);
}
void double_last_dim_inplace(multiple_placement_t& p) {
  double_last_dim_inplace(p.partition);
}
void double_last_dim_inplace(multiple_tensor_t& p) {
  double_last_dim_inplace(p.partition);
}

void halve_last_dim_inplace(partition_t& p) {
  partdim_t& partdim = p.partdims.back();
  auto szs = partdim.sizes();
  for(auto const& sz: szs) {
    if(sz % 2 != 0) {
      throw std::runtime_error("can't halve dim");
    }
  }
  partdim = partdim_t::from_sizes(vector_halve(szs));
}
void halve_last_dim_inplace(placement_t& p) {
  halve_last_dim_inplace(p.partition);
}
void halve_last_dim_inplace(multiple_placement_t& p) {
  halve_last_dim_inplace(p.partition);
}
void halve_last_dim_inplace(multiple_tensor_t& p) {
  halve_last_dim_inplace(p.partition);
}

multiple_placement_t::multiple_placement_t(
  partition_t const& pa,
  vtensor_t<set<int>> const& ls)
  : partition(pa), locations(ls)
{
  if(!vector_equal(partition.block_shape(), locations.get_shape())) {
    throw std::runtime_error("multiple_placement_t block shapes do not agree");
  }
}

multiple_placement_t multiple_placement_t::from_single_placement(placement_t const& p)
{
  vector<set<int>> locs;
  locs.reserve(p.locations.get().size());
  for(int const& loc: p.locations.get()) {
    locs.push_back({loc});
  }

  return multiple_placement_t(
    p.partition,
    vtensor_t<set<int>>(p.locations.get_shape(), locs));
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
  vtensor_t<set<int>> locations(partition.block_shape());
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

  return multiple_placement_t(
    std::move(partition),
    std::move(locations));
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
  vtensor_t<set<int>> locations(partition.block_shape());
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

  return multiple_placement_t(
    std::move(partition),
    std::move(locations));
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
  vtensor_t<set<int>> locations(inn_shape);

  do {
    auto inn_index = einsummable.get_input_from_join(join_index, which_input);
    locations.at(inn_index).insert(join_placement.locations.at(join_index));
  } while(increment_idxs(join_shape, join_index));

  return multiple_placement_t(
    std::move(partition),
    std::move(locations));
}

multiple_placement_t multiple_placement_t::make_select_input(
  placement_t const& join_pl,
  select_t const& select,
  int which_input)
{
  placement_t subset_pl = join_pl.subset(
    select.wrt_output_inn_hrect(which_input));
  partition_t const& subset_part = subset_pl.partition;

  auto const& shape = select.inn_shape(which_input);
  int rank = shape.size();

  hrect_t hrect = select.wrt_input_inn_hrect(which_input);

  vector<partdim_t> pds;
  vector<tuple<int,int>> region;
  pds.reserve(rank);
  region.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto const& [beg,end] = hrect[i];
    auto const& sub_pd = subset_part.partdims[i];
    auto const& dim = shape[i];
    vector<uint64_t> sizes;
    auto& [obeg, oend] = region.emplace_back();

    if(beg != 0) {
      sizes.push_back(beg);
      obeg = 1;
    } else {
      obeg = 0;
    }

    auto sub_pd_sizes = sub_pd.sizes();
    vector_concatenate_into(sizes, sub_pd_sizes);

    oend = obeg + sub_pd_sizes.size();

    if(end != dim) {
      sizes.push_back(dim - end);
    }

    pds.push_back(partdim_t::from_sizes(sizes));
  }

  partition_t partition(pds);
  vtensor_t<set<int>> locations(partition.block_shape());

  vector<int> offsets = vector_mapfst(region);
  vector<int> inn_index = offsets;
  do {
    vector<int> subset_index = vector_sub(inn_index, offsets);
    int const& loc = subset_pl.locations.at(subset_index);
    locations.at(inn_index).insert(loc);
  } while(increment_idxs_region(region, inn_index));

  return multiple_placement_t(
    std::move(partition),
    std::move(locations));
}

multiple_tensor_t::multiple_tensor_t(
  partition_t p,
  vtensor_t<vector<locid_t>> && t)
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

vtensor_t<int> multiple_tensor_t::to_tensor(placement_t const& placement) {
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

  return vtensor_t<int>(tensor.get_shape(), ret);
}

std::ostream& operator<<(std::ostream& out, multiple_tensor_t::locid_t const& x)
{
  auto const& [loc,id] = x;
  out << "loc" << loc << "id" << id;
  return out;
}

inline std::size_t tg_region_t::hash() const {
  auto const& xys = region;

  if(xys.size() == 0) {
    return 0;
  }

  std::hash<int> h;
  auto const& [x0,y0] = xys[0];
  auto ret = h(x0);
  hash_combine_impl(ret, h(y0));
  for(int i = 1; i != xys.size(); ++i) {
    auto const& [x,y] = xys[i];
    hash_combine_impl(ret, h(x));
    hash_combine_impl(ret, h(y));
  }
  return ret;
}

inline vector<int> tg_region_t::shape() const {
  vector<int> ret;
  ret.reserve(region.size());
  for(auto const& [x,y]: region) {
    ret.push_back(y-x);
  }
  return ret;
}

bool operator==(tg_region_t const& lhs, tg_region_t const& rhs) {
  return vector_equal(lhs.region, rhs.region);
}
bool operator!=(tg_region_t const& lhs, tg_region_t const& rhs) {
  return !(lhs == rhs);
}

int taskgraph_t::insert_input(
  int loc,
  dtype_t dtype,
  vector<uint64_t> shape,
  bool is_save)
{
  input_t input {
    .loc = loc,
    .size = product(shape) * dtype_size(dtype),
  };

  return insert(input, is_save);
}

int taskgraph_t::insert_constant(
  int loc,
  fill_t const& fill,
  bool is_save)
{
  constant_t constant {
    .loc = loc,
    .fill = fill,
  };

  return insert(constant, is_save);
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
  auto inn_dtypes = e.inn_dtypes();
  auto inn_shapes = e.inn_shapes();
  for(int i = 0; i != inns.size(); ++i) {
    int const& inn = inns[i];
    auto const& inn_dtype = inn_dtypes[i];
    auto const& inn_shape = inn_shapes[i];
    uint64_t actual_size = product(inn_shape) * dtype_size(inn_dtype);
    if(nodes[inn].op.out_size() != actual_size) {
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
    .size = nodes[inn].op.out_size()
  };

  return insert(move, is_save);
}

int taskgraph_t::insert_consumed_aggregate(
  int loc,
  dtype_t dtype,
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
  uint64_t _dtype_sz = dtype_size(dtype);
  if(sz % _dtype_sz != 0) {
    throw std::runtime_error("incorrect size!");
  }
  uint64_t nelem = sz / _dtype_sz;

  vector<input_op_t> inputs;
  inputs.reserve(inns.size());
  for(auto const& inn: inns) {
    if(sz != get_size_at(inn)) {
      throw std::runtime_error("not all the same size: insert consumed agg");
    }
    inputs.push_back(input_op_t {
      .id = inn,
      .consumable = true,
      .region = { inn_regiondim_t { .dim = nelem, .offset = 0 } }
    });
  }

  auto unit = partial_unit_t {
    .castable = castable,
    .out_region = { out_regiondim_t { .offset = 0, .size = nelem } },
    .inputs = inputs
  };

  return insert(partialize_t {
      .loc = loc,
      .dtype = dtype,
      .write_shape = { nelem },
      .units = {unit}
    },
    is_save);
}

int taskgraph_t::insert_select_subset(
  int loc,
  vector<regiondim_t> selection,
  int inn,
  dtype_t dtype,
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
    .castable = optional<castable_t>(),
    .dtype = dtype
  };

  int ret = new_partial(loc, dtype, write_shape, is_save);
  add_to_partial(ret, inn, touch);
  return ret;
}

int taskgraph_t::new_partial(
  int loc,
  dtype_t dtype,
  vector<uint64_t> write_shape,
  bool is_save)
{
  return insert(partialize_t {
      .loc = loc,
      .dtype = dtype,
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
  auto const& partialize = nodes[id_out].op.get_partialize();
  auto const& write_shape = partialize.write_shape;
  auto const& dtype = partialize.dtype;
  if(nodes[id_inn].op.out_size() != dtype_size(dtype) * product(write_shape))
  {
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
    .castable = optional<castable_t>(castable),
    .dtype = dtype
  };

  add_to_partial(id_out, id_inn, touch, consume);
}

uint64_t taskgraph_t::get_size_at(int id) const
{
  return nodes[id].op.out_size();
}

bool taskgraph_t::is_passthrough_partial(int id) const {
  auto const& node = nodes[id];
  if(node.is_save || !node.op.is_partialize()) {
    return false;
  }
  auto const& p = node.op.get_partialize();
  if(p.does_agg()) {
    return false;
  }
  for(auto const& out: node.outs) {
    if(!nodes[out].op.is_partialize()) {
      return false;
    }
  }

  // Now make sure that each partialize uses this node with
  // it's write shape
  for(auto const& out: node.outs) {
    auto const& p_out = nodes[out].op.get_partialize();
    for(auto const& used_shape: p_out.inn_shapes_of(id)) {
      if(!vector_equal(p.write_shape, used_shape)) {
        return false;
      }
    }
  }

  return true;
}

set<int> taskgraph_t::collect_passthrough_partials() const
{
  set<int> ret;
  for(int id = 0; id != nodes.size(); ++id) {
    if(is_passthrough_partial(id)) {
      ret.insert(id);
    }
  }
  return ret;
}

struct _reach_past_t {
  taskgraph_t const& taskgraph;
  set<int> const& pts;

  vector<int> recurse(int id) {
    auto const& p = taskgraph.nodes[id].op.get_partialize();
    vector<int> ret;
    for(auto const& [inn_id, touch]: p.as_touches_from_flat()) {
      if(pts.count(inn_id) == 0) {
        values.emplace_back(inn_id, touch);
        ret.push_back(values.size()-1);
      } else {
        vector<int> whiches = recurse(inn_id);
        ret.reserve(ret.size() + whiches.size());
        for(int which: whiches) {
          auto& [_, t] = values[which];
          auto maybe_tt = touch_compose(t, touch);
          if(maybe_tt) {
            t = maybe_tt.value();
            ret.push_back(which);
          }
        }
      }
    }
    return ret;
  }

  vector<tuple<int, touch_t>> values;
};

vector<tuple<int, touch_t>> _reach_past(
  taskgraph_t const& taskgraph,
  set<int> const& pts,
  int root_id)
{
  _reach_past_t reacher { taskgraph, pts };
  vector<int> whiches = reacher.recurse(root_id);

  vector<tuple<int, touch_t>> ret;
  ret.reserve(whiches.size());
  for(int const& which: whiches) {
    ret.push_back(reacher.values[which]);
  }

  return ret;
}

optional<
  tuple<
    map<int, int>,
    taskgraph_t > >
taskgraph_t::remove_passthrough_partials() const
{
  set<int> pts = collect_passthrough_partials();
  if(pts.size() == 0) {
    return std::nullopt;
  }

  map<int, int> remap;
  taskgraph_t new_tg;

  auto f_remap = [&remap](int id) {
    return remap.at(id);
  };

  using partialize_t = taskgraph_t::partialize_t;
  using op_t = taskgraph_t::op_t;

  for(int id: get_order()) {
    // skip all the pass through partializes;
    // they will be reached pass
    if(pts.count(id) > 0) {
      continue;
    }
    auto const& node = nodes[id];
    if(node.op.is_partialize()) {
      // TODO: need to do what with dtypes?
      auto const& partialize = node.op.get_partialize();
      auto const& loc = partialize.loc;

      // Find every touch path (example: ->here , pt->pt->pt->here, pt->here)
      // and compose the touches. Use the
      //   1. find every touch path through pass through partials to here
      //   2. compose all those paths
      //   3. create a new partialize that, if necc, makes smaller units
      vector<tuple<int, touch_t>> all_ts = _reach_past(*this, pts, id);
      for(auto& [inn_id, touch]: all_ts) {
        inn_id = f_remap(inn_id);
      }
      partialize_t new_partialize = partialize_t::make_from_touches(loc, all_ts);
      int new_id = new_tg.insert(new_partialize, node.is_save);
      remap.insert({id, new_id});
    } else {
      // In this case, just remap the op
      int new_id = new_tg.insert(node.op.remap(f_remap), node.is_save);
      remap.insert({id, new_id});
    }
  }

  using ret_t = tuple<map<int,int>, taskgraph_t>;
  return optional<ret_t>(ret_t{remap, new_tg});
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

tuple<vector<int>, vector<int>>
taskgraph_t::get_input_core_order() const
{
  auto is_inputable = [](op_t const& op) {
    if(op.is_input()) {
      return true;
    } else if(op.is_partialize()) {
      auto const& p = op.get_partialize();
      return !p.does_agg();
    } else if(op.is_move()) {
      return true;
    } else if(op.is_apply()) {
      return false;
    } else {
      throw std::runtime_error("should not reach: is_inputable");
    }
  };

  vector<int> inn_order;
  vector<int> core_order;
  set<int> inside_inn_order;
  for(int const& id: get_order()) {
    auto const& node = nodes[id];
    if(is_inputable(node.op)) {
      bool success = true;
      for(int const& inn: node.op.inputs()) {
        if(inside_inn_order.count(inn) == 0) {
          success = false;
          break;
        }
      }
      if(success) {
        inn_order.push_back(id);
        inside_inn_order.insert(id);
      } else {
        core_order.push_back(id);
      }
    } else {
      core_order.push_back(id);
    }
  }
  return {inn_order, core_order};
}

set<int> taskgraph_t::get_input_everywhere_ids() const
{
  auto [ids, _] = get_input_core_order();
  return set<int>(ids.begin(), ids.end());
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
    ret = std::max(ret, 1 + node.op.out_loc());
    if(node.op.is_move()) {
      ret = std::max(ret, 1 + node.op.get_move().src);
    }
  }
  return ret;
}

uint64_t taskgraph_t::total_bytes_moved() const {
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
    } else if(node.op.is_constant()) {
      auto const& constant = node.op.get_constant();
      es_proto::TGConstant* c = n->mutable_constant();
      c->set_loc(constant.loc);
      es_proto::Fill* f = c->mutable_fill();
      f->set_value(write_with_ss(constant.fill.value));
      for(auto const& dim: constant.fill.shape) {
        f->add_shape(dim);
      }
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
      auto const& [loc, dtype, write_shape, units] = node.op.get_partialize();

      es_proto::TGPartialize* t = n->mutable_partialize();
      t->set_loc(loc);
      t->set_dtype(write_with_ss(dtype));

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
    } else if(n.has_constant()) {
      auto const& c = n.constant();
      auto const& f = c.fill();

      fill_t fill;
      fill.value = parse_with_ss<scalar_t>(f.value());
      auto ds = f.shape();
      fill.shape = vector<uint64_t>(ds.begin(), ds.end());

      ret.nodes.emplace_back(
        op_t(constant_t { c.loc(), fill }),
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

      dtype_t dtype = parse_with_ss<dtype_t>(p.dtype());

      partialize_t partialize {
        .loc = p.loc(),
        .dtype = dtype,
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

uint64_t taskgraph_t::op_t::out_size() const
{
  if(is_input()) {
    return get_input().size;
  } else if(is_apply()) {
    return get_apply().einsummable.out_size();
  } else if(is_move()) {
    return get_move().size;
  } else if(is_partialize()) {
    auto p = get_partialize();
    return product(p.write_shape) * dtype_size(p.dtype);
  } else {
    throw std::runtime_error("should not reach");
  }
}

set<int> taskgraph_t::op_t::inputs() const
{
  if(is_input()) {
    return {};
  } else if(is_apply()) {
    auto const& inns = get_apply().inns;
    return set<int>(inns.begin(), inns.end());
  } else if(is_move()) {
    return {get_move().inn};
  } else if(is_partialize()) {
    set<int> ret;
    for(auto const& partial_unit: get_partialize().units) {
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

taskgraph_t::op_t
taskgraph_t::op_t::remap(std::function<int(int)> to_new_tid) const
{
  if(is_input()) {
    return *this;
  }
  if(is_apply()) {
    op_t ret = *this;
    auto& inns = ret.get_apply().inns;
    for(int& inn: inns) {
      inn = to_new_tid(inn);
    }
    return ret;
  }
  if(is_move()) {
    op_t ret = *this;
    int& inn = ret.get_move().inn;
    inn = to_new_tid(inn);
    return ret;
  }
  if(is_partialize()) {
    op_t ret = *this;
    for(auto& unit: ret.get_partialize().units) {
      for(auto& input: unit.inputs) {
        input.id = to_new_tid(input.id);
      }
    }
    return ret;
  }
  throw std::runtime_error("should not happen");
}

int taskgraph_t::op_t::out_loc() const {
  if(is_input()) {
    return get_input().loc;
  }
  if(is_apply()) {
    return get_apply().loc;
  }
  if(is_move()) {
    return get_move().dst;
  }
  if(is_partialize()) {
    return get_partialize().loc;
  }
  throw std::runtime_error("should not reach: out_loc");
}

bool taskgraph_t::op_t::is_local_to(int loc) const {
  if(is_input()) {
    return get_input().loc == loc;
  }
  if(is_apply()) {
    return get_apply().loc == loc;
  }
  if(is_move()) {
    return get_move().src == loc || get_move().dst == loc;
  }
  if(is_partialize()) {
    return get_partialize().loc == loc;
  }
  throw std::runtime_error("should not reach: is_local_to");
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

taskgraph_t::partialize_t
taskgraph_t::partialize_t::make_from_touches(
  int loc,
  vector<tuple<int, touch_t>> const& inn_touches)
{
  if(inn_touches.size() == 0) {
    throw std::runtime_error("make_from_touches empty input");
  }

  vector<uint64_t> write_shape = vector_from_each_member(
    std::get<1>(inn_touches[0]).selection, uint64_t, d_out);

  // make sure all the touches have the same write shape
  for(int i = 1; i != inn_touches.size(); ++i) {
    vector<uint64_t> write_shape_ = vector_from_each_member(
      std::get<1>(inn_touches[i]).selection, uint64_t, d_out);
    if(!vector_equal(write_shape, write_shape_)) {
      throw std::runtime_error("these touches should all have the same shape");
    }
  }

  dtype_t dtype = std::get<1>(inn_touches[0]).dtype;
  for(int i = 1; i != inn_touches.size(); ++i) {
    if(dtype != std::get<1>(inn_touches[i]).dtype) {
      throw std::runtime_error("these touches should have the same dtype");
    }
  }

  int rank = write_shape.size();
  partition_t refinement = [&]{
    vector<vector<partdim_t>> pds;
    pds.resize(rank);

    for(auto const& [_, touch]: inn_touches) {
      for(int i = 0; i != rank; ++i) {
        auto const& td = touch.selection[i];
        if(td.offset_out == 0) {
          if(td.size == td.d_out) {
            pds[i].push_back(partdim_t::from_spans({td.d_out}));
          } else {
            pds[i].push_back(partdim_t::from_spans({td.size, td.d_out}));
          }
        } else {
          if(td.offset_out + td.size == td.d_out) {
            pds[i].push_back(partdim_t::from_spans({
              td.offset_out,
              td.d_out}));
          } else {
            pds[i].push_back(partdim_t::from_spans({
              td.offset_out,
              td.offset_out + td.size,
              td.d_out}));
          }
        }
      }
    }

    vector<partdim_t> partdims;
    partdims.reserve(rank);
    for(auto const& pd: pds) {
      partdims.push_back(partdim_t::unions(pd));
    }
    return partition_t(partdims);
  }();

  auto block_shape = refinement.block_shape();

  vtensor_t<partial_unit_t> units(block_shape);

  // fill out each unit's out_region
  vector<int> idx(rank, 0);
  do {
    auto hrect = refinement.get_hrect(idx);
    auto& unit = units.at(idx);
    unit.out_region.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      auto [b,e] = hrect[i];
      unit.out_region.push_back(out_regiondim_t {
        .offset = b,
        .size = e-b
      });
    }
  } while(increment_idxs(block_shape, idx));

  // for each touch, add itself to the partial_unit
  for(auto const& [inn_id, touch]: inn_touches) {
    vector<tuple<uint64_t, uint64_t>> base_hrect;
    base_hrect.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      auto const& td = touch.selection[i];
      base_hrect.emplace_back(td.offset_out, td.offset_out + td.size);
    }
    // it could be the case that this touch spans multiple blocks
    auto region = refinement.get_exact_region(base_hrect);
    vector<int> idx = vector_mapfst(region);
    do {
      auto hrect = refinement.get_hrect(idx);
      auto& unit = units.at(idx);

      if(unit.inputs.size() == 0) {
        unit.castable = touch.castable;
      } else {
        if(unit.castable != touch.castable) {
          throw std::runtime_error("must have same castable across a unit");
        }
      }

      unit.inputs.push_back(input_op_t {
        .id = inn_id,
        .consumable = false,
        .region = {}
      });
      auto& inn_rds = unit.inputs.back().region;
      for(int i = 0; i != rank; ++i) {
        auto const& td = touch.selection[i];
        auto const& [fb,_0] = base_hrect[i];
        auto const& [hb,_1] = hrect[i];
        inn_rds.push_back(inn_regiondim_t {
          .dim = td.d_inn,
          .offset = td.offset_inn + (hb-fb)
        });
      }
    } while(increment_idxs_region(region, idx));
  }

  partialize_t ret{
    .loc = loc,
    .dtype = dtype,
    .write_shape = write_shape,
    .units = std::move(units.get())
  };

  if(!ret.valid()) {
    throw std::runtime_error("invalid return from make_from_touches");
  }

  return ret;
}

void taskgraph_t::partialize_t::make_parallel()
{
  vector<partial_unit_t> ret;
  ret.reserve(units.size());
  for(auto& unit: units) {
    int n = unit.inputs.size();
    if(n == 1) {
      ret.push_back(unit);
      continue;
    }
    uint64_t d0 = unit.out_region[0].size;
    if(d0 <= n) {
      ret.push_back(unit);
      continue;
    }
    partdim_t pd = partdim_t::split(d0, n);
    auto sub_sizes = pd.sizes();

    uint64_t offset = 0;
    for(int i = 0; i != n; ++i) {
      auto const& sub_size = sub_sizes[i];

      vector<out_regiondim_t> sub_out_region = unit.out_region;
      sub_out_region[0] = out_regiondim_t {
        .offset = unit.out_region[0].offset + offset,
        .size = sub_size
      };

      vector<input_op_t> sub_inputs = unit.inputs;
      for(input_op_t& sub_input: sub_inputs) {
        vector<inn_regiondim_t>& sub_inn_region = sub_input.region;
        sub_inn_region[0].offset += offset;
      }

      ret.push_back(partial_unit_t {
        .castable = unit.castable,
        .out_region = sub_out_region,
        .inputs = sub_inputs
      });
    }
  }

  units = ret;
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
    touch_t {
      .selection = ts,
      .castable = unit.inputs.size() == 1 ? std::nullopt : unit.castable,
      .dtype = dtype
    }
  };
}

bool taskgraph_t::partialize_t::valid() const
{
  // Make sure that if there are multiple inputs
  // to a unit, there is a castable to do the aggregation
  for(auto const& unit: units) {
    if(unit.inputs.size() > 1 && !bool(unit.castable)) {
      return false;
    }
  }

  // make sure the units partition the write shape
  vector<vector<tuple<uint64_t, uint64_t>>> hrects;
  hrects.reserve(units.size());
  for(auto const& unit: units) {
    hrects.emplace_back();
    auto& x = hrects.back();
    x.reserve(unit.out_region.size());
    for(auto const& [offset, size]: unit.out_region) {
      x.emplace_back(offset, offset + size);
    }
  }

  return partitions_region(hrects, write_shape);

  // TODO: any other checks?
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

bool taskgraph_t::partialize_t::does_agg() const {
  for(auto const& unit: units) {
    if(unit.inputs.size() == 0) {
      throw std::runtime_error("should never happen: unit inputs size 0");
    }
    if(unit.inputs.size() > 1) {
      return true;
    }
  }
  return false;
}

vector<vector<uint64_t>>
taskgraph_t::partialize_t::inn_shapes_of(int inn_id) const {
  vector<vector<uint64_t>> ret;
  for(auto const& unit: units) {
    for(auto const& input: unit.inputs) {
      if(input.id == inn_id) {
        ret.push_back(vector_from_each_member(input.region, uint64_t, dim));
      }
    }
  }
  return ret;
}

void taskgraph_t::print() const {
  std::cout << "taskgraph[num nodes = " << nodes.size() << "]" << std::endl;
  std::cout << std::endl;

  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];

    std::cout << "node " << id;
    if(node.is_save) {
      std::cout << " (save to loc " << node.op.out_loc() << ")";
    }
    std::cout << std::endl;

    auto inputs = node.op.inputs();
    std::cout << "inputs: " << vector<int>(inputs.begin(), inputs.end()) << std::endl;
    std::cout << "tensor size: " << node.op.out_size() << std::endl;

    if(node.op.is_input()) {
      auto const& [loc, _] = node.op.get_input();
      std::cout << "input | loc[" << loc << "]" << std::endl;
    } else if(node.op.is_apply()) {
      auto const& [loc, _, e] = node.op.get_apply();
      std::cout << "apply | loc[" << loc << "] " << e << std::endl;
    } else if(node.op.is_move()) {
      auto const& [src, dst, _0, _1] = node.op.get_move();
      std::cout << "move | loc[" << src << "] -> loc[" << dst << "]" << std::endl;
    } else if(node.op.is_partialize()) {
      int loc = node.op.out_loc();
      std::cout << "partialize | loc[" << loc << "]" << std::endl;
      for(auto const& ts: node.op.get_partialize().as_touches_from()) {
        for(auto const& [inn,touch]: ts) {
          std::cout << inn << " " << touch << std::endl;
        }
      }
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

  auto pts = collect_passthrough_partials();

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
      int loc = node.op.out_loc();
      if(loc < colors.size()) {
        color = colors[loc];
      }
      label = "partialize" + write_with_ss(id) + "@loc" + write_with_ss(loc);
    }

    if(pts.count(id) > 0) {
      color = "pink";
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
      out << tab << "n" << inn_id << " -> " << "n" << id << endl;
    } else if(op.is_partialize()) {
      for(auto const& inn_id: op.inputs()) {
        out << tab << "n" << inn_id << " -> " << "n" << id << endl;
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
  out << t.castable << " " << t.selection << " " << t.dtype;
  return out;
}

