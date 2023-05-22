#include "forwardsim.h"
#include "copyregion.h"

forward_state_t::forward_state_t(graph_t const& g)
  : graph(g),
    ginfos(g.nodes.size())
{
  vector<int> cs(g.nodes.size());
  std::iota(cs.begin(), cs.end(), 0);
  can_partition.insert(cs.begin(), cs.end());
}

forward_state_t::graph_node_info_t::graph_node_info_t()
  : partition(std::nullopt),
    joins(std::nullopt),
    locs(std::nullopt),
    refinement_partition(std::nullopt),
    refis(std::nullopt)
{}

set<int> const& forward_state_t::can_assign_partition() const {
  return can_partition;
}

void forward_state_t::assign_partition(int gid, partition_t const& new_part) {
  DOUT("ASSIGN PARTITION " << gid);
  if(new_part.total_shape() != graph.nodes[gid].op.shape()) {
    throw std::runtime_error("given partition has incorrect total shape");
  }

  auto& partition = ginfos[gid].partition;
  if(partition) {
    throw std::runtime_error("this has already been given a partition");
  }

  partition = new_part;

  ec_assign_partition(gid);
}

void forward_state_t::ec_assign_partition(int gid) {
  // setup the compute locations to all contain -1 initially
  auto const& node = graph.nodes[gid];
  auto& ginfo = ginfos[gid];
  ginfo.locs = tensor_t<int>(ginfo.partition.value().block_shape(), -1);

  // see if refinements can be setup
  auto inns = node.get_inns_set();
  if(inns.size() == 0) {
    setup_joins(gid);
  } else {
    for(auto const& gid_inn: node.get_inns_set()) {
      if(can_setup_refinement_partition(gid_inn)) {
        setup_refinement_partition(gid_inn);
      }
    }
  }
}

bool forward_state_t::can_setup_refinement_partition(int gid) const {
  if(ginfos[gid].refinement_partition) {
    // nothing to do if it has already been setup
    return false;
  }

  auto const& node = graph.nodes[gid];
  if(node.outs.size() == 0) {
    // if there is only one output, there is no refinement
    // partition as the node does not have a usage
    return false;
  }

  // see if each output has a partition
  for(auto const& gid_out: node.outs) {
    if(!ginfos[gid_out].partition) {
      return false;
    }
  }

  return true;
}

// Note: Almost a copy of union_partition_holders in src/taskgraph.cc
partition_t union_partitions(vector<partition_t> const& ps)
{
  if(ps.size() == 0) {
    throw std::runtime_error("union partitions: input is empty");
    return ps[0];
  }
  if(ps.size() == 1) {
    return ps[0];
  }

  vector<partdim_t> partdims;
  int rank = ps[0].block_shape().size();
  partdims.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    vector<partdim_t> xs;
    xs.reserve(ps.size());
    for(auto const& p: ps) {
      xs.push_back(p.partdims[i]);
    }
    partdims.push_back(partdim_t::unions(xs));
  }
  return partition_t(partdims);
}

void forward_state_t::setup_refinement_partition(int join_id) {
  DOUT("SETUP REFINEMENT PARTITION " << join_id);
  auto const& join_node = graph.nodes[join_id];
  vector<partition_t> usage_partitions;
  usage_partitions.reserve(2*join_node.outs.size());
  for(auto const& out_id: join_node.outs) {
    auto const& out_node = graph.nodes[out_id];
    if(out_node.op.is_formation()) {
      usage_partitions.push_back(out_node.placement.partition);
    } else {
      // Note that an einsummable node can use an input multiple times
      // and therefore there may be multiple usage partitions to collect
      auto const& einsummable = out_node.op.get_einsummable();
      for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
        if(out_node.inns[which_input] == join_id) {
          usage_partitions.emplace_back(einsummable.get_input_from_join(
            out_node.placement.partition.partdims,
            which_input));
        }
      }
    }
  }

  ginfos[join_id].refinement_partition = union_partitions(usage_partitions);

  if(can_setup_refis(join_id)) {
    setup_refis(join_id);
  }

  ec_setup_refinement(join_id);
}

void forward_state_t::ec_setup_refinement(int gid) {
  auto const& node = graph.nodes[gid];
  for(auto const& out_id: node.outs) {
    if(can_setup_joins(out_id)) {
      setup_joins(out_id);
    }
  }
}

bool forward_state_t::can_setup_refis(int gid) const {
  auto const& ginfo = ginfos[gid];
  return bool(ginfo.joins) && bool(ginfo.refinement_partition);
}

void forward_state_t::setup_refis(int graph_id) {
  DOUT("SETUP REFIS " << graph_id);
  auto const& node = graph.nodes[graph_id];
  auto& ginfo = ginfos[graph_id];

  partition_t const& join_partition = ginfo.partition.value();
  partition_t const& refi_partition = ginfo.refinement_partition.value();

  int join_rank = node.op.rank();
  int out_rank  = node.op.out_rank();
  int agg_rank  = join_rank - out_rank;

  auto join_shape = join_partition.block_shape();
  auto const& _join_partdims = join_partition.partdims;

  partition_t out_partition(vector<partdim_t>(
    _join_partdims.begin(),
    _join_partdims.begin() + out_rank));

  std::optional<vector<int>> maybe_agg_shape;
  if(agg_rank > 0) {
    partition_t agg_partition(vector<partdim_t>(
      _join_partdims.begin() + out_rank,
      _join_partdims.end()));

    maybe_agg_shape = agg_partition.block_shape();
  }

  // set up the refinement nodes
  auto refi_shape = refi_partition.block_shape();
  ginfo.refis = tensor_t<refinement_t>(refi_shape);
  tensor_t<refinement_t>& refis = ginfo.refis.value();

  auto& joins = ginfo.joins.value();

  vector<int> out_shape = out_partition.block_shape();
  vector<int> out_index(out_shape.size(), 0);
  do {
    copyregion_t get_regions(refi_partition, out_partition, out_index);
    do {
      vector<int> refi_index = vector_from_each_member(
        get_regions.info, int, idx);
      vector<uint64_t> read_shape = vector_from_each_member(
        get_regions.info, uint64_t, size);

      auto& refi = refis.at(refi_index);
      refi.units.push_back(agg_unit_t {
        .size = product(read_shape),
        .deps = {}
      });

      vector<int>& deps = refi.units.back().deps;
      if(maybe_agg_shape) {
        vector<int> agg_index(agg_rank, 0);
        auto const& agg_shape = maybe_agg_shape.value();
        deps.reserve(product(agg_shape));
        do {
          vector<int> join_index = vector_concatenate(out_index, agg_index);
          int join_bid = idxs_to_index(join_shape, join_index);
          deps.push_back(join_bid);
        } while(increment_idxs(agg_shape, agg_index));
      } else {
        // the join index is the out index if there is no agg
        // and there is only one input
        auto const& join_index = out_index;
        int join_bid = idxs_to_index(join_shape, join_index);
        deps.push_back(join_bid);
      }

      // an agg unit has been added to refi, so let the input
      // joins know they have an output here
      int refi_bid = idxs_to_index(refi_shape, refi_index);
      for(auto const& dep_join_bid: deps) {
        joins.get()[dep_join_bid].outs.insert(refi_bid);
      }
    } while(get_regions.increment());
  } while(increment_idxs(out_shape, out_index));
}

bool forward_state_t::can_setup_joins(int gid) const {
  auto const& ginfo = ginfos[gid];
  if(ginfo.joins) {
    // nothing to do if it has already been setup
    return false;
  }

  if(!ginfo.partition) {
    // if this doesn't have a partition, then joins can't be setup
    return false;
  }

  // see if each input has the refinement partition setup
  auto const& node = graph.nodes[gid];
  for(auto const& gid_inn: node.get_inns_set()) {
    if(!ginfos[gid_inn].refinement_partition) {
      return false;
    }
  }

  return true;
}

void forward_state_t::setup_joins(int graph_id) {
  DOUT("SETUP JOIN " << graph_id);
  auto const& node = graph.nodes[graph_id];
  auto& ginfo = ginfos[graph_id];

  partition_t const& join_partition = ginfo.partition.value();

  auto join_block_shape = join_partition.block_shape();

  ginfo.joins = tensor_t<join_t>(join_block_shape);
  tensor_t<join_t>& join_infos = ginfo.joins.value();

  vector<int> join_index(join_block_shape.size(), 0);

  do {
    int join_bid = idxs_to_index(join_block_shape, join_index);
    join_t& join_info = join_infos.at(join_index);

    // flops
    //   input nodes: 0
    //   formation nodes: 0
    //   join nodes: the tensor block size
    if(node.op.is_einsummable()) {
      join_info.flops = product(join_partition.tensor_shape_at(join_index));
    } else {
      join_info.flops = 0;
    }

    // deps
    //   input nodes: {}
    //   formation nodes: same as a straight einsummable op
    //   einsummable nodes: reach into each input and grab it
    if(node.op.is_input()) {
      join_info.deps = {};
    } else {
      optional<einsummable_t> maybe_einsummable;
      if(node.op.is_formation()) {
        auto op_shape = node.op.shape();
        int rank = op_shape.size();

        vector<vector<int>> inns(1);
        inns[0] = vector<int>(rank);
        std::iota(inns[0].begin(), inns[0].end(), 0);

        maybe_einsummable = einsummable_t(
          op_shape,
          inns,
          rank,
          scalarop_t::make_identity()); // will not be used
      } else {
        maybe_einsummable = node.op.get_einsummable();
      }

      auto hrect = join_partition.get_hrect(join_index);

      // for each input, get the inputs refinement ids that map
      // onto the corresponding input hrect.
      for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
        int const& inn = node.inns[which_inn];

        auto& inn_ginfo = ginfos[inn];
        auto& inn_refis = inn_ginfo.refis.value();

        partition_t const& inn_partition = inn_ginfo.refinement_partition.value();

        auto inn_shape = inn_partition.block_shape();

        auto const& e = maybe_einsummable.value();
        auto inn_hrect = e.get_input_from_join(hrect, which_inn);

        auto inn_region = inn_partition.get_region(inn_hrect);
        vector<int> inn_index = vector_mapfst(inn_region);
        do {
          int inn_refi_bid = idxs_to_index(inn_shape, inn_index);
          join_info.deps.push_back(rid_t { inn, inn_refi_bid });
          inn_refis.get()[inn_refi_bid].outs.insert(jid_t { graph_id, join_bid });
        } while(increment_idxs_region(inn_region, inn_index));
      }
    }
  } while(increment_idxs(join_block_shape, join_index));

  ec_setup_joins(graph_id);
}

void forward_state_t::ec_setup_joins(int gid) {
  if(can_setup_refis(gid)) {
    setup_refis(gid);
  }
}

bool operator==(forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator< (forward_state_t::jid_t const& lhs, forward_state_t::jid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
bool operator==(forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}
bool operator< (forward_state_t::rid_t const& lhs, forward_state_t::rid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}

