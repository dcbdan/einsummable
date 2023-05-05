#include "twolayergraph.h"
#include "copyregion.h"

int twolayergraph_t::insert_join(uint64_t flops, int util, gid_t gid, vector<rid_t> deps)
{
  int ret = joins.size();

  joins.push_back(join_t {
    .flops = flops,
    .util = util,
    .gid = gid,
    .deps = deps,
    .outs = {}
  });

  order.push_back(twolayerid_t { .id = ret, .is_join = true });

  for(auto const& rid: deps) {
    refinements[rid].outs.insert(ret);
  }

  return ret;
}

int twolayergraph_t::insert_empty_refinement()
{
  int ret = refinements.size();

  refinements.push_back(refinement_t {
    .units = {},
    .outs = {}
  });

  order.push_back(twolayerid_t { .id = ret, .is_join = false });

  return ret;
}

void twolayergraph_t::add_agg_unit(int rid, uint64_t bytes, vector<jid_t> deps)
{
  auto& refi = refinements[rid];

  refi.units.push_back(agg_unit_t {
    .bytes = bytes,
    .deps = deps
  });

  for(auto const& jid: deps) {
    joins[jid].outs.insert(rid);
  }
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

// This function is a riff on state_t::communicate
// in taskgraph.cc used in taskgraph_t::make.
twolayergraph_t twolayergraph_t::make(graph_t const& graph) {
  twolayergraph_t ret(graph);

  vector<tensor_t<rid_t>> all_refis(graph.nodes.size());
  map<int, partition_t> all_refinement_partitions;

  // set up all the refinement partitions, which for a given node is
  // the refinement of the usage partitions
  for(int join_id = 0; join_id != graph.nodes.size(); ++join_id) {
    auto const& join_node = graph.nodes[join_id];
    if(join_node.outs.size() == 0) {
      // Note: nodes that have no outs do not have a usage
      continue;
    }
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

    all_refinement_partitions.insert({join_id, union_partitions(usage_partitions)});
  }

  for(auto const& graph_id: graph.get_order()) {
    auto const& node = graph.nodes[graph_id];

    partition_t const& join_partition = node.placement.partition;
    auto const& join_locations = node.placement.locations;

    auto join_block_shape = join_partition.block_shape();

    // initialize join_ids by inserting
    // join ops into the graph for each block
    tensor_t<jid_t> join_ids = tensor_t<jid_t>(join_block_shape);
    {
      vector<int> join_index(join_block_shape.size(), 0);

      // flops
      //   input nodes: 0
      //   formation nodes: 0
      //   join nodes: the tensor block size
      std::function<uint64_t()> get_flops;
      if(node.op.is_einsummable()) {
        get_flops = [&] {
          return product(join_partition.tensor_shape_at(join_index));
        };
      } else {
        get_flops = []{ return 0; };
      }

      // TODO: Implement get_util somehow..
      //
      // The actual worker utilization depends on what the device is,
      // which this function should remain agnostic to.
      //
      // A gpu may have 1000 workers whereas a cpu may have 10.
      // A matmul will have a full utilization whereas an elementwise
      // op may have a utilization of 1 on a cpu if it is single threaded.
      //
      // Perhaps have every op either get a full utilization or
      // singleton utilization.
      //
      // In any case, a utilization of 1 and a worker capacity of 1
      // is equivalent to not having worker capacities anyway, which
      // is what should be implemented in the meantime.
      std::function<int()> get_util = []{ return 1; };

      // deps
      //   input nodes: {}
      //   formation nodes: same as a straight einsummable op
      //   einsummable nodes: reach into each input and grab it
      std::function<vector<int>()> get_deps;
      einsummable_t einsummable;
      if(node.op.is_input()) {
        get_deps = []{ return vector<int>(); };
      } else {
        if(node.op.is_formation()) {
          auto op_shape = node.op.shape();
          int rank = op_shape.size();

          vector<vector<int>> inns(1);
          inns[0] = vector<int>(rank);
          std::iota(inns[0].begin(), inns[0].end(), 0);

          einsummable = einsummable_t {
            .join_shape = op_shape,
            .inns = inns,
            .out_rank = rank,
            .join = scalarop_t::make_mul(), // will not be used
            .castable = castable_t::add,    // will not be used
          };
        } else {
          einsummable = node.op.get_einsummable();
        }

        get_deps = [&]{
          vector<int> ret;
          ret.reserve(4*node.inns.size()); // just a guess on the number of inputs

          auto hrect = join_partition.get_hrect(join_index);

          // for each input, get the inputs refinement ids that map
          // onto the corresponding input hrect.
          for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
            int const& inn = node.inns[which_inn];

            partition_t const& inn_partition = all_refinement_partitions.at(inn);

            tensor_t<rid_t> const& inn_refis = all_refis[inn];

            auto inn_hrect = einsummable.get_input_from_join(hrect, which_inn);

            auto inn_region = inn_partition.get_region(inn_hrect);
            vector<int> inn_index = vector_mapfst(inn_region);
            do {
              ret.push_back(inn_refis.at(inn_index));
            } while(increment_idxs_region(inn_region, inn_index));
          }

          return ret;
        };
      }

      do {
        gid_t gid {
          .id = graph_id,
          .index = idxs_to_index(join_block_shape, join_index)
        };
        join_ids.at(join_index) = ret.insert_join(
          get_flops(),
          get_util(),
          gid,
          get_deps());
      } while(increment_idxs(join_block_shape, join_index));
    }

    if(node.outs.size() == 0) {
      // output graph nodes do not need to be refined
      continue;
    }

    partition_t const& refi_partition = all_refinement_partitions.at(graph_id);

    int join_rank = node.op.rank();
    int out_rank  = node.op.out_rank();
    int agg_rank  = join_rank - out_rank;

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
    all_refis[graph_id] = tensor_t<rid_t>(refi_shape);
    tensor_t<rid_t>& refi_ids = all_refis[graph_id];

    // initialize empty refinements
    {
      vector<int> refi_index(refi_shape.size(), 0);
      do {
        refi_ids.at(refi_index) = ret.insert_empty_refinement();
      } while(increment_idxs(refi_shape, refi_index));
    }

    vector<int> out_shape = out_partition.block_shape();
    vector<int> out_index(out_shape.size(), 0);
    do {
      copyregion_t get_regions(refi_partition, out_partition, out_index);
      do {
        vector<int> refi_index = vector_from_each_member(
          get_regions.info, int, idx);
        vector<uint64_t> read_shape = vector_from_each_member(
          get_regions.info, uint64_t, size);

        vector<jid_t> deps;
        if(maybe_agg_shape) {
          vector<int> agg_index(agg_rank, 0);
          auto const& agg_shape = maybe_agg_shape.value();
          deps.reserve(product(agg_shape));
          do {
            vector<int> join_index = vector_concatenate(out_index, agg_index);
            deps.push_back(join_ids.at(join_index));
          } while(increment_idxs(agg_shape, agg_index));
        } else {
          // the join index is the out index if there is no agg
          // and there is only one input
          auto const& join_index = out_index;
          deps.push_back(join_ids.at(join_index));
        }

        ret.add_agg_unit(
          refi_ids.at(refi_index),
          product(read_shape),
          deps);
      } while(get_regions.increment());
    } while(increment_idxs(out_shape, out_index));
  }

  return ret;
}

uint64_t twolayergraph_t::count_bytes_to(jid_t jid, int loc) const
{
  auto const& join = joins[jid];

  uint64_t ret;
  for(auto const& rid: join.deps) {
    auto const& refinement = refinements[rid];
    for(auto const& agg_unit: refinement.units) {
      for(auto const& dep_jid: agg_unit.deps) {
        if(join_location(dep_jid) != loc) {
          ret += agg_unit.bytes;
        }
      }
    }
  }
  return ret;
}

bool operator<(twolayergraph_t::gid_t const& lhs, twolayergraph_t::gid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
bool operator==(twolayergraph_t::gid_t const& lhs, twolayergraph_t::gid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}

