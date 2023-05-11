#include "twolayergraph.h"
#include "copyregion.h"

int twolayergraph_t::insert_join(uint64_t flops, vector<rid_t> const& deps)
{
  int ret = joins.size();

  joins.push_back(join_t {
    .flops = flops,
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

void twolayergraph_t::add_agg_unit(int rid, uint64_t size, vector<jid_t> deps)
{
  auto& refi = refinements[rid];

  refi.units.push_back(agg_unit_t {
    .size = size,
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
tuple<
  vector<tensor_t<int>>,
  equal_items_t<int>,
  twolayergraph_t>
twolayergraph_t::make(graph_t const& graph)
{
  twolayergraph_t ret;

  equal_items_t<int> equal_items;

  vector<tensor_t<rid_t>> all_jids(graph.nodes.size());
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
    all_jids[graph_id] = tensor_t<int>(join_block_shape);
    tensor_t<jid_t>& join_ids = all_jids[graph_id];

    // initialize join_ids by inserting
    // join ops into the graph for each block
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

          einsummable = einsummable_t(
            op_shape,
            inns,
            rank,
            scalarop_t::make_identity()); // will not be used
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
        join_ids.at(join_index) = ret.insert_join(
          get_flops(),
          get_deps());
      } while(increment_idxs(join_block_shape, join_index));

      // Set equal items whenever this is a formation with the same
      // partition as the input or a unary einsummable without an agg
      // and a possible permutation with the same corr input partition
      if(einsummable.inns.size() == 1 &&
         !einsummable.has_aggregation())
      {
        int const& id_inn = node.inns[0];
        auto const& node_inn = graph.nodes[id_inn];
        auto const& inn_join_ids = all_jids[id_inn];

        auto const& part     = node.placement.partition;
        auto const& part_inn = node_inn.placement.partition;

        auto partdims_with_respect_to_inn =
          einsummable.get_input_from_join(part.partdims, 0);

        if(part_inn.partdims == partdims_with_respect_to_inn) {
          // now we have to reach past the permutation
          join_index = vector<int>(join_block_shape.size(), 0);
          do {
            vector<int> inn_index = einsummable.get_input_from_join(join_index, 0);
            equal_items.insert(
              inn_join_ids.at(inn_index),
              join_ids.at(join_index));
          } while(increment_idxs(join_block_shape, join_index));
        }
      }
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

  return {all_jids, equal_items, ret};
}

uint64_t twolayergraph_t::count_elements_to(
  vector<int> const& locations,
  jid_t jid,
  int dst) const
{
  auto const& join = joins[jid];

  uint64_t ret = 0;
  for(auto const& rid: join.deps) {
    auto const& refinement = refinements[rid];
    for(auto const& agg_unit: refinement.units) {
      // This agg unit needs to be moved to this location.
      // This happens by first locally aggregating at
      // each source location and then moving from that source
      // location to the destination.

      // src_locs keeps track of which source locations
      // have already been sent from. Only send at most
      // once per location. Don't send from dst.
      set<int> src_locs;
      for(auto const& dep_jid: agg_unit.deps) {
        int const& src = locations[dep_jid];
        if(src != dst && src_locs.count(src) == 0) {
          ret += agg_unit.size;
          src_locs.insert(src);
        }
      }
    }
  }
  return ret;
}

void twolayergraph_t::print_graphviz(std::ostream& out)
{
  print_graphviz(out, [](int){ return ""; });
}

void twolayergraph_t::print_graphviz(
  std::ostream& out,
  std::function<string(int)> const& jid_to_color)
{
  using std::endl;

  string tab = "  ";
  out << "digraph {" << endl;

  for(int jid = 0; jid != joins.size(); ++jid) {
    string color = jid_to_color(jid);
    out << tab << "j" << jid;
    if(color != "") {
      out << " [style=filled,color=\"" << color << "\"]";
    }
    out << endl;
  }
  for(int rid = 0; rid != refinements.size(); ++rid) {
    out << tab << "r" << rid << endl;
    auto const& refi = refinements[rid];

    for(int uid = 0; uid != refi.units.size(); ++uid) {
      auto const& unit = refi.units[uid];
      for(auto const& inn_jid: unit.deps) {
        out << tab << "j" << inn_jid << " -> " << "r" << rid
          << " [label=\"" << uid << "\"]" << endl;
      }
    }

    for(auto const& out_jid: refi.outs) {
      out << tab << "r" << rid << " -> " << "j" << out_jid << endl;
    }
  }
  out << "}" << endl;
}

vector<int> graph_locations_to_tasklayer(
  graph_t const& graph,
  vector<tensor_t<int>> const& g_to_tl)
{
  vector<tuple<int,int>> tl_to_g =
    twolayer_join_holder_t<int>::make_tl_to_g(g_to_tl);

  vector<int> items(tl_to_g.size());

  for(int jid = 0; jid != tl_to_g.size(); ++jid) {
    auto const& [gid,bid] = tl_to_g[jid];
    items[jid] = graph.nodes[gid].placement.locations.get()[bid];
  }

  return items;
}

