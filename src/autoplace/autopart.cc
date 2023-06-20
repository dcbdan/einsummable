#include "autopart.h"
#include "../einsummable/taskgraph.h"

partition_t _make_finer(partition_t const& p) {
  auto const& pds = p.partdims;

  uint64_t mx_sz = 0;
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    auto sizes = pd.sizes();
    uint64_t sz = *std::min_element(sizes.begin(), sizes.end());
    if(sz >= 2 && sz > mx_sz) {
      mx_sz = sz;
      split_d = d;
    }
  }

  if(split_d == -1) {
    throw std::runtime_error("could not _make_finer");
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::split_each(pds[split_d], 2);

  return partition_t(new_pds);
}

partition_t _make_coarser(partition_t const& p) {
  auto const& pds = p.partdims;

  uint64_t mn_sz = std::numeric_limits<uint64_t>::max();
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    auto sizes = pd.sizes();
    uint64_t sz = *std::min_element(sizes.begin(), sizes.end());
    if(sizes.size() >= 2 && sz < mn_sz) {
      mn_sz = sz;
      split_d = d;
    }
  }

  if(split_d == -1) {
    throw std::runtime_error("could not _make_coarser");
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::merge_each(pds[split_d], 2);

  return partition_t(new_pds);
}

vector<partition_t> autopartition(
  graph_t const& graph,
  int nmax,
  int nloc,
  equal_items_t<int> equal_constraints)
{
  autopartition_state_t state(
    graph, nmax, nloc, equal_constraints,
    _make_finer, _make_coarser);

  // 0. set mmlike nodes
  // 1. Walk forwards from the nodes that have been
  //    set and set remaining nodes from input partitions
  // 2. Walk backwards setting a node from output partitions,
  //    excluding output formation nodes
  // 3. set output formation nodes

  // Step 0.
  // Seed the subsequent computation by
  // setting all mmlike partitions directly
  int n_nodes = graph.nodes.size();
  for(int id = 0; id != n_nodes; ++id) {
    if(state.is_mmlike(id)) {
      state.set_mmlike(id);
    }
  }

  // Step 1.
  {
    bool found;
    do {
      found = false;
      set<int> rem = state.remaining;
      for(auto const& id: rem) {
        // will not do any setting if inputs are not available.
        found = found || state.set_from_inputs_and_recurse(id);
      }
    } while(found);
  }

  // Step 2.
  {
    set<int> rem = state.remaining;
    for(auto const& id: rem) {
      if(!state.is_output_formation(id)) {
        // will not do any setting if outputs are not available.
        state.set_from_outputs_and_recurse(id);
      }
    }
  }

  // Step 3.
  // Set any remaining output formation nodes from input partitions
  {
    set<int> rem = state.remaining;
    for(auto const& id: rem) {
      if(state.is_output_formation(id)) {
        // there is no recursion to do
        state.set_from_inputs_and_recurse(id);
      } else {
        throw std::runtime_error("this should have already been set");
      }
    }
  }

  // Just in case
  if(state.remaining.size() != 0) {
    throw std::runtime_error("autopartition did not get all partiitons");
  }

  return vector_from_each_method(state.ret, partition_t, value);
}

autopartition_state_t::autopartition_state_t(
  graph_t const& g,
  int nloc,
  int nmax,
  equal_items_t<int> const& eqd,
  std::function<partition_t(partition_t const&)> make_finer,
  std::function<partition_t(partition_t const&)> make_coarser)
  : graph(g), nloc(nloc), nmax(nmax), equals(eqd),
    make_finer(make_finer), make_coarser(make_coarser)
{
  int n_nodes = graph.nodes.size();

  ret = vector<optional<partition_t>>(n_nodes);

  for(int i = 0; i != n_nodes; ++i) {
    remaining.insert(i);
  }
}

void autopartition_state_t::set_partition(int id, partition_t const& p) {
  if(remaining.count(id) == 0) {
    throw std::runtime_error("not in remaining");
  }
  remaining.erase(id);

  if(ret[id]) {
    throw std::runtime_error("already set");
  }
  ret[id] = p;

  if(equals.has(id)) {
    auto const& eqs = equals.get_at(id);
    for(int const& other: eqs) {
      if(other != id) {
        if(remaining.count(other) == 0) {
          throw std::runtime_error("not in remaining ");
        }
        remaining.erase(other);
        ret[other] = p;
      }
    }
  }
}

bool autopartition_state_t::is_mmlike(int id) const {
  auto const& node = graph.nodes[id];
  if(node.op.is_einsummable()) {
    auto const& e = node.op.get_einsummable();
    return e.inns.size() > 1 && e.out_rank < e.join_shape.size();
  } else {
    return false;
  }
}

bool autopartition_state_t::is_output_formation(int id) const {
  auto const& node = graph.nodes[id];
  return node.outs.size() == 0 && node.op.is_formation();
}

void _update_pds_and_choice_from_input_for_nonmmlike(
  vector<vector<partdim_t>>& pds,
  optional<partition_t>& choice,
  vector<int> const& is,
  vector<partdim_t> const& inn_partdims)
{
  if(inn_partdims.size() == pds.size() && !choice) {
    vector<partdim_t> choice_partdims(pds.size());
    for(int inn_idx = 0; inn_idx != is.size(); ++inn_idx) {
      int const& join_idx = is[inn_idx];
      auto const& inn_partdim = inn_partdims[inn_idx];
      choice_partdims[join_idx] = inn_partdim;
    }
    choice = partition_t(choice_partdims);
  }

  for(int inn_idx = 0; inn_idx != is.size(); ++inn_idx) {
    int const& join_idx = is[inn_idx];
    auto const& inn_partdim = inn_partdims[inn_idx];
    pds[join_idx].push_back(inn_partdims[inn_idx]);
  }
}

// Note: this may take into account input partitions, if available.
void autopartition_state_t::set_from_outputs_and_recurse(int id) {
  if(ret[id]) {
    return;
  }

  if(is_mmlike(id)) {
    throw std::runtime_error("should not happen");
  }

  if(is_output_formation(id)) {
    throw std::runtime_error("should not happen");
  }

  auto const& node = graph.nodes[id];
  auto shape = node.op.shape();

  bool has_agg = node.op.has_aggregation();

  vector<vector<partdim_t>> pds(shape.size());
  {
    // set pds after taking union equivalent to singleton
    partition_t singleton = partition_t::singleton(shape);
    for(int i = 0; i != singleton.partdims.size(); ++i) {
      pds[i].push_back(singleton.partdims[i]);
    }
  }

  optional<partition_t> choice;

  auto update_with = [&](vector<partdim_t> const& partdims) {
    if(!has_agg && !choice && partdims.size() == shape.size()) {
      choice = partition_t(partdims);
    }
    for(int i = 0; i != partdims.size(); ++i) {
      auto const& partdim = partdims[i];
      pds[i].push_back(partdim);
    }
  };

  // for each usage, make sure it is set (recurse)
  // and update choice and pds
  for(auto const& out_id: node.outs) {
    if(is_output_formation(out_id)) {
      continue;
    }
    set_from_outputs_and_recurse(out_id);

    auto const& out_part = ret[out_id].value();
    auto const& out_node = graph.nodes[out_id];

    if(out_node.op.is_einsummable()) {
      auto const& e = out_node.op.get_einsummable();
      auto const& out_inns = out_node.inns;
      for(int which_inn = 0; which_inn != out_inns.size(); ++which_inn) {
        if(out_inns[which_inn] == id) {
          vector<partdim_t> out_reordered_partdims =
            e.get_input_from_join(out_part.partdims, which_inn);
          update_with(out_reordered_partdims);
        }
      }
    } else if(out_node.op.is_concat()) {
      auto const& concat = out_node.op.get_concat();
      auto out_partdims = out_part.partdims;
      out_partdims[concat.dim] = partdim_t::singleton(shape[concat.dim]);
      update_with(out_partdims);
    } else if(out_node.op.is_subset()) {
      // just ignoring
    } else {
      // for complexer and formation
      update_with(out_part.partdims);
    }
  }

  // This recursion is very dirty. Calling set_from_outputs_and_recurse(out_id)
  // may have set this node because one of it's equivalences was hit.
  if(ret[id]) {
    // TODO: Is this the desired behavior for such cases? It seems unlikely
    //       that this will be triggered.
    return;
  }

  // now there may be input partitions available,
  // so add them to pds
  // (this will probably only be relevant nodes
  //  with > 1 input where 1 input is not defined)
  for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
    int inn_id = node.inns[which_inn];
    if(!ret[inn_id]) {
      continue;
    }

    auto const& inn_partdims = ret[inn_id].value().partdims;

    if(node.op.is_einsummable()) {
      vector<int> const& is = node.op.get_einsummable().inns[which_inn];
      _update_pds_and_choice_from_input_for_nonmmlike(pds, choice, is, inn_partdims);
    } else if(node.op.is_concat()) {
      // TODO?
    } else if(node.op.is_subset()) {
      // TODO?
    } else {
      update_with(inn_partdims);
    }
  }

  vector<partdim_t> new_partdims;
  new_partdims.reserve(shape.size());
  for(auto const& pd: pds) {
    new_partdims.push_back(partdim_t::unions(pd));
  }

  // ijk,ij->ij
  // * has_agg is true,
  // * if the rhs input has a partition, then that is the choice partition

  // ijk->ij
  // * if the input has a partition, that is the partition.
  // * if the input does not have a partition (more likely),
  //   choice is none, k is singleton in new_partdims

  partition_t new_part = construct_minsized_partition(
    partition_t(new_partdims),
    choice
  );

  set_partition(id, new_part);
}

bool autopartition_state_t::set_from_inputs_and_recurse(int id) {
  // If this was already computed, stop here
  if(ret[id]) {
    return false;
  }
  if(is_mmlike(id)) {
    throw std::runtime_error("should not happen");
  }

  auto const& node = graph.nodes[id];

  if(node.op.is_input()) {
    // do not set inputs
    return false;
  }

  vector<partition_t> inn_parts;
  inn_parts.reserve(node.inns.size());
  for(auto const& inn_id: node.inns) {
    if(!ret[inn_id]) {
      // If not all of the input partitions have been set,
      // don't do anything
      return false;
    }
    inn_parts.push_back(ret[inn_id].value());
  }

  // Either this is
  // (1) non-mmlike einsummable
  // (2) a formation node or
  // (3) a concat node
  auto shape = node.op.shape();
  if(node.op.is_einsummable()) {
    optional<partition_t> choice;
    auto const& e = node.op.get_einsummable();
    vector<vector<partdim_t>> pds(shape.size());
    for(int which_inn = 0; which_inn != e.inns.size(); ++which_inn) {
      auto const& is = e.inns[which_inn];
      auto const& inn_partdims = inn_parts[which_inn].partdims;

      _update_pds_and_choice_from_input_for_nonmmlike(pds, choice, is, inn_partdims);
    }
    vector<partdim_t> join_partdims;
    join_partdims.reserve(shape.size());
    for(auto const& pd: pds) {
      join_partdims.push_back(partdim_t::unions(pd));
    }

    // Note: When ij,jk->ijk choice is none but not
    //       when ijk,jk->ijk
    partition_t new_partition = construct_minsized_partition(
      partition_t(join_partdims),
      choice
    );
    set_partition(id, new_partition);
  } else if(node.op.is_formation() || node.op.is_complexer()) {
    // This is a formation node.

    auto try_to_halve = [](partition_t& new_part) {
      bool can_halve = true;
      auto sizes = new_part.partdims.back().sizes();
      for(auto const& sz: sizes) {
        if(sz % 2 != 0) {
          can_halve = false;
        }
      }

      if(can_halve) {
        halve_last_dim_inplace(new_part);
      } else {
        // oh well
        partdim_t& partdim = new_part.partdims.back();
        partdim = partdim_t::singleton(partdim.total());
      }
    };

    auto const& inn_part     = inn_parts[0];
    auto const& inn_partdims = inn_part.partdims;
    auto const& inn_id       = node.inns[0];
    auto const& inn_node     = graph.nodes[inn_id];

    bool has_agg = inn_node.op.has_aggregation();

    if(has_agg) {
      auto const& e = inn_node.op.get_einsummable();
      int join_rank = e.join_shape.size();

      int n_aggregates = 1;
      for(int i = e.out_rank; i != join_rank; ++i) {
        auto const& inn_partdim = inn_partdims[i];
        n_aggregates *= inn_partdim.num_parts();
      }

      vector<partdim_t> partdims(
        inn_partdims.begin(),
        inn_partdims.begin() + e.out_rank);

      // We want to split a partdim in partdims so that
      // each block has n_aggregates sub blocks.
      //
      // Walk through the partdims and find the first one
      // for which that can happen and do so.
      //
      // Preferably this will happen at the first dimension
      // since things are row-major ordered, and that
      // produces a beter touch operation.
      for(partdim_t& partdim: partdims) {
        for(uint64_t sz: partdim.sizes()) {
          if(sz < n_aggregates) {
            break;
          }
        }
        // partdim is big enough
        partdim = partdim_t::split_each(partdim, n_aggregates);
        break;
      }
      // If none of the partdims are big enough, that is weird,
      // but none of them are split and that is fine.

      partition_t new_part = partition_t(partdims);

      if(node.op.is_complexer()) {
        if(dtype_is_real(node.op.out_dtype())) {
          // complex -> real
          double_last_dim_inplace(new_part);
        } else {
          // real -> complex
          try_to_halve(new_part);
        }
      }

      if(new_part.total_shape() != node.op.shape()) {
        throw std::runtime_error("new part is incorrect");
      }

      set_partition(id, new_part);
    } else {
      if(node.op.is_complexer()) {
        partition_t new_part = inn_parts[0];
        if(dtype_is_real(node.op.out_dtype())) {
          double_last_dim_inplace(new_part);
        } else {
          try_to_halve(new_part);
        }
        set_partition(id, new_part);
      } else {
        set_partition(id, inn_parts[0]);
      }
    }
  } else if(node.op.is_concat()) {
    auto const& concat = node.op.get_concat();

    vector<uint64_t> dim_szs;
    int rank = concat.shape().size();
    vector<vector<partdim_t>> all_pds(rank);

    for(auto const& inn_part: inn_parts) {
      auto const& pds = inn_part.partdims;
      for(int i = 0; i != rank; ++i) {
        auto const& partdim = pds[i];
        if(i == concat.dim) {
          vector_concatenate_into(dim_szs, partdim.sizes());
        } else {
          all_pds[i].push_back(partdim);
        }
      }
    }

    vector<partdim_t> pds;
    for(int i = 0; i != rank; ++i) {
      if(i == concat.dim) {
        pds.push_back(partdim_t::from_sizes(dim_szs));
      } else {
        pds.push_back(partdim_t::unions(all_pds[i]));
      }
    }

    partition_t new_part(pds);

    if(is_too_fine(new_part)) {
      return false;
    }
    set_partition(id, new_part);
  } else if(node.op.is_subset()) {
    auto const& subset = node.op.get_subset();
    auto const& inn_part = inn_parts[0];
    auto new_part = inn_part.subset(subset.get_hrect());
    set_partition(id, new_part);
  } else {
    throw std::runtime_error(
      "set_from_inputs_and_recurse: "
      "should not reach");
  }

  // This guy has been set, so we can recurse
  for(auto const& out_id: node.outs) {
    set_from_inputs_and_recurse(out_id);
  }

  return true;
}

void autopartition_state_t::set_mmlike(int id) {
  if(!is_mmlike(id)) {
    throw std::runtime_error("this is no mmlike!");
  }

  auto const& node = graph.nodes[id];

  auto shape = node.op.shape();

  partition_t p = partition_t::singleton(shape);

  if(product(shape) > nloc) {
    while(p.num_parts() < nloc) {
      p = make_finer(p);
    }
  }

  set_partition(id, p);
}

partition_t autopartition_state_t::construct_minsized_partition(
  partition_t const& maybe_too_fine_partition,
  optional<partition_t> const& choice) const
{
  if(choice) {
    if(is_too_fine(maybe_too_fine_partition)) {
      return choice.value();
    } else {
      return maybe_too_fine_partition;
    }
  }

  // there is no choice, gotta do something else

  partition_t p = maybe_too_fine_partition;
  do {
    p = make_coarser(p);
  } while(is_too_fine(p));

  return p;
}

bool autopartition_state_t::is_too_fine(partition_t const& partition) const
{
  return partition.num_parts() > nmax;
}
