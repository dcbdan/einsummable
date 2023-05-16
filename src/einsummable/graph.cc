#include "graph.h"

int graph_t::insert_input(
  placement_t placement)
{
  return this->insert(
    input_t{ .shape = placement.total_shape()},
    {},
    placement);
}

int graph_t::insert_input(
  partition_t partition)
{
  return this->insert_input(
    placement_t(placement_t(partition)));
}

int graph_t::insert_input(
  vector<uint64_t> shape)
{
  return this->insert_input(partition_t::singleton(shape));
}

int graph_t::insert_einsummable(
  placement_t placement,
  einsummable_t e,
  vector<int> inns)
{
  if(e.inns.size() != inns.size()) {
    throw std::runtime_error("did not get expected number of inputs");
  }

  auto expected_inn_shapes = e.inn_shapes();
  for(int i = 0; i != inns.size(); ++i) {
    if(!vector_equal(expected_inn_shapes[i], out_shape(inns[i]))) {
      throw std::runtime_error("shapes do not match: insert einsummable");
    }
  }

  return this->insert(
    e,
    inns,
    placement);
}

int graph_t::insert_einsummable(
  partition_t partition,
  einsummable_t e,
  vector<int> inns)
{
  return this->insert_einsummable(placement_t(partition), e, inns);
}

int graph_t::insert_einsummable(
  einsummable_t e,
  vector<int> inns)
{
  return this->insert_einsummable(
    partition_t::singleton(e.join_shape),
    e,
    inns);
}

int graph_t::insert_formation(
  placement_t placement,
  int inn,
  bool is_save)
{
  if(!vector_equal(placement.total_shape(), out_shape(inn))) {
    throw std::runtime_error("invalid shape: insert_formation");
  }

  return this->insert(
    formation_t {
      .shape = placement.total_shape(),
      .is_save = is_save },
    {inn},
    placement);
}

int graph_t::insert_formation(
  partition_t partition,
  int inn,
  bool is_save)
{
  return this->insert_formation(placement_t(partition), inn, is_save);
}

int graph_t::insert_formation(
  int inn,
  bool is_save)
{
  auto const& inn_node = nodes[inn];
  auto shape = inn_node.op.out_shape();
  return this->insert_formation(partition_t::singleton(shape), inn, is_save);
}

void graph_t::set_saves() {
  int num_nodes_time_zero = nodes.size();
  for(int i = 0; i != num_nodes_time_zero; ++i) {

    node_t& n = nodes[i];
    if(n.outs.size() == 0 && !n.op.is_save()) {
      if(n.op.is_formation()) {
        n.op.get_formation().is_save = true;
      } else {
        this->insert_formation(
          placement_t::join_to_out(n.placement, n.op.out_rank()),
          i,
          true);
      }
    }
  }
}

vector<uint64_t> graph_t::out_shape(int id) const {
  return nodes[id].op.out_shape();
}

vector<int> graph_t::get_order() const {
  // Because of the way the graph is constructed,
  // it must be the case that a valid ordering of the compute
  // graph is 0,1,2,...
  vector<int> ret(nodes.size());
  std::iota(ret.begin(), ret.end(), 0);
  return ret;
}

int graph_t::num_locs() const {
  int ret = 1;
  for(auto const& node: nodes) {
    ret = std::max(ret, node.num_locs());
  }

  return ret;
}

int graph_t::insert(
  op_t const& op,
  vector<int> inns,
  placement_t placement)
{
  int ret = nodes.size();
  nodes.push_back(node_t {
    .op = op,
    .inns = inns,
    .outs = {},
    .placement = placement
  });

  for(auto inn: inns) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

void graph_t::print() const {
  std::cout <<
    "graph[num nodes = " << nodes.size() << ", " <<
          "num locs = " << num_locs() << "]" << std::endl;
  std::cout << std::endl;

  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];

    std::cout << "node id: " << id
      << " with out shape " << node.op.out_shape() << std::endl;
    std::cout << "inputs: " << node.inns << std::endl;
    std::cout << "partition: " << node.placement.partition << std::endl;
    if(node.op.is_input()) {
      std::cout << "input" << std::endl;
    } else if(node.op.is_einsummable()) {
      std::cout << "einsummable " << node.op.get_einsummable() << std::endl;
    } else if(node.op.is_formation()) {
      std::cout << "formation (is save = " << std::boolalpha << node.op.is_save() << ")" << std::endl;
    }

    std::cout << std::endl;
  }
}

vector<int> graph_t::get_inputs() const {
  vector<int> ret;
  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];
    if(node.op.is_input()) {
      ret.push_back(id);
    }
  }
  return ret;
}

void graph_t::reset_annotations(
  vector<partition_t> const& new_partitions)
{
  if(new_partitions.size() != nodes.size()) {
    throw std::runtime_error("incorrect number of partitions");
  }

  for(int id = 0; id != nodes.size(); ++id) {
    node_t& node = nodes[id];
    partition_t const& new_part = new_partitions[id];
    nodes[id].placement = placement_t(new_part);
  }
}

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_t three_dimensional_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk,
  int num_processors)
{
  // The mapping from "procesor" grid to actual processor;
  // this is necessary for when pi*pj*pk > num_processors
  auto to_processor = [&](int i, int j, int k) {
    int index = idxs_to_index({pi,pj,pk}, {i,j,k});
    return index % num_processors;
  };

  // rcp = row, column, row_part
  enum class rcp_t { ijk, jki, ikj };

  // All matrices are partitioned along the rows and then the
  // columns, but then each block is further partitioned.
  //
  // So if A is partitioned rcp_t::ijk, then there are pi rows,
  // pj columns to form Aij. But each Aij is partitioned further
  // along the rows, into pk parts.
  // That means the partition is really (pi*pk, pj).
  auto make_matrix_partition = [&](rcp_t which) {
    int nr;
    int nc;
    int np;
    uint64_t dr;
    uint64_t dc;
    if(which == rcp_t::ijk) {
      nr = pi;
      nc = pj;
      np = pk;
      dr = di;
      dc = dj;
    } else if(which == rcp_t::jki) {
      nr = pj;
      nc = pk;
      np = pi;
      dr = dj;
      dc = dk;
    } else if(which == rcp_t::ikj) {
      nr = pi;
      nc = pk;
      np = pj;
      dr = di;
      dc = dk;
    } else {
      throw std::runtime_error("should not reach");
    }

    // Take dr, repeat it nr times,
    // then each of those nr blocks gets
    // split np ways.
    partdim_t part_row = partdim_t::split_each(
      partdim_t::repeat(nr, dr),
      np);

    partdim_t part_col = partdim_t::repeat(nc, dc);

    return partition_t({part_row, part_col});
  };

  // For which == rcp_t::ijk,
  // Aij(k) lives at processor (i,j,k).
  // That is, A is partitioned into pi rows, pj columns.
  // Each Aij is distributed across (i,j,*) and
  // Aij is chopped along it's rows forming Aij(k) for
  // some k in 0,...,pk-1.
  auto make_matrix_locs = [&](rcp_t which) {
    vector<int> shape;
    int nr;
    int nc;
    int np;
    if(which == rcp_t::ijk) {
      shape = {pi*pk, pj};
      nr = pi;
      nc = pj;
      np = pk;
    } else if(which == rcp_t::jki) {
      shape = {pj*pi, pk};
      nr = pj;
      nc = pk;
      np = pi;
    } else if(which == rcp_t::ikj) {
      shape = {pi*pj, pk};
      nr = pi;
      nc = pk;
      np = pj;
    } else {
      throw std::runtime_error("should not reach");
    }
    tensor_t<int> locs(shape);

    int i;
    int j;
    int k;
    for(int r = 0; r != nr; ++r) {
    for(int c = 0; c != nc; ++c) {
    for(int p = 0; p != np; ++p) {
      if(which == rcp_t::ijk) {
        i = r;
        j = c;
        k = p;
      } else if(which == rcp_t::jki) {
        i = p;
        j = r;
        k = c;
      } else if(which == rcp_t::ikj) {
        i = r;
        j = p;
        k = c;
      } else {
        throw std::runtime_error("should not reach");
      }

      locs(r*np + p, c) = to_processor(i,j,k);
    }}}

    return locs;
  };

  auto make_matrix_placement = [&](rcp_t rcp) {
    return placement_t(
      make_matrix_partition(rcp),
      make_matrix_locs(rcp)
    );
  };

  graph_t ret;

  int id_lhs = ret.insert_input(make_matrix_placement(rcp_t::ijk));
  int id_rhs = ret.insert_input(make_matrix_placement(rcp_t::jki));

  int id_op;
  {
    einsummable_t matmul = einsummable_t::from_matmul(di*pi, dj*pj, dk*pk);
    // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

    partition_t part({
      partdim_t::repeat(pi, di),
      partdim_t::repeat(pk, dk),
      partdim_t::repeat(pj, dj)
    });

    tensor_t<int> locs({pi,pk,pj});

    for(int i = 0; i != pi; ++i) {
    for(int j = 0; j != pj; ++j) {
    for(int k = 0; k != pk; ++k) {
      locs(i,k,j) = to_processor(i,j,k);
    }}}

    placement_t placement(part, locs);

    id_op = ret.insert_einsummable(placement, matmul, {id_lhs, id_rhs});
  }

  // the save node
  ret.insert_formation(make_matrix_placement(rcp_t::ikj), id_op, true);

  return ret;
}

graph_t straight_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk)
{
  graph_t graph;

  partdim_t pdi = partdim_t::repeat(pi, di);
  partdim_t pdj = partdim_t::repeat(pj, dj);
  partdim_t pdk = partdim_t::repeat(pk, dk);

  int id_lhs = graph.insert_input(partition_t({pdi,pdj}));
  int id_rhs = graph.insert_input(partition_t({pdj,pdk}));

  einsummable_t matmul = einsummable_t::from_matmul(pi*di, pj*dj, pk*dk);
  // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

  int id_join = graph.insert_einsummable(
    partition_t({pdi, pdk, pdj}),
    matmul,
    {id_lhs, id_rhs});

  graph.insert_formation(
    partition_t({pdi, pdk}),
    id_join,
    true);

  return graph;
}

struct autopartition_state_t {
  autopartition_state_t(
    graph_t const& g,
    uint64_t const& mms,
    uint64_t const& ms,
    set<tuple<int, int>> const& eqd,
    map<int, partition_t> const& fixed_constraints);


  // Set nodes from the input partitions and
  // return whether or not this node was set.
  // Does not set if input node, mmlike or
  // not all inputs are available.
  bool set_from_inputs_and_recurse(int id);

  // Set the partition of any non-mmlike node.
  // Decide the partition based on (1) the
  // usage partitions and (2) the input partitions,
  // if available.
  //
  // There is one caveat: formation
  // nodes that do not have an ouput are not set.
  // Consider
  //   node 0: A/Input
  //   node 1: C = A + B
  //   node 2: Form(C)
  // Where A does not have a partition.
  // Then, with output-formation nodes considered,
  // 1. set node 2 to singleton
  // 2. set C to intersection of node 2 and node B
  //    which is just node B
  // 3. Set node 0 to the partition of node 1
  //
  // How it'd work now, not setting output-formation
  // nodes:
  // 1. set node 1 to partition of node B
  // 2. set node 0 to partition of C
  // Now node 2 does not have a partition, so go
  // back and set it to the partition of C.
  void set_from_outputs_and_recurse(int id);

  void set_mmlike(int id);

  void set_partition(int id, partition_t const& p);

  bool is_mmlike(int id) const;
  bool is_output_formation(int id) const;

  // If the partition has blocks finer
  // than min_sizing, then either return
  // choice if not none, or make it coarser
  // by iteratively making dimensions coarser
  partition_t construct_minsized_partition(
    partition_t const& maybe_too_fine_partition,
    optional<partition_t> const& choice) const;

  bool is_too_fine(partition_t const& p) const;

  graph_t const& graph;
  uint64_t const& mmlike_sizing;
  uint64_t const& min_sizing;

  set<int> remaining;
  vector<optional<partition_t>> ret;

  equal_items_t<int> equal_items;
};

vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t mmlike_sizing,
  uint64_t min_sizing)
{
  return autopartition(graph, mmlike_sizing, min_sizing, {}, {});
}

vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t mmlike_sizing,
  uint64_t min_sizing,
  // make sure each of these pair have the same partition
  set<tuple<int, int>> const& equal_constraints,
  map<int, partition_t> const& fixed_constraints)
{
  autopartition_state_t state(
    graph, mmlike_sizing, min_sizing,
    equal_constraints, fixed_constraints);

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
  uint64_t const& mms,
  uint64_t const& ms,
  set<tuple<int, int>> const& eqd,
  map<int, partition_t> const& fixed_constraints)
  : graph(g), mmlike_sizing(mms), min_sizing(ms),
    equal_items(eqd)
{
  int n_nodes = graph.nodes.size();

  ret = vector<optional<partition_t>>(n_nodes);

  for(int i = 0; i != n_nodes; ++i) {
    remaining.insert(i);
  }

  for(auto const& [id, p]: fixed_constraints) {
    set_partition(id, p);
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

  if(equal_items.has(id)) {
    set<int> equiv_ids = equal_items.pop_at(id);
    for(int const& equiv_id: equiv_ids) {
      if(equiv_id != id) {
        set_partition(equiv_id, p);
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
    } else {
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

  // Either this is a non-mmlike einsummable or a formation node.
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
  } else {
    // This is a formation node.

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
      // for which that can happena nd do so.
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
      if(new_part.total_shape() != node.op.shape()) {
        throw std::runtime_error("new part is incorrect");
      }

      set_partition(id, new_part);
    } else {
      set_partition(id, inn_parts[0]);
    }
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

  // A list of (partsize, npart) that will be turned
  // into partitions. The total size of each block
  // will be the product of the partsize, which
  // should be close to mmlike_sizing.
  vector<tuple<uint64_t, int>> items;
  items.reserve(shape.size());
  for(uint64_t const& sz: shape) {
    items.emplace_back(sz, 1);
  }
  // Note: this should err on slightly
  //       smaller blocks than mmlike_sizing
  while(product(vector_mapfst(items)) > mmlike_sizing) {
    // find the item with the largest part size
    // and increment the number of parts
    int which = 0;
    uint64_t score = std::get<0>(items[0]);
    for(int i = 0; i != shape.size(); ++i) {
      auto const& [s,d_at_s] = items[i];
      if(s > score) {
        which = i;
        score = s;
      } else if(s == score) {
        // tie breaks go to the larger dimension
        uint64_t d_at_which = std::get<1>(items[which]);
        if(d_at_s > d_at_which) {
          which = i;
          score = s;
        }
      }
    }
    auto& [partsize, npart] = items[which];
    npart += 1;
    partsize = shape[which] / npart;
  }

  vector<partdim_t> partdims;
  partdims.reserve(shape.size());
  for(int i = 0; i != shape.size(); ++i) {
    auto const& [_, npart] = items[i];
    uint64_t const& sz = shape[i];
    partdims.push_back(partdim_t::split(sz, npart));
  }

  set_partition(id, partition_t(partdims));
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
  auto shape = p.total_shape();
  while(is_too_fine(p)) {
    vector<tuple<uint64_t, int>> ms;
    ms.reserve(p.partdims.size());
    for(int i = 0; i != p.partdims.size(); ++i) {
      auto const& partdim = p.partdims[i];
      if(partdim.num_parts() > 1) {
        auto sizes = partdim.sizes();
        uint64_t const& sz = *std::min_element(sizes.begin(), sizes.end());
        ms.emplace_back(sz, i);
      }
    }

    if(ms.size() == 0) {
      throw std::runtime_error(
        "minsized: should not happen, nothing to make coarser"
      );
    }

    auto const& [_, argmin] = *std::min_element(ms.begin(), ms.end());

    vector<partdim_t> new_partdims = p.partdims;
    int arg_npart = new_partdims[argmin].num_parts();
    new_partdims[argmin] = partdim_t::split(shape[argmin], arg_npart-1);
    p = partition_t(new_partdims);
  }
  return p;
}

bool autopartition_state_t::is_too_fine(partition_t const& partition) const
{
  if(product(partition.total_shape()) < min_sizing) {
    // It's not _too_ fine if the total size is < min_sizing!
    return false;
  }
  uint64_t min_block_size = 1;
  for(auto const& partdim: partition.partdims) {
    auto sizes = partdim.sizes();
    min_block_size *= (*std::min_element(sizes.begin(), sizes.end()));
  }
  return min_block_size < min_sizing;
}

