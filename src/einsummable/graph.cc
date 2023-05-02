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

    std::cout << "inputs: " << node.inns << std::endl;
    std::cout << "partition: " << node.placement.partition << std::endl;
    if(node.op.is_input()) {
      std::cout << "input" << std::endl;
    } else if(node.op.is_einsummable()) {
      std::cout << "einsummable" << std::endl;
    } else if(node.op.is_formation()) {
      std::cout << "formation (is save = " << std::boolalpha << node.op.is_save() << ")" << std::endl;
    }

    std::cout << std::endl;
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
  // Decide the partition based on the the usage
  // partitions.
  void set_from_outputs_and_recurse(int id);

  void set_mmlike(int id);

  void set_partition(int id, partition_t const& p);

  bool is_mmlike(int id) const;

  // If the partition has blocks finer
  // than min_sizing, then return choice
  partition_t construct_minsized_partition(
    partition_t const& maybe_too_fine_partition,
    partition_t const& choice) const;
  // If the partition has blocks finer
  // than min_sizing, then iteratively
  // add singleton dimensions.
  partition_t construct_minsized_partition(
    partition_t const& maybe_too_fine_partition) const;

  bool is_too_fine(partition_t const& p) const;

  graph_t const& graph;
  uint64_t const& mmlike_sizing;
  uint64_t const& min_sizing;

  set<int> remaining;
  vector<optional<partition_t>> ret;
  map<int, int> equald;
};

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

  // Seed the subsequent computation by
  // setting all mmlike partitions directly
  int n_nodes = graph.nodes.size();
  for(int id = 0; id != n_nodes; ++id) {
    if(state.is_mmlike(id)) {
      state.set_mmlike(id);
    }
  }

  // Now walk forwards from the nodes that have been
  // set and set remaining nodes from it's input partitions,
  //
  // Then walk backwards setting a node from it's output partitions.

  // Set remaining nodes forward from inputs
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

  // Set remaining nodes backwards from outputs
  // This is guaranteed to finish because there must exist
  // nodes that have no outputs.
  while(state.remaining.size() > 0) {
    set<int> rem = state.remaining;
    for(auto const& id: rem) {
      // will not do any setting if outputs are not available.
      state.set_from_outputs_and_recurse(id);
    }
  }

  return vector_from_each_method(state.ret, partition_t, value);
}

autopartition_state_t::autopartition_state_t(
  graph_t const& g,
  uint64_t const& mms,
  uint64_t const& ms,
  set<tuple<int, int>> const& eqd,
  map<int, partition_t> const& fixed_constraints)
  : graph(g), mmlike_sizing(mms), min_sizing(ms)
{
  int n_nodes = graph.nodes.size();

  ret = vector<optional<partition_t>>(n_nodes);

  for(int i = 0; i != n_nodes; ++i) {
    remaining.insert(i);
  }

  for(auto const& [i,j]: eqd) {
    equald.insert({i,j});
    equald.insert({j,i});
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

  if(equald.count(id) > 0) {
    int other = equald.at(id);

    // first erase then recurse on other
    // (the other way around won't work!)
    equald.erase(id);
    equald.erase(other);

    set_partition(other, p);
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

void autopartition_state_t::set_from_outputs_and_recurse(int id) {
  if(ret[id]) {
    return;
  }

  if(is_mmlike(id)) {
    throw std::runtime_error("should not happen");
  }

  auto const& node = graph.nodes[id];

  if(node.outs.size() == 0) {
    set_partition(id, partition_t::singleton(node.op.shape()));
    return;
  }

  // for each usage, make sure it is set and get the usage(s)
  vector<partition_t> out_ps;
  for(auto const& out_id: node.outs) {
    set_from_outputs_and_recurse(out_id);

    auto const& out_node = graph.nodes[out_id];
    if(out_node.op.is_einsummable()) {
      auto const& e = out_node.op.get_einsummable();
      auto const& out_inns = out_node.inns;
      for(int which_inn = 0; which_inn != out_inns.size(); ++which_inn) {
        if(out_inns[which_inn] == id) {
          auto const& pds = ret[out_id].value().partdims;
          out_ps.push_back(partition_t(e.get_input_from_join(pds, which_inn)));
        }
      }
    } else {
      out_ps.push_back(ret[out_id].value());
    }
  }

  auto shape = node.op.shape();
  vector<vector<partdim_t>> pds;
  pds.reserve(shape.size());
  {
    // set pds after taking union equivalent to singleton
    partition_t singleton = partition_t::singleton(shape);
    for(int i = 0; i != singleton.partdims.size(); ++i) {
      pds[i].push_back(singleton.partdims[i]);
    }
  }

  for(auto const& partition: out_ps) {
    // Note: if partdims.size() != shape.size(), then
    //       this is an agg node
    auto const& partdims = partition.partdims;
    for(int rank = 0; rank != partdims.size(); ++rank) {
      pds[rank].push_back(partdims[rank]);
    }
  }

  // If this is an agg node, then the agg dimensions
  // are singleton.

  vector<partdim_t> new_partdims;
  new_partdims.reserve(shape.size());
  for(auto const& pd: pds) {
    new_partdims.push_back(partdim_t::unions(pd));
  }

  // account for the possible agg dims
  vector<partdim_t> choice = out_ps[0].partdims;
  for(int i = choice.size(); i != new_partdims.size(); ++i) {
    choice.push_back(partdim_t::from_sizes({shape[i]}));
  }

  partition_t new_part = construct_minsized_partition(
    partition_t(new_partdims),
    partition_t(choice)
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

      if(inn_partdims.size() == shape.size() && !choice) {
        vector<partdim_t> choice_partdims(shape.size());
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
    vector<partdim_t> join_partdims;
    join_partdims.reserve(shape.size());
    for(auto const& pd: pds) {
      join_partdims.push_back(partdim_t::unions(pd));
    }

    if(choice) {
      partition_t new_partition = construct_minsized_partition(
        partition_t(join_partdims),
        choice.value());
      set_partition(id, new_partition);
    } else {
      // Note: this block will get it when ij,jk->ijk but not
      //       when ijk,jk->ijk.
      partition_t new_partition = construct_minsized_partition(
        partition_t(join_partdims));
      set_partition(id, new_partition);
    }
  } else {
    // This is a formation node.

    auto const& inn_part     = inn_parts[0];
    auto const& inn_partdims = inn_part.partdims;
    auto const& inn_id       = node.inns[0];
    auto const& inn_node     = graph.nodes[inn_id];

    bool has_agg = false;
    if(inn_node.op.is_einsummable()) {
      auto const& e = inn_node.op.get_einsummable();
      if(e.out_rank < e.join_shape.size()) {
        has_agg = true;
      }
    }

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
        bool bigenough = true;
        for(uint64_t sz: partdim.sizes()) {
          if(sz < n_aggregates) {
            bigenough = false;
            break;
          }
        }
        if(bigenough) {
          if(partdim.total() > n_aggregates) {
            partdim = partdim_t::split_each(partdim, n_aggregates);
            break;
          }
        }
      }
      // If none of the partdims are big enough, that is weird,
      // but none of them are split and that is fine.

      set_partition(id, partition_t(partdims));
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
  // Note: this should result in slightly
  //       smaller blocks than mmlike_sizing
  while(product(vector_mapfst(items)) >= mmlike_sizing) {
    // find the item with the largest part size
    // and increment the number of parts
    int which = 0;
    uint64_t score = std::get<0>(items[0]);
    for(int i = 0; i != shape.size(); ++i) {
      auto const& [s,_] = items[i];
      if(s > score) {
        which = i;
        score = s;
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
  partition_t const& choice) const
{
  if(is_too_fine(maybe_too_fine_partition)) {
    return choice;
  } else {
    return maybe_too_fine_partition;
  }
}

partition_t autopartition_state_t::construct_minsized_partition(
  partition_t const& maybe_too_fine_partition) const
{
  partition_t p = maybe_too_fine_partition;
  auto shape = p.total_shape();
  while(is_too_fine(p)) {
    vector<uint64_t> ms;
    for(auto const& partdim: p.partdims) {
      auto sizes = partdim.sizes();
      ms.push_back(*std::min_element(sizes.begin(), sizes.end()));
    }
    int argmin = std::min_element(ms.begin(), ms.end()) - ms.begin();
    vector<partdim_t> new_partdims = p.partdims;
    int arg_npart = new_partdims[argmin].num_parts();
    if(arg_npart < 1) {
      throw std::runtime_error(
        "minsized partition construction: should not happen");
    }
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
  return min_block_size >= min_sizing;
}

