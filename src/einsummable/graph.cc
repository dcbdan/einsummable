#include "graph.h"

int graph_constructor_t::insert_input(
  placement_t placement)
{
  int ret = graph.insert_input(placement.total_shape());
  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_input(
  partition_t partition)
{
  return insert_input(placement_t(partition));
}

int graph_constructor_t::insert_input(
  vector<uint64_t> shape)
{
  return insert_input(partition_t::singleton(shape));
}

int graph_t::insert_input(
  vector<uint64_t> shape)
{
  return this->insert(
    input_t { .shape = shape },
    {});
}

int graph_constructor_t::insert_einsummable(
  placement_t placement,
  einsummable_t e,
  vector<int> inns)
{
  if(placement.total_shape() != e.join_shape) {
    throw std::runtime_error("graph constructor: invalid insert_einsummable inputs");
  }

  int ret = graph.insert_einsummable(e, inns);
  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_einsummable(
  partition_t partition,
  einsummable_t e,
  vector<int> inns)
{
  return insert_einsummable(placement_t(partition), e, inns);
}

int graph_constructor_t::insert_einsummable(
  einsummable_t e,
  vector<int> inns)
{
  auto const& shape = e.join_shape;
  return insert_einsummable(partition_t::singleton(shape), e, inns);
}

int graph_t::insert_einsummable(
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
    inns);
}

int graph_constructor_t::insert_formation(
  placement_t placement,
  int inn,
  bool is_save)
{
  if(!vector_equal(placement.total_shape(), graph.out_shape(inn))) {
    throw std::runtime_error("invalid shape: insert_formation (constructing)");
  }

  int ret = graph.insert_formation(inn, is_save);
  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_formation(
  partition_t partition,
  int inn,
  bool is_save)
{
  return this->insert_formation(placement_t(partition), inn, is_save);
}

int graph_constructor_t::insert_formation(
  int inn,
  bool is_save)
{
  auto const& inn_node = graph.nodes[inn];
  auto shape = inn_node.op.out_shape();
  return this->insert_formation(partition_t::singleton(shape), inn, is_save);
}

int graph_t::insert_formation(
  int inn,
  bool is_save)
{
  return this->insert(
    formation_t {
      .shape = out_shape(inn),
      .is_save = is_save },
    {inn});
}

vector<placement_t> graph_constructor_t::get_placements() const {
  vector<placement_t> ret;
  ret.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    ret.push_back(placements.at(gid));
  }
  return ret;
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

int graph_t::insert(
  op_t const& op,
  vector<int> inns)
{
  int ret = nodes.size();
  nodes.push_back(node_t {
    .op = op,
    .inns = inns,
    .outs = {}
  });

  for(auto inn: inns) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

void graph_t::print() const {
  std::cout <<
    "graph[num nodes = " << nodes.size() << "]" << std::endl;
  std::cout << std::endl;

  for(int id = 0; id != nodes.size(); ++id) {
    auto const& node = nodes[id];

    std::cout << "node id: " << id
      << " with out shape " << node.op.out_shape() << std::endl;
    std::cout << "inputs: " << node.inns << std::endl;
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

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_constructor_t
three_dimensional_matrix_multiplication(
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

  graph_constructor_t ret;

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

graph_constructor_t straight_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk)
{
  graph_constructor_t graph;

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

// Swap B and E when
// lhs = B interval and
// rhs = D interval.
//   A B C D E -> A D C B E
template <typename T>
void rotate_sections(
  vector<T>& items,
  tuple<int,int> lhs,
  tuple<int,int> rhs)
{
  {
    // make the lhs come before the rhs side
    auto const& [a,b] = lhs;
    auto const& [c,d] = rhs;
    if(a > c) {
      std::swap(lhs, rhs);
    }
  }

  auto const& [a,b] = lhs;
  auto const& [c,d] = rhs;

  if(a >= b || b > c || c >= d) {
    throw std::runtime_error("should not happen");
  }

  std::rotate(
    items.begin() + a,
    items.begin() + c,
    items.begin() + d);

  if(b != c) {
    int nl = b-a;
    int nr = d-c;
    std::rotate(
      items.begin() + (a + nr),
      items.begin() + (a + nr + nl),
      items.begin() + d);
  }
}

graph_writer_t::tensor_t::tensor_t(
  vector<uint64_t> const& _shape,
  vector<uint64_t> const& _full_shape,
  int _id,
  graph_writer_t& _self)
    : shape(_shape), full_shape(_full_shape),
      id(_id), self(_self)
{
  modes.resize(full_shape.size());
  std::iota(modes.begin(), modes.end(), 0);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::permute(int i, int j) const
{
  auto breaks = get_breaks();

  tensor_t ret = *this;

  std::swap(ret.shape.at(i), ret.shape.at(j));

  rotate_sections(ret.modes,      breaks[i], breaks[j]);
  rotate_sections(ret.full_shape, breaks[i], breaks[j]);

  return ret;
}

bool _check_new_shape(
  vector<uint64_t> const& full_shape,
  vector<uint64_t> const& new_shape)
{
  if(new_shape.size() > full_shape.size()) {
    return false;
  }

  vector<uint64_t> full(full_shape.size());
  vector<uint64_t> news(new_shape.size());

  std::inclusive_scan(
    full_shape.begin(), full_shape.end(),
    full.begin(),
    std::multiplies<>{});
  std::inclusive_scan(
    new_shape.begin(),  new_shape.end(),
    news.begin(),
    std::multiplies<>{});

  auto n = news.begin();
  auto f = full.begin();
  for(; n != news.end(); ++n) {
    for(; f != full.end(); ++f) {
      if(*n == *f) {
        break;
      }
    }
    if(f == full.end()) {
      return false;
    }
    ++f;
  }
  return true;
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::view(vector<uint64_t> new_shape) const
{
  if(!_check_new_shape(full_shape, new_shape)) {
    throw std::runtime_error(
      "could not create view " + write_with_ss(new_shape) + " from " +
      write_with_ss(full_shape));
  }

  tensor_t ret = *this;
  ret.shape = new_shape;
  return ret;
}

void graph_writer_t::tensor_t::save() {
  // permute this node if necc
  physically_permute();

  {
    auto& op = self.graph.nodes[id].op;
    if(op.is_formation()) {
      op.get_formation().is_save = true;
      return;
    }
  }

  id = self.graph.insert_formation(id, true);
}

void graph_writer_t::tensor_t::physically_permute() {
  vector<int> no_permute_modes(modes.size());
  std::iota(
    no_permute_modes.begin(),
    no_permute_modes.end(),
    0);

  if(modes == no_permute_modes) {
    return;
  }

  string str;
  {
    vector<char> letters(modes.size());
    std::iota(letters.begin(), letters.end(), 'a');

    string inn(letters.begin(), letters.end());

    string out;
    for(auto const& m: modes) {
      out.push_back(letters[m]);
    }

    str = inn + "->" + out;
  }

  id = self._insert_elementwise(
    str,
    scalarop_t::make_identity(),
    id);
  modes = no_permute_modes;
}

vector<tuple<int,int>>
graph_writer_t::tensor_t::get_breaks() const
{
  return get_breaks_(shape, full_shape);
}

vector<tuple<int,int>>
graph_writer_t::tensor_t::get_breaks_(
  vector<uint64_t> const& shape,
  vector<uint64_t> const& full_shape)
{
  vector<tuple<int,int>> ret;

  int f = 0;
  for(int d = 0; d != shape.size(); ++d) {
    ret.emplace_back(f,0);

    auto& [_, b] = ret.back();

    int sz = shape[d];
    while(sz != 1) {
      sz /= full_shape[f];
      ++f;
    }
    b = f;
  }

  return ret;
}

vector<vector<uint64_t>>
graph_writer_t::tensor_t::_full_shape() const
{
  auto breaks = get_breaks();
  vector<vector<uint64_t>> ret;
  for(auto const& [b,e]: breaks) {
    ret.push_back({});
    for(int i = b; i != e; ++i) {
      ret.back().push_back(full_shape[i]);
    }
  }
  return ret;
}

graph_writer_t::tensor_t
graph_writer_t::insert_input(
  vector<uint64_t> shape)
{
  int id = graph.insert_input(shape);
  return tensor_t(shape, shape, id, *this);
}

vector<uint64_t> graph_writer_t::to_einsummable_info_t::get_out_full_shape() const
{
  return vector<uint64_t>(
    full_join_shape.begin(),
    full_join_shape.begin() + full_out_rank);
}

vector<uint64_t> graph_writer_t::to_einsummable_info_t::get_out_shape() const
{
  return vector<uint64_t>(
    join_shape.begin(),
    join_shape.begin() + out_rank);
}

einsummable_t graph_writer_t::to_einsummable_info_t::build_einsummable(
  scalarop_t join,
  optional<castable_t> castable) const
{
  return einsummable_t(
    full_join_shape,
    full_inns,
    full_out_rank,
    join,
    castable);
}

graph_writer_t::tensor_t
graph_writer_t::insert_contraction(
  string str,
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  auto maybe_info = make_einsummable_info(str, {lhs,rhs});
  if(!maybe_info) {
    throw std::runtime_error("graph_writer_t constraction: could not create einsummable");
  }

  auto const& info = maybe_info.value();

  einsummable_t e = info.build_einsummable(scalarop_t::make_mul(), castable_t::add);

  if(!e.is_contraction()) {
    throw std::runtime_error("build einsummable is not a contraction");
  }

  int id = graph.insert_einsummable(e, {lhs.id, rhs.id});

  vector<int> modes(info.full_out_rank);
  std::iota(modes.begin(), modes.end(), 0);

  return tensor_t(
    info.get_out_shape(),
    info.get_out_full_shape(),
    id,
    *this);
}

int graph_writer_t::_insert_elementwise(
  string str,
  scalarop_t op,
  int id)
{
  auto [inns, out_rank] = einsummable_t::parse_str(str);
  if(inns.size() != 1) {
    throw std::runtime_error("invalid str to _insert_elementwise");
  }

  auto inn_shape = graph.out_shape(id);
  if(out_rank != inn_shape.size()) {
    throw std::runtime_error("_insert_elementwise not elementwise");
  }

  auto maybe_join_shape = einsummable_t::construct_join_shape(inns, { inn_shape });
  if(!maybe_join_shape) {
    throw std::runtime_error("_insert_elementwise: no join shape");
  }
  auto const& join_shape = maybe_join_shape.value();

  einsummable_t e(join_shape, inns, out_rank, op);
  if(inn_shape != e.inn_shapes()[0]) {
    throw std::runtime_error("_insert_elementwise something wrong");
  }

  return graph.insert_einsummable(e, {id});
}

// There are a few things that get packed into this.
// (1) All the input tensors maintain a permutation
// (2) All the tensors have a current shape and a current full shape
//     (which are the shape after the virtual permutation)
// (3) str refers to the current shape
optional<graph_writer_t::to_einsummable_info_t>
graph_writer_t::make_einsummable_info(
  string str,
  vector<graph_writer_t::tensor_t> const& tensors)
{
  // does the str and current shapes line up correctly?
  auto [inns, out_rank] = einsummable_t::parse_str(str);

  vector<uint64_t> join_shape;
  {
    vector<vector<uint64_t>> inn_shapes;
    for(auto const& inn_tensor: tensors) {
      inn_shapes.push_back(inn_tensor.shape);
    }
    auto maybe = einsummable_t::construct_join_shape(inns, inn_shapes);
    if(!maybe) {
      return std::nullopt;
    }
    join_shape = maybe.value();
  }

  vector<uint64_t> full_join_shape;
  int full_out_rank;
  {
    vector<vector<vector<uint64_t>>> fs;
    for(auto const& inn_tensor: tensors) {
      fs.push_back(inn_tensor._full_shape());
    }
    auto maybe = einsummable_t::construct_join_shape_(
      inns, fs, vector<uint64_t>{}, vector_equal<uint64_t>);
    if(!maybe) {
      return std::nullopt;
    }

    auto const& v = maybe.value();

    full_out_rank = 0;
    for(int i = 0; i != out_rank; ++i) {
      full_out_rank += v[i].size();
    }

    full_join_shape = vector_flatten(v);
  }

  vector<vector<int>> full_inns_before_perm;
  {
    auto breaks = tensor_t::get_breaks_(join_shape, full_join_shape);
    for(vector<int> const& is: inns) {
      full_inns_before_perm.push_back({});
      auto& full_is = full_inns_before_perm.back();
      for(auto const& ii: is) {
        auto const& [b,e] = breaks[ii];
        for(int i = b; i != e; ++i) {
          full_is.push_back(i);
        }
      }
    }
  }

  vector<vector<int>> full_inns;
  {
    // permute each input based on modes
    for(int which_inn = 0; which_inn != inns.size(); ++which_inn) {
      tensor_t const& inn_tensor = tensors[which_inn];
      vector<int> const& modes = inn_tensor.modes;

      auto& before_perm = full_inns_before_perm[which_inn];

      full_inns.push_back(vector<int>(before_perm.size(), -1));
      auto& after_perm = full_inns.back();

      // modes = 01234->42310
      // where before_perm currently contains v4 v2 v3 v1 v0
      // and after_perm needs to contain v0 v1 v2 v3 v4
      for(int after_i = 0; after_i != after_perm.size(); ++after_i) {
        int const& before_i = modes[after_i];
        after_perm[after_i] = before_perm[before_i];
      }
    }
  }

  return to_einsummable_info_t {
    .full_join_shape = full_join_shape,
    .join_shape = join_shape,
    .full_inns = full_inns,
    .full_out_rank = full_out_rank,
    .out_rank = out_rank
  };
}

