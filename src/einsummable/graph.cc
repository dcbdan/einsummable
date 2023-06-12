#include "graph.h"

concat_t::concat_t(int d, dtype_t dt, vector<vector<uint64_t>> const& ss):
  dim(d), dtype(dt), inn_shapes(ss)
{
  optional<string> err_msg = check_concat_shapes(dim, inn_shapes);
  if(err_msg) {
    throw std::runtime_error("concat_t: " + err_msg.value());
  }

  if(inn_shapes.size() <= 1) {
    throw std::runtime_error("concat_t: expects >1 input");
  }
}

vector<uint64_t> concat_t::shape() const
{
  vector<uint64_t> ret = inn_shapes[0];
  for(int i = 1; i != inn_shapes.size(); ++i) {
    ret[dim] += inn_shapes[i][dim];
  }
  return ret;
}

vector<uint64_t> concat_t::dim_parts() const
{
  vector<uint64_t> ret;
  ret.reserve(inn_shapes.size());
  for(auto const& inn_shape: inn_shapes) {
    ret.push_back(inn_shape[dim]);
  }
  return ret;
}

vector<tuple<uint64_t, uint64_t>>
concat_t::get_hrect(int which_inn) const
{
  vector<tuple<uint64_t, uint64_t>> ret;
  auto offsets = get_offsets();
  int rank = inn_shapes[0].size();
  ret.reserve(rank);
  for(int which_dim = 0; which_dim != rank; ++which_dim) {
    if(which_dim == dim) {
      uint64_t offset = offsets[which_inn];
      uint64_t const& sz = inn_shapes[which_inn][dim];
      ret.emplace_back(offset, offset + sz);
    } else {
      uint64_t const& sz = inn_shapes[0][which_dim];
      ret.emplace_back(0, sz);
    }
  }
  return ret;
}

vector<uint64_t> concat_t::get_offsets() const {
  vector<uint64_t> ret(inn_shapes.size());
  auto ds = dim_parts();

  // 0, ds[0], ds[0] + ds[1], ...
  std::exclusive_scan(
    ds.begin(),
    ds.end(),
    ret.begin(),
    0);

  return ret;
}

int graph_constructor_t::insert_input(
  placement_t placement, dtype_t dtype)
{
  int ret = graph.insert_input(placement.total_shape(), dtype);
  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_input(
  partition_t partition,
  dtype_t dtype)
{
  return insert_input(placement_t(partition), dtype);
}

int graph_constructor_t::insert_input(
  vector<uint64_t> shape,
  dtype_t dtype)
{
  return insert_input(partition_t::singleton(shape), dtype);
}

int graph_t::insert_input(
  vector<uint64_t> shape,
  dtype_t dtype)
{
  return this->insert(
    input_t { .dtype = dtype, .shape = shape },
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
      .dtype = out_dtype(inn),
      .shape = out_shape(inn),
      .is_save = is_save },
    {inn});
}

int graph_constructor_t::insert_to_complex(
  placement_t placement,
  int inn)
{
  int ret = graph.insert_to_complex(inn);
  if(!vector_equal(graph.out_shape(ret), placement.total_shape())) {
    throw std::runtime_error("invalid shape: insert_to_complex (constructing)");
  }

  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_to_complex(
  partition_t partition,
  int inn)
{
  return this->insert_to_complex(placement_t(partition), inn);
}

int graph_constructor_t::insert_to_complex(
  int inn)
{
  auto shape = graph.out_shape(inn);
  shape.back() /= 2;
  return this->insert_to_complex(partition_t::singleton(shape), inn);
}

int graph_t::insert_to_complex(int inn)
{
  if(out_dtype(inn) != dtype_t::f32) {
    throw std::runtime_error("can only convert to dtype_t::c64");
  }
  vector<uint64_t> shape = out_shape(inn);
  if(shape.back() % 2 == 1) {
    throw std::runtime_error("must have last even last input dim");
  }
  shape.back() /= 2;

  return this->insert(
    complexer_t {
      .dtype = dtype_t::c64,
      .shape = shape
    },
    {inn});
}

int graph_constructor_t::insert_to_real(
  placement_t placement,
  int inn)
{
  int ret = graph.insert_to_real(inn);
  if(!vector_equal(graph.out_shape(ret), placement.total_shape())) {
    throw std::runtime_error("invalid shape: insert_to_real (constructing)");
  }

  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_to_real(
  partition_t partition,
  int inn)
{
  return this->insert_to_real(placement_t(partition), inn);
}

int graph_constructor_t::insert_to_real(
  int inn)
{
  auto shape = graph.out_shape(inn);
  shape.back() *= 2;
  return this->insert_to_real(partition_t::singleton(shape), inn);
}

int graph_t::insert_to_real(int inn)
{
  if(out_dtype(inn) != dtype_t::c64) {
    throw std::runtime_error("can only convert from dtype_t::c64");
  }
  vector<uint64_t> shape = out_shape(inn);
  shape.back() *= 2;

  return this->insert(
    complexer_t {
      .dtype = dtype_t::f32,
      .shape = shape
    },
    {inn});
}

int graph_constructor_t::insert_concat(
  placement_t placement,
  int dim,
  vector<int> inns)
{
  int ret = graph.insert_concat(dim, inns);

  if(placement.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid concat");
  }

  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_concat(
  partition_t partition,
  int dim,
  vector<int> inns)
{
  int ret = graph.insert_concat(dim, inns);

  if(partition.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid concat");
  }

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_constructor_t::insert_concat(
  int dim,
  vector<int> inns)
{
  int ret = graph.insert_concat(dim, inns);

  partition_t partition = partition_t::singleton(graph.out_shape(ret));

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_t::insert_concat(
  int dim,
  vector<int> inns)
{
  if(inns.size() <= 1) {
    throw std::runtime_error("concat must have multiple arguments");
  }

  dtype_t dtype = out_dtype(inns[0]);
  for(int i = 1; i != inns.size(); ++i) {
    if(out_dtype(inns[i]) != dtype) {
      throw std::runtime_error("dtype error at insert_concat");
    }
  }

  vector<vector<uint64_t>> shapes;
  for(int const& inn: inns) {
    shapes.push_back(out_shape(inn));
  }

  return this->insert(concat_t(dim, dtype, shapes), inns);
}

dtype_t graph_t::complexer_t::inn_dtype() const {
  if(dtype == dtype_t::f32) {
    return dtype_t::c64;
  }
  if(dtype == dtype_t::c64) {
    return dtype_t::f32;
  }
  throw std::runtime_error("inn_dtype complexer: invalid dtype");
}

vector<uint64_t>
graph_t::complexer_t::inn_shape() const {
  vector<uint64_t> ret = shape;
  if(dtype_is_real(dtype)) {
    ret.back() *= 2;
    return ret;
  } else if(dtype_is_complex(dtype)) {
    if(ret.back() % 2 == 1) {
      throw std::runtime_error("invalid complexer shape");
    }
    ret.back() /= 2;
    return ret;
  } else {
    throw std::runtime_error("should not reach");
  }
}

dtype_t graph_t::op_t::out_dtype() const {
  if(is_input()) {
    return get_input().dtype;
  }
  if(is_formation()) {
    return get_formation().dtype;
  }
  if(is_complexer()) {
    return get_complexer().dtype;
  }
  if(is_concat()) {
    return get_concat().dtype;
  }
  if(is_einsummable()) {
    return get_einsummable().out_dtype();
  }
  throw std::runtime_error("graph::op_t should not reach");
}

vector<uint64_t>
graph_t::op_t::out_shape() const {
  if(is_input()) {
    return get_input().shape;
  }
  if(is_formation()) {
    return get_formation().shape;
  }
  if(is_complexer()) {
    return get_complexer().shape;
  }
  if(is_concat()) {
    return get_concat().shape();
  }
  if(is_einsummable()) {
    return get_einsummable().out_shape();
  }
  throw std::runtime_error("graph::op_t should not reach");
}

vector<uint64_t>
graph_t::op_t::shape() const {
  if(is_input()) {
    return get_input().shape;
  }
  if(is_formation()) {
    return get_formation().shape;
  }
  if(is_complexer()) {
    return get_complexer().shape;
  }
  if(is_concat()) {
    return get_concat().shape();
  }
  if(is_einsummable()) {
    return get_einsummable().join_shape;
  }
  throw std::runtime_error("graph::op_t should not reach");
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

dtype_t graph_t::out_dtype(int id) const {
  return nodes[id].op.out_dtype();
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

    std::cout << "node id]: " << id
      << " with out shape " << node.op.out_shape()
      << " | " << node.op.out_dtype() << std::endl;
    std::cout << "inputs: " << node.inns << std::endl;
    if(node.op.is_input()) {
      std::cout << "input" << std::endl;
    } else if(node.op.is_einsummable()) {
      std::cout << "einsummable " << node.op.get_einsummable() << std::endl;
    } else if(node.op.is_formation()) {
      std::cout << "formation (is save = " << std::boolalpha << node.op.is_save() << ")" << std::endl;
    } else if(node.op.is_complexer()) {
      if(node.op.get_complexer().is_to_real()) {
        std::cout << "complexer (to real)" << std::endl;
      } else {
        std::cout << "complexer (to complex)" << std::endl;
      }
    } else if(node.op.is_concat()) {
      std::cout << "concat[dim=" << node.op.get_concat().dim << "]" << std::endl;
    } else {
      throw std::runtime_error("graph_t print should not reach");
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
graph_writer_t::tensor_t::transpose(int i, int j) const
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
      "could not create view from " + write_with_ss(new_shape) + " to " +
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

dtype_t graph_writer_t::tensor_t::get_dtype() const {
  return self.graph.out_dtype(id);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_complex() const {
  return self.to_complex(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_real() const {
  return self.to_real(*this);
}

graph_writer_t::tensor_t&
graph_writer_t::tensor_t::operator=(
  graph_writer_t::tensor_t const& other)
{
  shape = other.shape;
  full_shape = other.full_shape;
  modes = other.modes;
  id = other.id;

  return *this;
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
graph_writer_t::input(
  vector<uint64_t> shape,
  dtype_t dtype)
{
  int id = graph.insert_input(shape, dtype);
  return tensor_t(shape, shape, id, *this);
}

vector<uint64_t> graph_writer_t::to_einsummable_info_t::get_out_full_shape() const
{
  return vector<uint64_t>(
    full_join_shape.begin(),
    full_join_shape.begin() + full_out_rank);
}

vector<partition_t> graph_t::make_singleton_partition() const {
  vector<partition_t> ps;
  ps.reserve(nodes.size());
  for(int gid = 0; gid != nodes.size(); ++gid) {
    auto const& node = nodes[gid];
    ps.push_back(partition_t::singleton(node.op.shape()));
  }
  return ps;
}

vector<placement_t> graph_t::make_singleton_placement() const {
  vector<placement_t> pls;
  pls.reserve(nodes.size());
  for(auto const& part: make_singleton_partition()) {
    pls.emplace_back(part);
  }
  return pls;
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
graph_writer_t::contraction(
  string str,
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  auto maybe_info = make_einsummable_info(str, {lhs,rhs});
  if(!maybe_info) {
    throw std::runtime_error("graph_writer_t contraction: could not create einsummable");
  }

  auto const& info = maybe_info.value();

  einsummable_t e = info.build_einsummable(scalarop_t::make_mul(), castable_t::add);

  if(!e.is_contraction()) {
    throw std::runtime_error("build einsummable is not a contraction");
  }

  int id = graph.insert_einsummable(e, {lhs.id, rhs.id});
  id = graph.insert_formation(id, false);

  return tensor_t(
    info.get_out_shape(),
    info.get_out_full_shape(),
    id,
    *this);
}

graph_writer_t::tensor_t
graph_writer_t::reduction(
  string str,
  castable_t castable,
  graph_writer_t::tensor_t const& inn)
{
  auto maybe_info = make_einsummable_info(str, {inn});
  if(!maybe_info) {
    throw std::runtime_error("graph_writer_t reduction: could not create einsummable");
  }

  auto const& info = maybe_info.value();

  einsummable_t e = info.build_einsummable(scalarop_t::make_mul(), castable_t::add);

  if(!e.has_aggregation()) {
    throw std::runtime_error("build einsummable is not a reduction");
  }

  int id = graph.insert_einsummable(e, {inn.id});
  id = graph.insert_formation(id, false);

  return tensor_t(
    info.get_out_shape(),
    info.get_out_full_shape(),
    id,
    *this);
}

graph_writer_t::tensor_t
graph_writer_t::ew(
  string str,
  scalarop_t op,
  graph_writer_t::tensor_t const& inn)
{
  return ew(str, op, vector<tensor_t>{inn});
}

graph_writer_t::tensor_t
graph_writer_t::ew(
  string str,
  scalarop_t op,
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  return ew(str, op, vector<tensor_t>{lhs, rhs});
}

graph_writer_t::tensor_t
graph_writer_t::ew(
  string str,
  scalarop_t op,
  vector<graph_writer_t::tensor_t> const& inns)
{
  auto maybe_info = make_einsummable_info(str, inns);
  if(!maybe_info) {
    throw std::runtime_error("graph writer ew: coult not create einsummable");
  }

  auto const& info = maybe_info.value();
  einsummable_t e = info.build_einsummable(op);

  if(e.has_aggregation()) {
    throw std::runtime_error("ew op has aggregation");
  }

  vector<int> inn_ids = vector_from_each_member(inns, int, id);
  int id = graph.insert_einsummable(e, inn_ids);

  return tensor_t(
    info.get_out_shape(),
    info.get_out_full_shape(),
    id,
    *this);
}

graph_writer_t::tensor_t
graph_writer_t::concat(
  int dim,
  vector<graph_writer_t::tensor_t> const& inns_)
{
  vector<tensor_t> inns = inns_;

  // make sure all the inputs have no hidden permutation
  for(tensor_t& inn: inns) {
    inn.physically_permute();
  }

  // only the full dimensions can be concatenated
  for(tensor_t const& inn: inns) {
    auto [i,j] = inn.get_breaks()[dim];
    if(j-i != 1) {
      throw std::runtime_error(
        "graph writer concat: only full dims can be concated");
    }
  }

  // check the shapes (but not full shapes) are correct
  vector<uint64_t> shape;
  {
    vector<vector<uint64_t>> shapes = vector_from_each_member(
      inns, vector<uint64_t>, shape);
    optional<string> err_msg = check_concat_shapes(dim, shapes);
    if(err_msg) {
      throw std::runtime_error("graph writer insert concat: " + err_msg.value());
    }

    shape = shapes[0];
    shape[dim] = 0;
    for(int i = 0; i != inns.size(); ++i) {
      shape[dim] += shapes[i][dim];
    }
  }

  auto breaks = inns[0].get_breaks();
  int full_dim = std::get<0>(breaks[dim]);

  // now graph concat will check the full shapes
  vector<int> inn_ids = vector_from_each_member(inns, int, id);
  int id = graph.insert_concat(full_dim, inn_ids);

  auto full_shape = graph.out_shape(id);

  return tensor_t(
    shape,
    full_shape,
    id,
    *this);
}

graph_writer_t::tensor_t
graph_writer_t::to_real(
  graph_writer_t::tensor_t const& inn)
{
  if(!dtype_is_complex(inn.get_dtype())) {
    throw std::runtime_error("must have complex to convert to real");
  }
  return insert_complexer(inn);
}

graph_writer_t::tensor_t
graph_writer_t::to_complex(
  graph_writer_t::tensor_t const& inn)
{
  if(!dtype_is_real(inn.get_dtype())) {
    throw std::runtime_error("must have real to convert to complex");
  }
  return insert_complexer(inn);
}

graph_writer_t::tensor_t
graph_writer_t::insert_complexer(
  graph_writer_t::tensor_t tensor)
{
  bool to_real = dtype_is_complex(tensor.get_dtype());

  // Only do a permutation if the actual last dimension
  // is not at the end
  if(tensor.modes.back() != tensor.modes.size() - 1) {
    tensor.physically_permute();
  }

  if(to_real) {
    tensor.id = graph.insert_to_real(tensor.id);
    tensor.full_shape.back() *= 2;
    tensor.shape.back() *= 2;
  } else {
    tensor.id = graph.insert_to_complex(tensor.id);
    tensor.full_shape.back() /= 2;
    tensor.shape.back() /= 2;
  }

  return tensor;
}

graph_writer_t::tensor_t
graph_writer_t::matmul(
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  int nl = lhs.shape.size();
  int nr = rhs.shape.size();
  if(nl < 2 || nr < 2) {
    throw std::runtime_error("graph writer matmul: must have atleast rank 2");
  }

  auto make_header = [](int n) {
    string ret(n, ' ');
    std::iota(ret.begin(), ret.end(), 'd');
    return ret;
  };

  string sl = "ab";
  string sr = "bc";
  string so;

  if(nl == 2 && nr == 2) {
    so = "ac";
  } else if(nl == 2)  {
    int n = nr-2;
    string header = make_header(n);
    sr = header + "bc";
    so = header + "ac";
  } else if(nr == 2)  {
    int n = nl-2;
    string header = make_header(n);
    sl = header + "ab";
    so = header + "ac";
  } else if(nl == nr) {
    int n = nl-2;
    string header = make_header(n);
    sl = header + "ab";
    sr = header + "bc";
    so = header + "ac";
  } else {
    throw std::runtime_error("graph writer matmul: invalid inputs");
  }

  return contraction(sl + "," + sr + "->" + so, lhs, rhs);
}

graph_writer_t::tensor_t
graph_writer_t::softmax(
  graph_writer_t::tensor_t const& inn)
{
  int n = inn.shape.size() - 1;

  string h(n, ' ');
  std::iota(h.begin(), h.end(), 'b');
  string ha = h + "a";
  string redstr = ha + "->" + h;
  string ewustr = ha + "->" + ha;
  string ewbstr = ha + "," + h + "->" + ha;

  tensor_t x = inn;

  tensor_t c = reduction(
    redstr,
    castable_t::max,
    x);

  // x = x + c
  x = ew(
    ewbstr,
    scalarop_t::make_add(),
    x, c);

  // ex = exp(x)
  tensor_t ex = ew(
    ewustr,
    scalarop_t::make_exp(),
    x);

  tensor_t sum_ex = reduction(
    redstr,
    castable_t::add,
    x);

  return ew(
    ewbstr,
    scalarop_t::make_div(),
    ex, sum_ex);
}

graph_writer_t::tensor_t
graph_writer_t::add(
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  if(lhs.shape != rhs.shape) {
    throw std::runtime_error("graph writer add : invalid shapes");
  }

  string x(lhs.shape.size(), ' ');
  std::iota(x.begin(), x.end(), 'a');

  return ew(
    x + "," + x + "->" + x,
    scalarop_t::make_add(),
    lhs, rhs);
}

graph_writer_t::tensor_t
graph_writer_t::scale(
  float val,
  graph_writer_t::tensor_t const& inn)
{
  string x(inn.shape.size(), ' ');
  std::iota(x.begin(), x.end(), 'a');

  // TODO: will need to deal with dtypes
  return ew(x + "->" + x, scalarop_t::make_scale(scalar_t(val)), inn);
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

      // modes=42310 means permutation 01234->42310
      // currently, before_perm has           abcde
      // and need to get               edbca
      //
      for(int i = 0; i != modes.size(); ++i) {
        after_perm[modes[i]] = before_perm[i];
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

