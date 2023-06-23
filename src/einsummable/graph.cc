#include "graph.h"
#include "../base/hrect.h"

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

tuple<int, int> concat_t::get_inn_region(uint64_t beg, uint64_t end) const {
  auto offsets = get_offsets();
  uint64_t dim_nelem = offsets.back() + inn_shapes.back()[dim];
  if(beg >= end || end > dim_nelem) {
    throw std::runtime_error("concat_t::get_inns invalid inputs");
  }
  offsets.push_back(dim_nelem);

  int b = -1;
  int e = -1;
  for(int i = 0; i != offsets.size()-1; ++i) {
    auto const& inn_b = offsets[i  ];
    auto const& inn_e = offsets[i+1];
    if(beg >= inn_b && beg < inn_e) {
      b = i;
    }
    if(end > inn_b && end <= inn_e) {
      e = i;
    }
  }
  if(b == -1 || e == -1) {
    throw std::runtime_error("concat_t::get_inn_region");
  }
  return {b,e+1};
}

tuple<int, int> concat_t::get_inn_region(
  tuple<uint64_t,uint64_t> const& be) const
{
  auto const& [b,e] = be;
  return get_inn_region(b,e);
}

vector<subset_t::subsetdim_t>
subset_t::make_selection(
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  vector<uint64_t> inn_shape)
{
  int rank = hrect.size();
  if(inn_shape.size() != rank) {
    throw std::runtime_error("subset_t::make_selection");
  }

  vector<subsetdim_t> ret;
  ret.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto const& d_inn = inn_shape[i];
    auto const& [b,e] = hrect[i];
    if(e <= b) {
      throw std::runtime_error("must have positive hrect");
    }
    uint64_t d_out = e-b;
    if(d_inn < d_out) {
      throw std::runtime_error("can't subset to bigger");
    }

    ret.push_back(subsetdim_t {
      .d_inn = d_inn,
      .d_out = d_out,
      .offset = b
    });
  }

  return ret;
}

subset_t::subset_t(
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  vector<uint64_t> inn_shape,
  set<int> squeeze,
  dtype_t dtype)
  : subset_t(make_selection(hrect, inn_shape), squeeze, dtype)
{}

subset_t::subset_t(
  vector<subsetdim_t> se,
  set<int> sq,
  dtype_t dt)
  : dtype(dt), selection(se), squeeze(sq)
{
  for(auto const& [d_inn,d_out,offset]: selection) {
    if(d_out + offset > d_inn) {
      throw std::runtime_error("subset construction: invalid selection");
    }
  }
  int rank = selection.size();
  for(auto const& s: squeeze) {
    if(s < 0 || s >= rank) {
      throw std::runtime_error("can only sqeeze modes in [0,rank)");
    }
    if(selection[s].d_out != 1) {
      throw std::runtime_error("can only squeeze modes with size 1");
    }
  }

  if(squeeze.size() == selection.size()) {
    throw std::runtime_error("selection must end up with more than zero dims");
  }

  bool is_no_op = true;
  for(auto const& [d_inn, d_out, offset]: selection) {
    if(d_inn != d_out || offset != 0) {
      is_no_op = false;
      break;
    }
  }
  if(is_no_op) {
    throw std::runtime_error("subset_t cannot be no op");
  }
}

vector<uint64_t> subset_t::inn_shape() const {
  return vector_from_each_member(selection, uint64_t, d_inn);
}

vector<uint64_t> subset_t::out_shape() const {
  int rank = selection.size();
  vector<uint64_t> shape;
  shape.reserve(rank - squeeze.size());
  for(int i = 0; i != rank; ++i) {
    if(squeeze.count(i) > 0) {
      // don't add this rank
    } else {
      shape.push_back(selection[i].d_out);
    }
  }
  return shape;
}

vector<tuple<uint64_t, uint64_t>>
subset_t::get_hrect() const {
  vector<tuple<uint64_t, uint64_t>> ret;
  int rank = selection.size();
  ret.reserve(rank);
  for(auto const& [d_inn, d_out, offset]: selection) {
    ret.emplace_back(offset, offset+d_out);
  }
  return ret;
}

touch_t subset_t::as_touch() const {
  vector<touchdim_t> tds;
  tds.reserve(selection.size());
  for(auto const& [d_inn, d_out, offset]: selection) {
    tds.push_back(touchdim_t {
      .d_inn      = d_inn,
      .d_out      = d_out,
      .offset_inn = offset,
      .offset_out = 0,
      .size       = d_out
    });
  }
  return touch_t {
    .selection = tds,
    .castable  = std::nullopt,
    .dtype     = dtype
  };
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
  auto expected_inn_dtypes = e.inn_dtypes();
  for(int i = 0; i != inns.size(); ++i) {
    if(!vector_equal(expected_inn_shapes[i], out_shape(inns[i]))) {
      throw std::runtime_error("shapes do not match: insert einsummable");
    }
    if(expected_inn_dtypes[i] != out_dtype(inns[i])) {
      throw std::runtime_error("dtype error in graph insert einsumable");
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

int graph_constructor_t::insert_subset(
  placement_t placement,
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn,
  set<int> squeeze)
{
  int ret = graph.insert_subset(hrect, inn, squeeze);

  if(placement.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid concat");
  }

  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_subset(
  partition_t partition,
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn,
  set<int> squeeze)
{
  int ret = graph.insert_subset(hrect, inn, squeeze);

  if(partition.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid concat");
  }

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_constructor_t::insert_subset(
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn,
  set<int> squeeze)
{
  int ret = graph.insert_subset(hrect, inn, squeeze);

  partition_t partition = partition_t::singleton(graph.out_shape(ret));

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_t::insert_subset(
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn,
  set<int> squeeze)
{
  dtype_t dtype = out_dtype(inn);
  vector<uint64_t> inn_shape = out_shape(inn);
  subset_t subset(hrect, inn_shape, squeeze, dtype);

  return this->insert(subset_t(hrect, inn_shape, squeeze, dtype), {inn});
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
  if(is_subset()) {
    return get_subset().dtype;
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
  if(is_subset()) {
    return get_subset().out_shape();
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
  if(is_subset()) {
    return get_subset().out_shape();
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

    std::cout << "node id: " << id
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
    } else if(node.op.is_subset()) {
      std::cout << "subset" << std::endl;
    } else {
      throw std::runtime_error("graph_t print should not reach");
    }

    std::cout << std::endl;
  }
}

void graph_t::print_graphviz(std::ostream& out) const {
  using std::endl;
  string tab = "  ";
  out << "digraph {" << endl;

  for(int id = 0; id != nodes.size(); ++id) {
    node_t const& node = nodes[id];
    op_t const& op = node.op;

    string label;
    string color = "";
    if(op.is_input()) {
      label = "input" + write_with_ss(id);
    } else if(op.is_formation()) {
      label = "form" + write_with_ss(id);
    } else if(op.is_complexer()) {
      label = "complexer" + write_with_ss(id);
    } else if(op.is_einsummable()) {
      label = "einsummable" + write_with_ss(id);
    } else if(op.is_concat()) {
      label = "concat" + write_with_ss(id);
    } else if(op.is_subset()) {
      label = "subset" + write_with_ss(id);
    } else {
      throw std::runtime_error("printgraphviz missing graph node type");
    }
    out << tab
      << "n" << id
      << " [style=filled,label=\"" << label << "\"";
    if(color != "") {
      out << ",color=\"" << color << "\"";
    }
    out << "]" << endl;

    for(auto const& inn: node.get_inns_set()) {
      out << tab << "n" << inn << " -> " << "n" << id << endl;
    }
  }
  out << "}" << endl;
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
    vtensor_t<int> locs(shape);

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

    vtensor_t<int> locs({pi,pk,pj});

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

tuple<uint64_t, uint64_t>
graph_writer_t::idx_t::get(uint64_t d) const {
  if(std::holds_alternative<rng>(op)) {
    auto const& [beg,end] = std::get<rng>(op);
    return {to_index(d, beg), to_index(d, end)};
  } else if(std::holds_alternative<idx>(op)) {
    int64_t const& i = std::get<idx>(op).v;
    auto b = to_index(d, i);
    return {b, b+1};
  } else if(std::holds_alternative<all>(op)) {
    return {0, d};
  } else {
    throw std::runtime_error("should not happen: idx_t get in graph writer");
  }
}

uint64_t graph_writer_t::idx_t::to_index(uint64_t total, int64_t held) {
  if(held < 0) {
    held = std::abs(held);
    if(total < held) {
      throw std::runtime_error("too large negative value in to index");
    }
    return total - held;
  } else {
    if(held > total) {
      throw std::runtime_error("too large value in index");
    }
    return held;
  }
}

graph_writer_t::full_dim_t
graph_writer_t::full_dim_t::singleton(uint64_t d) {
  return full_dim_t({d});
}

vector<uint64_t>
graph_writer_t::full_shape_t::full() const {
  return vector_flatten(
    vector_from_each_member(parts, vector<uint64_t>, parts)
  );
}

vector<uint64_t>
graph_writer_t::full_shape_t::operator()() const {
  return vector_from_each_method(parts, uint64_t, operator());
}

vector<tuple<int,int>>
graph_writer_t::full_shape_t::get_breaks() const
{
  int b = 0;
  vector<tuple<int,int>> ret;
  for(auto const& p: parts) {
    int d = p.parts.size();
    ret.emplace_back(b, b + d);
    b = b + d;
  }
  return ret;
}

vector<vector<uint64_t>>
graph_writer_t::full_shape_t::as_vecvec() const
{
  vector<vector<uint64_t>> ret;
  ret.reserve(parts.size());
  for(auto const& p: parts) {
    ret.push_back(p.parts);
  }
  return ret;
}

graph_writer_t::full_shape_t
graph_writer_t::full_shape_t::from_full(
  vector<uint64_t> const& ss)
{
  vector<full_dim_t> parts;
  parts.reserve(ss.size());
  for(auto const& d: ss) {
    parts.push_back(full_dim_t::singleton(d));
  }
  return full_shape_t(parts);
}

graph_writer_t::full_shape_t
graph_writer_t::full_shape_t::from_vecvec(
  vector<vector<uint64_t>> const& ss)
{
  vector<full_dim_t> parts;
  parts.reserve(ss.size());

  for(auto const& s: ss) {
    parts.push_back(full_dim_t { s });
  }

  return full_shape_t(parts);
}

graph_writer_t::tensor_t::tensor_t(
  full_shape_t const& _shape,
  int _id,
  graph_writer_t* _self)
    : shape(_shape), id(_id), self(_self)
{
  modes.resize(shape.full_rank());
  std::iota(modes.begin(), modes.end(), 0);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::transpose(int i, int j) const
{
  auto breaks = shape.get_breaks();

  tensor_t ret = *this;

  std::swap(ret.shape.parts.at(i), ret.shape.parts.at(j));

  rotate_sections(ret.modes, breaks[i], breaks[j]);

  return ret;
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::view(
  graph_writer_t::full_shape_t const& new_shape) const
{
  if(!vector_equal(shape.full(), new_shape.full())) {
    throw std::runtime_error("incorrect tensor_t view argument");
  }

  tensor_t ret = *this;
  ret.shape = new_shape;
  return ret;
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::view(
  vector<vector<uint64_t>> const& new_shape) const
{
  return view(full_shape_t::from_vecvec(new_shape));
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::view_full() const
{
  return view(full_shape_t::from_full(shape.full()));
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::view_full(
  vector<uint64_t> const& full_shape) const
{
  if(!vector_equal(full_shape, shape.full())) {
    throw std::runtime_error("in view full: incorrect full shape passed in");
  }
  return view(full_shape_t::from_full(full_shape));
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::save() const
{
  // this will permute if necc
  tensor_t ret = physically_permute();

  {
    auto& op = self->graph.nodes[ret.id].op;
    if(op.is_formation()) {
      op.get_formation().is_save = true;
      return ret;
    }
  }

  ret.id = self->graph.insert_formation(ret.id, true);

  return ret;
}

dtype_t graph_writer_t::tensor_t::get_dtype() const {
  return self->graph.out_dtype(id);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::scale(scalar_t const& val) const {
  return self->scale(val, *this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::scale(string const& val) const {
  return self->scale(val, *this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_complex() const {
  return self->to_complex(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_real() const {
  return self->to_real(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_dtype(dtype_t d) const {
  return self->to_dtype(d, *this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_f16() const {
  return self->to_f16(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_f32() const {
  return self->to_f32(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::to_f64() const {
  return self->to_f64(*this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::subset(
  vector<graph_writer_t::idx_t> const& idxs) const
{
  auto _shape = shape();
  int rank = _shape.size();
  if(idxs.size() != rank) {
    throw std::runtime_error("tensor subset: invalid rank idxs");
  }
  vector<tuple<uint64_t, uint64_t>> hrect;
  hrect.reserve(rank);
  set<int> squeeze;
  for(int i = 0; i != rank; ++i) {
    auto const& idx = idxs[i];
    hrect.push_back(idx.get(_shape[i]));
    if(idx.is_squeeze()) {
      squeeze.insert(i);
    }
  }
  return self->subset(hrect, squeeze, *this);
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::physically_permute() const {
  tensor_t ret = *this;

  vector<int> no_permute_modes(ret.modes.size());
  std::iota(
    no_permute_modes.begin(),
    no_permute_modes.end(),
    0);

  if(ret.modes == no_permute_modes) {
    return ret;
  }

  string str;
  {
    vector<char> letters(ret.modes.size());
    std::iota(letters.begin(), letters.end(), 'a');

    string inn(letters.begin(), letters.end());

    string out;
    for(auto const& m: ret.modes) {
      out.push_back(letters[m]);
    }

    str = inn + "->" + out;
  }

  dtype_t dtype = ret.get_dtype();
  ret.id = self->_insert_elementwise(
    str,
    scalarop_t::make_identity(dtype),
    ret.id);
  ret.modes = no_permute_modes;

  return ret;
}

graph_writer_t::tensor_t
graph_writer_t::input(
  vector<uint64_t> shape,
  dtype_t dtype)
{
  return this->input(full_shape_t::from_full(shape), dtype);
}

graph_writer_t::tensor_t
graph_writer_t::input(
  full_shape_t shape,
  dtype_t dtype)
{
  int id = graph.insert_input(shape.full(), dtype);
  return tensor_t(shape, id, this);
}

graph_writer_t::tensor_t
graph_writer_t::input(
  vector<vector<uint64_t>> const& shape,
  dtype_t dtype)
{
  return this->input(full_shape_t::from_vecvec(shape), dtype);
}

graph_writer_t::full_shape_t
graph_writer_t::to_einsummable_info_t::get_out_shape() const
{
  return full_shape_t(vector<full_dim_t>(
      join_shape.parts.begin(),
      join_shape.parts.begin() + out_rank));
}

einsummable_t
graph_writer_t::to_einsummable_info_t::build_einsummable(
  scalarop_t join,
  optional<castable_t> castable) const
{
  return einsummable_t(
    join_shape.full(),
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

  if(lhs.get_dtype() != rhs.get_dtype()) {
    throw std::runtime_error("must contraction with same input dtypes");
  }

  einsummable_t e = info.build_einsummable(
    scalarop_t::make_mul(lhs.get_dtype()),
    castable_t::add);

  if(!e.is_contraction()) {
    throw std::runtime_error("build einsummable is not a contraction");
  }

  int id = graph.insert_einsummable(e, {lhs.id, rhs.id});
  id = graph.insert_formation(id, false);

  return tensor_t(
    info.get_out_shape(),
    id,
    this);
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

  einsummable_t e = info.build_einsummable(
    scalarop_t::make_identity(inn.get_dtype()),
    castable_t::add);

  if(!e.has_aggregation()) {
    throw std::runtime_error("build einsummable is not a reduction");
  }

  int id = graph.insert_einsummable(e, {inn.id});
  id = graph.insert_formation(id, false);

  return tensor_t(
    info.get_out_shape(),
    id,
    this);
}

graph_writer_t::tensor_t
graph_writer_t::ew(
  scalarop_t op,
  graph_writer_t::tensor_t const& inn)
{
  int out_rank = inn.rank();

  string ijk(out_rank, ' ');
  std::iota(ijk.begin(), ijk.end(), 'a');

  string str = ijk + "->" + ijk;

  return ew(str, op, vector<tensor_t>{inn});
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
    id,
    this);
}

graph_writer_t::tensor_t
graph_writer_t::concat(
  int dim,
  vector<graph_writer_t::tensor_t> const& inns_)
{
  if(inns_.size() == 0) {
    throw std::runtime_error("graph writer concat empty input");
  }

  vector<tensor_t> inns = inns_;

  // make sure all the inputs have no hidden permutation
  for(tensor_t& inn: inns) {
    inn = inn.physically_permute();
  }

  // only the full dimensions can be concatenated
  for(tensor_t const& inn: inns) {
    auto [i,j] = inn.get_shape().get_breaks()[dim];
    if(j-i != 1) {
      throw std::runtime_error(
        "graph writer concat: only full dims can be concated");
    }
  }

  // check the shapes (but not full shapes) are correct
  full_shape_t shape = inns[0].shape;
  {
    vector<vector<uint64_t>> shapes = vector_from_each_method(
      inns, vector<uint64_t>, get_shape().operator());
    optional<string> err_msg = check_concat_shapes(dim, shapes);
    if(err_msg) {
      throw std::runtime_error("graph writer insert concat: " + err_msg.value());
    }

    uint64_t d_after_concat = 0;
    for(auto const& shape: shapes) {
      d_after_concat += shape[dim];
    }
    shape.parts[dim] = full_dim_t::singleton(d_after_concat);
  }

  auto breaks = shape.get_breaks();
  int full_dim = std::get<0>(breaks[dim]);

  // now graph concat will check the full shapes
  vector<int> inn_ids = vector_from_each_member(inns, int, id);
  int id = graph.insert_concat(full_dim, inn_ids);

  auto full_shape = graph.out_shape(id);
  if(!vector_equal(full_shape, shape.full())) {
    throw std::runtime_error("graph writer concat implementation error");
  }

  return tensor_t(
    shape,
    id,
    this);
}

graph_writer_t::tensor_t
graph_writer_t::subset(
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  set<int> squeeze,
  tensor_t const& inn)
{
  full_shape_t inn_shape = inn.get_shape();
  vector<uint64_t> inn_shape_ = inn_shape();
  int rank = inn_shape_.size();

  if(hrect.size() != rank) {
    throw std::runtime_error("graph writer subset: incorrect len hrect");
  }

  {
    bool is_no_op = true;
    for(int i = 0; i != rank; ++i) {
      auto [b,e] = hrect[i];
      if(b != 0 || e != inn_shape_[i]) {
        is_no_op = false;
        break;
      }
    }
    if(is_no_op) {
      return inn;
    }
  }

  // make sure that only singleton full_dim_t's are being
  // subset.

  auto is_subset_dim = [&](int i) {
    auto const& [b,e] = hrect[i];
    auto const& d = inn_shape_[i];
    return (b != 0 || e != d);
  };

  auto breaks = inn_shape.get_breaks();
  for(int i = 0; i != rank; ++i) {
    auto const& [dim_idx_b,dim_idx_e] = breaks[i];
    if(dim_idx_b + 1 != dim_idx_e) {
      // dim part i is not a singleton
      if(is_subset_dim(i)) {
        throw std::runtime_error(
          "only subsetting singleton dimensions are supported");
      }
    }
  }

  vector<uint64_t> inn_full_shape = inn_shape.full();

  vector<tuple<uint64_t, uint64_t>> full_hrect;
  full_hrect.reserve(inn_full_shape.size());

  set<int> full_squeeze;

  vector<vector<uint64_t>> out_vecvec;

  for(int i = 0; i != rank; ++i) {
    auto const& [b,e] = breaks[i];

    if(is_subset_dim(i)) {
      full_hrect.push_back(hrect[i]);
      if(squeeze.count(i) == 0) {
        auto const& [_b,_e] = hrect[i];
        out_vecvec.push_back({ _e-_b});
      } else {
        full_squeeze.insert(b);
      }
    } else {
      if(squeeze.count(i) > 0) {
        throw std::runtime_error("subset graph writer error");
      }
      out_vecvec.emplace_back();
      for(int f = b; f != e; ++f) {
        uint64_t const& d = inn_full_shape[f];
        full_hrect.emplace_back(0, d);
        out_vecvec.back().push_back(d);
      }
    }
  }

  int out_id = graph.insert_subset(full_hrect, inn.get_id(), full_squeeze);

  full_shape_t out_shape = full_shape_t::from_vecvec(out_vecvec);

  if(!vector_equal(graph.out_shape(out_id), out_shape.full())) {
    throw std::runtime_error("impl error: graph writer subset");
  }

  return tensor_t(out_shape, out_id, this);
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
graph_writer_t::to_dtype(dtype_t dtype, tensor_t const& inn) {
  if(dtype == dtype_t::c64) {
    throw std::runtime_error(
      "cannot call dtype with c64; maybe call to_real instead");
  }

  dtype_t inn_dtype = inn.get_dtype();

  if(dtype == inn_dtype) {
    return inn;
  }

  scalarop_t f = scalarop_t::make_convert_dtype(inn_dtype, dtype);

  return ew(f, inn);
}

graph_writer_t::tensor_t graph_writer_t::to_f16(tensor_t const& inn) {
  return to_dtype(dtype_t::f16, inn);
}

graph_writer_t::tensor_t graph_writer_t::to_f32(tensor_t const& inn) {
  return to_dtype(dtype_t::f32, inn);
}

graph_writer_t::tensor_t graph_writer_t::to_f64(tensor_t const& inn) {
  return to_dtype(dtype_t::f64, inn);
}

graph_writer_t::tensor_t
graph_writer_t::insert_complexer(
  graph_writer_t::tensor_t tensor)
{
  bool to_real = dtype_is_complex(tensor.get_dtype());

  // Only do a permutation if the actual last dimension
  // is not at the end
  if(tensor.modes.back() != tensor.modes.size() - 1) {
    tensor = tensor.physically_permute();
  }

  // get a reference to the last dim part
  auto& full_shape_parts = tensor.shape.parts;
  auto& last_full_dim = full_shape_parts.back();
  auto& last_dim_part = last_full_dim.parts.back();

  if(to_real) {
    tensor.id = graph.insert_to_real(tensor.id);
    last_dim_part *= 2;
  } else {
    tensor.id = graph.insert_to_complex(tensor.id);
    last_dim_part /= 2;
  }

  return tensor;
}

graph_writer_t::tensor_t
graph_writer_t::matmul(
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  int nl = lhs.rank();
  int nr = rhs.rank();
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
  dtype_t dtype = inn.get_dtype();

  int n = inn.rank() - 1;

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
    scalarop_t::make_add(dtype),
    x, c);

  // ex = exp(x)
  tensor_t ex = ew(
    ewustr,
    scalarop_t::make_exp(dtype),
    x);

  tensor_t sum_ex = reduction(
    redstr,
    castable_t::add,
    x);

  return ew(
    ewbstr,
    scalarop_t::make_div(dtype),
    ex, sum_ex);
}

graph_writer_t::tensor_t
graph_writer_t::add(
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  return straight_bew(
    scalarop_t::make_add(lhs.get_dtype()),
    lhs,
    rhs);
}

graph_writer_t::tensor_t
graph_writer_t::mul(
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  return straight_bew(
    scalarop_t::make_mul(lhs.get_dtype()),
    lhs,
    rhs);
}

graph_writer_t::tensor_t
graph_writer_t::straight_bew(
  scalarop_t op,
  graph_writer_t::tensor_t const& lhs,
  graph_writer_t::tensor_t const& rhs)
{
  if(lhs.shape != rhs.shape) {
    throw std::runtime_error("graph writer add : invalid shapes");
  }

  dtype_t dtype = lhs.get_dtype();
  if(dtype != rhs.get_dtype()) {
    throw std::runtime_error("cannot add with different dtypes");
  }

  string x(lhs.rank(), ' ');
  std::iota(x.begin(), x.end(), 'a');

  return ew(
    x + "," + x + "->" + x,
    op,
    lhs, rhs);
}

graph_writer_t::tensor_t
graph_writer_t::scale(
  string val,
  graph_writer_t::tensor_t const& inn)
{
  dtype_t dtype = inn.get_dtype();
  return scale(scalar_t(dtype, val), inn);
}

graph_writer_t::tensor_t
graph_writer_t::scale(
  scalar_t val,
  graph_writer_t::tensor_t const& inn)
{
  string x(inn.rank(), ' ');
  std::iota(x.begin(), x.end(), 'a');

  return ew(x + "->" + x, scalarop_t::make_scale(val), inn);
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
      inn_shapes.push_back(inn_tensor.shape());
    }
    auto maybe = einsummable_t::construct_join_shape(inns, inn_shapes);
    if(!maybe) {
      return std::nullopt;
    }
    join_shape = maybe.value();
  }

  full_shape_t full_join_shape;
  int full_out_rank;
  {
    vector<vector<vector<uint64_t>>> fs;
    for(auto const& inn_tensor: tensors) {
      fs.push_back(inn_tensor.shape.as_vecvec());
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

    vector<full_dim_t> ds;
    ds.reserve(v.size());
    for(auto const& dim_parts: v) {
      ds.emplace_back(dim_parts);
    }
    full_join_shape = full_shape_t(ds);
  }

  vector<vector<int>> full_inns_before_perm;
  {
    auto breaks = full_join_shape.get_breaks();
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
    .join_shape = full_join_shape,
    .full_inns = full_inns,
    .full_out_rank = full_out_rank,
    .out_rank = out_rank
  };
}

bool operator==(
  graph_writer_t::full_dim_t const& lhs,
  graph_writer_t::full_dim_t const& rhs)
{
  return vector_equal(lhs.parts, rhs.parts);
}
bool operator!=(
  graph_writer_t::full_dim_t const& lhs,
  graph_writer_t::full_dim_t const& rhs)
{
  return !(lhs == rhs);
}

bool operator==(
  graph_writer_t::full_shape_t const& lhs,
  graph_writer_t::full_shape_t const& rhs)
{
  return vector_equal(lhs.parts, rhs.parts);
}
bool operator!=(
  graph_writer_t::full_shape_t const& lhs,
  graph_writer_t::full_shape_t const& rhs)
{
  return !(lhs == rhs);
}

std::ostream& operator<<(
  std::ostream& out,
  graph_writer_t::full_shape_t const& shape)
{
  out << shape();
  return out;
}
