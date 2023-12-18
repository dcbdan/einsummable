#include "graph.h"
#include "../base/hrect.h"

select_t::select_t(
  dtype_t dtype,
  vector<uint64_t> const& os,
  vector<select_t::inn_region_t> const& irs)
  : dtype(dtype), out_shape(os), inn_regions(irs)
{
  int rank = out_shape.size();
  if(rank == 0) {
    throw std::runtime_error("cannot have empty out shape in select");
  }
  for(auto const& d: out_shape) {
    if(d == 0) {
      throw std::runtime_error("cannot have dimension of size zero");
    }
  }

  vector<vector<tuple<uint64_t, uint64_t>>> hrects;
  hrects.reserve(inn_regions.size());
  for(auto const& inn_region: inn_regions) {
    if(inn_region.size() != rank) {
      throw std::runtime_error("inn region must have same rank as output");
    }

    hrects.emplace_back();
    auto& hrect = hrects.back();
    hrect.reserve(rank);

    for(int i = 0; i != rank; ++i) {
      auto const& d_out = out_shape[i];
      auto const& [d_inn, offset_inn, offset_out, size] = inn_region[i];

      if(size + offset_out > d_out ||
         size + offset_inn > d_inn ||
         d_inn == 0 || size == 0)
      {
        throw std::runtime_error("invalid select given");
      }

      hrect.emplace_back(offset_out, offset_out + size);
    }
  }

  if(!partitions_region(hrects, out_shape)) {
    throw std::runtime_error("This select does not partition the write region");
  }
}

vector<touch_t> select_t::as_touches() const {
  vector<touch_t> ret;
  ret.reserve(inn_regions.size());
  for(int i = 0; i != inn_regions.size(); ++i) {
    ret.push_back(as_touch(i));
  }
  return ret;
}

touch_t select_t::as_touch(int which) const {
  vector<touchdim_t> tds;
  auto const& sds = inn_regions.at(which);
  for(int i = 0; i != sds.size(); ++i) {
    uint64_t d_out = out_shape[i];
    auto const& sd = sds[i];
    tds.push_back(touchdim_t {
      .d_inn      = sd.d_inn,
      .d_out      = d_out,
      .offset_inn = sd.offset_inn,
      .offset_out = sd.offset_out,
      .size       = sd.size
    });
  }

  return touch_t {
    .selection = tds,
    .castable = std::nullopt,
    .dtype = dtype
  };
}

vector<uint64_t>
select_t::wrt_output_point(
  vector<uint64_t> const& inn_point,
  int which_inn) const
{
  auto const& inn_region = inn_regions.at(which_inn);
  int rank = inn_region.size();

  vector<uint64_t> out_point;
  out_point.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto const& sd = inn_region[i];
    uint64_t const& b_inn = inn_point[i];
    out_point.push_back(
      sd.offset_out + (b_inn - sd.offset_inn)
    );
  }

  return out_point;
}

hrect_t select_t::wrt_output_hrect(hrect_t const& inn_hrect, int which_inn) const
{
  auto ret_b = wrt_output_point(vector_mapfst(inn_hrect), which_inn);
  auto ret_e = wrt_output_point(vector_mapsnd(inn_hrect), which_inn);
  return vector_zip(ret_b, ret_e);
}

hrect_t select_t::wrt_output_inn_hrect(int which_input) const
{
  hrect_t ret;
  for(auto const& sd: inn_regions.at(which_input)) {
    ret.emplace_back(sd.offset_out, sd.offset_out + sd.size);
  }
  return ret;
}

hrect_t select_t::wrt_input_inn_hrect(int which_input) const
{
  hrect_t ret;
  for(auto const& sd: inn_regions.at(which_input)) {
    ret.emplace_back(sd.offset_inn, sd.offset_inn + sd.size);
  }
  return ret;
}

vector<tuple<hrect_t, int>>
select_t::collect(hrect_t out_hrect) const
{
  // for each input, does it intersect, if so, where?
  vector<tuple<hrect_t, int>> ret;
  int rank = out_shape.size();
  for(int which_inn = 0; which_inn != inn_regions.size(); ++which_inn) {
    auto const& inn_region = inn_regions[which_inn];
    vector<tuple<uint64_t, uint64_t>> inn_hrect;
    for(int i = 0; i != rank; ++i) {
      selectdim_t const& sd = inn_region[i];
      auto const& [out_blk_b, out_blk_e] = out_hrect[i];
      uint64_t const& here_b = sd.offset_out;
      uint64_t        here_e = sd.offset_out + sd.size;
      uint64_t out_b = std::max(out_blk_b, here_b);
      uint64_t out_e = std::min(out_blk_e, here_e);
      if(out_b < out_e) {
        inn_hrect.emplace_back(
          sd.offset_inn + (out_b - sd.offset_out),
          sd.offset_inn + (out_e - sd.offset_out));
      } else {
        break;
      }
    }
    if(inn_hrect.size() == rank) {
      ret.emplace_back(inn_hrect, which_inn);
    }
  }

  return ret;
}

vector<uint64_t>
select_t::inn_shape(int which_inn) const
{
  return vector_from_each_member(inn_regions[which_inn], uint64_t, d_inn);
}

select_t make_concat(
  int dim,
  dtype_t dtype,
  vector<vector<uint64_t>> const& input_shapes)
{
  using selectdim_t = select_t::selectdim_t;

  if(input_shapes.size() == 0) {
    throw std::runtime_error("cannot concat empty list");
  }

  uint64_t offset_dim = 0;

  vector<vector<selectdim_t>> sdss;
  sdss.reserve(input_shapes.size());

  for(auto const& inn_shape: input_shapes) {
    sdss.emplace_back();
    auto& sds = sdss.back();
    for(int i = 0; i != inn_shape.size(); ++i) {
      uint64_t const& d_inn = inn_shape[i];
      if(i == dim) {
        sds.push_back(selectdim_t {
          .d_inn      = d_inn,
          .offset_inn = 0,
          .offset_out = offset_dim,
          .size       = d_inn
        });
        offset_dim += d_inn;
      } else {
        sds.push_back(selectdim_t {
          .d_inn      = d_inn,
          .offset_inn = 0,
          .offset_out = 0,
          .size       = d_inn
        });
      }
    }
  }
  vector<uint64_t> out_shape = vector_from_each_member(sdss[0], uint64_t, size);
  out_shape[dim] = offset_dim;

  return select_t(dtype, out_shape, sdss);
}

select_t make_subset(
  dtype_t dtype,
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  vector<uint64_t> inn_shape)
{
  using selectdim_t = select_t::selectdim_t;

  int rank = hrect.size();
  if(inn_shape.size() != rank || rank == 0) {
    throw std::runtime_error("invalid input to make_subset");
  }

  vector<selectdim_t> sds;
  vector<uint64_t> out_shape;
  sds.reserve(rank);
  out_shape.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto const& [beg,end] = hrect[i];
    uint64_t size = end-beg;
    sds.push_back(selectdim_t {
      .d_inn = inn_shape[i],
      .offset_inn = beg,
      .offset_out = 0,
      .size = size
    });
    out_shape.push_back(size);
  }

  return select_t(dtype, out_shape, {sds});
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
    op_t(formation_t {
      .dtype = out_dtype(inn),
      .shape = out_shape(inn) },
      is_save),
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

int graph_t::insert_fill(fill_t const& fill)
{
  if(fill.shape.size() == 0) {
    throw std::runtime_error("invalid fill");
  }
  for(auto const& dim: fill.shape) {
    if(dim == 0) {
      throw std::runtime_error("invalid dim in fill");
    }
  }

  return this->insert(fill, {});
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

  return this->insert(make_concat(dim, dtype, shapes), inns);
}

int graph_constructor_t::insert_subset(
  placement_t placement,
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn)
{
  int ret = graph.insert_subset(hrect, inn);

  if(placement.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid subset");
  }

  placements.insert({ret, placement});
  return ret;
}

int graph_constructor_t::insert_subset(
  partition_t partition,
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn)
{
  int ret = graph.insert_subset(hrect, inn);

  if(partition.total_shape() != graph.out_shape(ret)) {
    throw std::runtime_error("graph constructor: invalid subset");
  }

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_constructor_t::insert_subset(
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn)
{
  int ret = graph.insert_subset(hrect, inn);

  partition_t partition = partition_t::singleton(graph.out_shape(ret));

  placements.insert({ret, placement_t(partition)});
  return ret;
}

int graph_t::insert_subset(
  vector<tuple<uint64_t, uint64_t>> hrect,
  int inn)
{
  dtype_t dtype = out_dtype(inn);
  vector<uint64_t> inn_shape = out_shape(inn);

  return this->insert(make_subset(dtype, hrect, inn_shape), {inn});
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

graph_t::op_t::op_t(graph_t::op_t::_op_t op_, bool s)
  : op(op_), is_save_(s)
{
  if(has_aggregation() && is_save()) {
    throw std::runtime_error("an einsummable with an aggregation cannot be saved");
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
  if(is_fill()) {
    return get_fill().value.dtype;
  }
  if(is_select()) {
    return get_select().dtype;
  }
  if(is_einsummable()) {
    return get_einsummable().out_dtype();
  }
  throw std::runtime_error("graph::op_t should not reach");
}

void graph_t::op_t::set_save(bool s) {
  if(has_aggregation() && s) {
    throw std::runtime_error("set_save: "
      "an einsummable with an aggregation cannot be saved");
  }
  is_save_ = s;
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
  if(is_fill()) {
    return get_fill().shape;
  }
  if(is_select()) {
    return get_select().out_shape;
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
  if(is_fill()) {
    return get_fill().shape;
  }
  if(is_select()) {
    return get_select().out_shape;
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
      if(n.op.has_aggregation()) {
        this->insert_formation(i, true);
      } else {
        n.op.set_save(true);
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

vector<int> graph_t::get_reverse_order() const {
  // Can't tell if this is just the reverse of get_order() or not,
  // so wrote the full algorithm
  vector<int> ret;
  // reserve to not invalidate iterators
  ret.reserve(nodes.size());

  vector<int> deps;
  deps.reserve(nodes.size());
  for(int gid = 0; gid != nodes.size(); ++gid) {
    auto const& node = nodes[gid];
    int ndep = node.outs.size();
    if(ndep == 0) {
      ret.push_back(gid);
    }
    deps[gid] = ndep;
  }

  for(auto iter = ret.begin(); iter != ret.end(); ++iter) {
    int gid = *iter;
    auto const& node = nodes[gid];
    set<int> inns(node.inns.begin(), node.inns.end());
    for(auto const& out_gid: inns) {
      int& cnt = deps[out_gid];
      cnt--;
      if(cnt == 0) {
        ret.push_back(out_gid);
      }
    }
  }

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
    } else if(node.op.is_fill()) {
      std::cout << "fill[" << node.op.get_fill().value << "]" << std::endl;
    } else if(node.op.is_select()) {
      std::cout << "select" << std::endl;
    } else {
      throw std::runtime_error("graph_t print should not reach");
    }

    std::cout << std::endl;
  }
}

void graph_t::print_graphviz(
  std::ostream& out,
  map<int, string> get_color) const
{
  print_graphviz(out, make_singleton_partition(), get_color);
}

void graph_t::print_graphviz(
  std::ostream& out,
  vector<partition_t> const& parts,
  map<int, string> get_color) const
{
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
      label += "\n" + write_with_ss(op.shape());
    } else if(op.is_formation()) {
      label = "form" + write_with_ss(id);
      label += "\n" + write_with_ss(op.out_shape());
    } else if(op.is_complexer()) {
      label = "complexer" + write_with_ss(id);
    } else if(op.is_einsummable()) {
      auto const& e = op.get_einsummable();
      label = "einsummable" + write_with_ss(id) +
        ":" + e.str();
      if(e.is_contraction()) {
        color = "pink";
      }
      label += "\n" + e.join.to_cppstr() + "  |  " + write_with_ss(e.castable);
    } else if(op.is_fill()) {
      label = "fill" + write_with_ss(id) + ":" + write_with_ss(op.get_fill().value);
    } else if(op.is_select()) {
      label = "select" + write_with_ss(id);
    } else {
      throw std::runtime_error("printgraphviz missing graph node type");
    }
    //label += "\n" + write_with_ss(parts[id].block_shape());
    label += "\n" + write_with_ss(parts[id].total_shape()) + ":" + write_with_ss(out_dtype(id));
    out << tab
      << "n" << id
      << " [style=filled,label=\"" << label << "\"";

    // set the color with get_color as precedent
    {
      auto iter = get_color.find(id);
      if(iter != get_color.end()) {
        color = iter->second;
      }
    }

    if(color != "") {
      out << ",color=\"" << color << "\"";
    }
    out << "]" << endl;

    int _i = 0;
    for(auto const& inn: node.inns) {
      out << tab << "n" << inn << " -> " << "n" << id <<
        "[label=\"" << write_with_ss(_i++) << "\"]" << endl;
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

set<int> graph_t::compute_nodeset(
  vector<int> const& upps,
  vector<int> const& dwns,
  bool include_upps_dwns) const
{
  set<int> upp_dwn;
  for(auto const& upp: upps) {
    for(auto const& inn: nodes[upp].get_inns_set()) {
      upp_dwn.insert(inn);
    }
  }
  {
    set<int> pending = upp_dwn;
    while(pending.size() > 0) {
      set<int> next_up;
      for(auto const& upp: pending) {
        for(auto const& inn: nodes[upp].get_inns_set()) {
          upp_dwn.insert(inn);
          next_up.insert(inn);
        }
      }
      pending = std::move(next_up);
    }
  }

  set<int> dwn_upp;
  for(auto const& dwn: dwns) {
    for(auto const& out: nodes[dwn].outs) {
      dwn_upp.insert(out);
    }
  }
  {
    set<int> pending = dwn_upp;
    while(pending.size() > 0) {
      set<int> next_up;
      for(auto const& dwn: pending) {
        for(auto const& out: nodes[dwn].outs) {
          dwn_upp.insert(out);
          next_up.insert(out);
        }
      }
      pending = std::move(next_up);
    }
  }

  set<int> ret;
  for(auto const& id: upp_dwn) {
    if(dwn_upp.count(id) > 0) {
      ret.insert(id);
    }
  }

  if(include_upps_dwns) {
    for(auto const& upp: upps) {
      ret.insert(upp);
    }
    for(auto const& dwn: dwns) {
      ret.insert(dwn);
    }
  }

  return ret;
}

vector<int> graph_t::backprop(int out, vector<int> weights) {
  // Get nodes which values affect output of the graph
  set<int> nodeset = compute_nodeset({out}, weights, true);

  backprop_state_t state {
    .grads = {},
    .self = *this,
    .nodeset = std::move(nodeset)
  };

  state.start(out);

  vector<int> grads;
  grads.reserve(weights.size());
  for(auto const& weight : weights) {
    backprop_tensor_t grad = state[weight];
    if(grad.is_constant()) {
      grads.push_back(insert_fill(grad.get_fill()));
    } else {
      grads.push_back(grad.get_id());
    }
  }

  for(int i = 0; i != grads.size(); ++i) {
    dtype_t w_dtype = nodes[weights[i]].op.out_dtype();
    dtype_t g_dtype = nodes[grads[i]].op.out_dtype();
    if(w_dtype != g_dtype) {
      throw std::runtime_error("incorrect dtype of grad");
    }
  }

  return grads;
}

graph_t::backprop_tensor_t::backprop_tensor_t()
  : op(-1)
{}

graph_t::backprop_tensor_t::backprop_tensor_t(int id)
  : op(id)
{}

graph_t::backprop_tensor_t::backprop_tensor_t(fill_t const& fill)
  : op(fill)
{}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_t::backprop_tensor_t::ones(
  dtype_t const& dtype,
  vector<uint64_t> const& shape)
{
  scalar_t value;
  if(dtype_is_complex(dtype)) {
    if(dtype != dtype_t::c64) {
      throw std::runtime_error("not supported complex");
    }
    value = scalar_t(std::complex<float>(1.0, 0.0));
  } else {
    value = scalar_t(dtype, "1.0");
  }

  return backprop_tensor_t(fill_t {
    .value = value,
    .shape = shape
  });
}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_t::backprop_tensor_t::zeros(
  dtype_t const& dtype,
  vector<uint64_t> const& shape)
{
  return backprop_tensor_t(fill_t {
    .value = scalar_t::zero(dtype),
    .shape = shape
  });
}

int const& graph_t::backprop_tensor_t::get_id() const {
  return std::get<int>(op);
}

fill_t const& graph_t::backprop_tensor_t::get_fill() const {
  return std::get<fill_t>(op);
}

scalar_t graph_t::backprop_tensor_t::get_constant() const {
  return get_fill().value;
}

bool graph_t::backprop_tensor_t::is_constant() const {
  return std::holds_alternative<fill_t>(op);
}

bool graph_t::backprop_tensor_t::is_constant_of(scalar_t v) const {
  if(is_constant()) {
    return get_fill().value == v;
  }
  return false;
}

bool graph_t::backprop_tensor_t::is_zeros() const {
  if(is_constant()) {
    scalar_t const& v = get_fill().value;
    return scalar_t::zero(v.dtype) == v;
  }
  return false;
}

bool graph_t::backprop_tensor_t::is_ones() const {
  if(is_constant()) {
    auto const& [scalar, _] = get_fill();

    // scalar_t::one is not valid for complex values, so use this
    // 1.0 value in the context of backprop
    if(dtype_is_complex(scalar.dtype)) {
      if(scalar.dtype != dtype_t::c64) {
        throw std::runtime_error("this complex dtype is not supported");
      }
      std::complex<float> v(1.0,0.0);
      return scalar_t(v) == scalar;
    }
    return scalar_t::one(scalar.dtype) == scalar;
  }
  return false;
}

dtype_t graph_t::backprop_tensor_t::dtype(graph_t& self) const {
  if(is_constant()) {
    return get_fill().value.dtype;
  } else {
    int const& id = get_id();
    return self.nodes[id].op.out_dtype();
  }
}

vector<uint64_t> graph_t::backprop_tensor_t::shape(graph_t& self) const {
  if(is_constant()) {
    return get_fill().shape;
  } else {
    int const& id = get_id();
    return self.nodes[id].op.out_shape();
  }
}

void graph_t::backprop_state_t::start(int out_id)
{
  auto const& op = self.nodes[out_id].op;
  backprop_tensor_t tensor =
    backprop_tensor_t::ones(op.out_dtype(), op.out_shape());
  grads.insert({out_id, tensor});
}

vector<graph_t::backprop_state_t::out_edge_t>
graph_t::backprop_state_t::get_out_edges(int id) const
{
  auto const& outs = self.nodes[id].outs;

  vector<out_edge_t> ret;
  ret.reserve(2*outs.size());

  for (auto const& out : outs) {
    if (nodeset.count(out) > 0) {
      auto inns = self.nodes[out].inns;
      for (int which_inn = 0; which_inn != inns.size(); ++which_inn)
      {
        auto const& inn = inns[which_inn];
        if (inn == id) {
          ret.emplace_back(out_edge_t {
            .out = out,
            .which_inn = which_inn
          });
        }
      }
    }
  }

  return ret;
}

graph_t::backprop_tensor_t
graph_t::backprop_state_t::operator[](int id)
{
  if(grads.count(id) > 0 ) {
    return grads.at(id);
  }
  if(nodeset.count(id) == 0) {
    throw std::runtime_error("This id is not in the nodeset");
  }

  auto const& node = self.nodes[id];
  vector<out_edge_t> out_edges = get_out_edges(id);
  dtype_t dtype = node.op.out_dtype();

  vector<backprop_tensor_t> terms;
  terms.reserve(out_edges.size());
  for(auto const& [out, which_inn] : out_edges) {
    // building grad term for out with respect to this id
    backprop_tensor_t out_grad = (*this)[out];
    backprop_tensor_t term = self.build_grad_term(out, which_inn, out_grad);
    if(term.dtype(self) != dtype) {
      throw std::runtime_error("invalid term dtype during backprop");
    }
    terms.push_back(term);
  }

  if(terms.size() == 0) {
    throw std::runtime_error("No terms, no compute path");
  }

  backprop_tensor_t ret;
  if(terms.size() == 1) {
    ret = terms[0];
  } else {
    ret = self.insert_adds(terms);
  }

  grads.insert({id, ret});

  return ret;
}

graph_t::backprop_tensor_t
graph_t::build_grad_term(int id, int which_inn, backprop_tensor_t grad_id)
{
  auto const& node = nodes[id];
  auto const& op = node.op;
  auto const& inns = node.inns;

  if(which_inn < 0 || which_inn >= inns.size()) {
    throw std::runtime_error("Invalid which_inn in build graph term");
  }

  if(op.is_input()) {
    throw std::runtime_error("should not happen: grad at input");
  } else if(op.is_formation()) {
    // this node, at the graph_t level here, is just an identity operation.
    return grad_id;
  } else if(op.is_complexer()) {
    return build_grad_term_complexer(grad_id);
  } else if(op.is_fill()) {
    // fill is just constant values that don't depend on anything
    return backprop_tensor_t::zeros(op.out_dtype(), op.out_shape());
  } else if(op.is_select()) {
    return build_grad_term_select(
      op.get_select(), which_inn, grad_id);
  } else if(op.is_einsummable()) {
    return build_grad_term_einsummable(
      op.get_einsummable(), inns, which_inn, grad_id);
  } else {
    throw std::runtime_error("should not reach: missing graph type");
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_einsummable(
  einsummable_t const& e,
  vector<int> const& inn_ids,
  int which_inn,
  backprop_tensor_t grad_id)
{
  int num_inn = inn_ids.size();
  if(num_inn > 2 || num_inn == 0) {
    throw std::runtime_error("build grad term: only einsummable "
                             "with 1 or 2 inputs supported");
  }

  if(e.has_broadcast()) {
    // Given ijk->ijkl,
    // form ijkl->ijk
    vector<int> inns;
    int out_rank;
    {
      set<int> bmodes = e.get_broadcast_modes();

      string letters(e.out_rank, ' ');
      std::iota(letters.begin(), letters.end(), 'a');

      string const& inn = letters;

      string out(e.out_rank - bmodes.size(), ' ');
      auto iter = out.begin();
      for(int i = 0; i != e.out_rank; ++i) {
        if(bmodes.count(i) == 0) {
          *iter++ = letters[i];
        }
      }
      if(iter != out.end()) {
        throw std::runtime_error("invalid iter end state");
      }

      string str = inn + "->" + out;

      auto [inns_, out_rank_] = einsummable_t::parse_str(str);
      inns = inns_[0];
      out_rank = out_rank_;
    }

    backprop_tensor_t fixed_grad_id = backprop_tensor_aggregate(grad_id, inns, out_rank);

    if(e.is_broadcast()) {
      // If this is just a broadcast, we have the answer
      return fixed_grad_id;
    } else {
      // This einsummable is a compute and then a broadcast,
      // recurse to the compute part.
      return build_grad_term_einsummable(
        e.remove_broadcast(),
        inn_ids,
        which_inn,
        fixed_grad_id);
    }
  }

  if(num_inn == 1) {
    if(which_inn != 0) {
      throw std::runtime_error("invalid which inn");
    }
    if(e.has_aggregation()) {
      // TODO
      // What if the join is not identity?
    } else {
      return build_grad_term_ew(e, inn_ids, 0, grad_id);
    }
  } else {
    if(e.has_aggregation()) {
      if(!(e.join.is_mul() && e.castable.value() == castable_t::add)) {
        throw std::runtime_error("with multiple inputs and an agg, must be contraction");
      }

      // This is a contraction, so multiply the grad_id with either the
      // left or the right input
      return build_grad_term_contraction(e, inn_ids, which_inn, grad_id);
    } else {
      // This is a binary elementwise op
      return build_grad_term_ew(e, inn_ids, which_inn, grad_id);
    }
  }

  throw std::runtime_error(
    "should not reach; something probably not implemented");
}

graph_t::backprop_tensor_t
graph_t::backprop_tensor_aggregate(
  graph_t::backprop_tensor_t const& tensor,
  vector<int> const& inn,
  int out_rank)
{
  einsummable_t e = einsummable_t::aggregate(
    tensor.shape(*this),
    inn, out_rank,
    tensor.dtype(*this),
    castable_t::add);

  if(tensor.is_constant()) {
    vector<uint64_t> out_shape = e.out_shape();
    scalar_t value = tensor.get_constant();
    uint64_t nelem_inn = product(e.join_shape);
    uint64_t nelem_out = product(out_shape);
    double multiplier(nelem_inn / nelem_out);
    value *= multiplier;
    return backprop_tensor_t(fill_t {
      .value = value,
      .shape = out_shape
    });
  } else {
    int const& id = tensor.get_id();
    return backprop_tensor_t(insert_einsummable(e, { id }));
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_contraction(
  einsummable_t const& e,
  vector<int> const& inn_ids,
  int which_inn,
  backprop_tensor_t grad)
{
  if(which_inn != 0 && which_inn != 1) {
    throw std::runtime_error("contraction has two inputs..");
  }

  if(grad.is_zeros()) {
    dtype_t dtype = e.out_dtype();
    vector<uint64_t> shape =
      which_inn == 0 ?
      e.inn_shape(0) :
      e.inn_shape(1) ;
    return backprop_tensor_t::zeros(dtype, shape);
  }

  if(grad.is_constant()) {
    //  Suppose we have this contraction
    //    ij,jk->ik
    //  where the rhs is a constant of value v != 0.0
    //
    //  Out[i,k] = Sum_j Lhs[i,j] * Rhs[j,k]
    //           = Sum_j v * Lhs[i,j]
    //  This is einsummable
    //    ij->ik
    //  where join_op = lambda x0: v*x0

    auto const& value = grad.get_fill().value;

    string new_str;
    vector<uint64_t> new_inn_shape, new_out_shape;
    int new_inn_id;
    auto [_, inn_strs] = e.str_terms();
    if(which_inn == 0) {
      new_inn_shape = e.inn_shape(1);
      new_inn_id = inn_ids[1];
      new_out_shape = e.inn_shape(0);
      new_str = inn_strs[1] + "->" + inn_strs[0];
    } else {
      new_inn_shape = e.inn_shape(0);
      new_inn_id = inn_ids[0];
      new_out_shape = e.inn_shape(1);
      new_str = inn_strs[0] + "->" + inn_strs[1];
    }

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      new_out_shape, new_inns, { new_inn_shape });

    // Note: if value == 1.0, this should get simplified to the identity function
    scalarop_t join = scalarop_t::make_scale(value);
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank, join, e.castable);

    int ret_id = insert_einsummable(new_e, {new_inn_id});
    if(new_e.has_aggregation()) {
      ret_id = insert_formation(ret_id);
    }
    return backprop_tensor_t(ret_id);
  }

  int const& grad_id = grad.get_id();
  string new_str;
  int new_l_id, new_r_id;
  vector<uint64_t> new_l_shape, new_r_shape, new_o_shape;
  auto [out_str, inn_strs] = e.str_terms();
  if(which_inn == 0) {
    new_l_id = grad_id;
    new_r_id = inn_ids[1];
    new_str = out_str + "," + inn_strs[1] + "->" + inn_strs[0];
    new_l_shape = e.out_shape();
    new_r_shape = e.inn_shape(1);
    new_o_shape = e.inn_shape(0);
  } else { // which_inn == 1
    new_l_id = inn_ids[0];
    new_r_id = grad_id;
    new_str = inn_strs[0] + "," + out_str + "->" + inn_strs[1];
    new_l_shape = e.inn_shape(0);
    new_r_shape = e.out_shape();
    new_o_shape = e.inn_shape(1);
  }

  auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
  vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
    new_o_shape, new_inns, { new_l_shape, new_r_shape });
  einsummable_t new_e(new_join_shape, new_inns, new_out_rank, e.join, e.castable);

  int join_id = insert_einsummable(new_e, {new_l_id, new_r_id});
  int term_id = insert_formation(join_id);
  return backprop_tensor_t(term_id);
}

// example: (yhat - y)**2 => 2(yhat - y) * node_grad
//          -------------    -----------
//          ^ op             ^ deri_op
//                           -----------------------
//                           ^ join_op
//          (here derivative wrt input yhat)


// example: f(x) => f'(x) * node_grad
//                  -----
//                  ^ deri_op
//                  -----------------
//                  ^ join_op
//
// For op = inn -> out:
//   If neither constant:
//     f'(inn) * node_grad
//     inn,out -> inn
//   If just deri_op is a constant:
//     deri_op_v * node_grad
//     out -> inn
//   If node_grad is a constant:
//     f'(x) * grad_v
//     inn -> inn
//   If both are constants:
//     v( = deri_op_v * grad_v)
//     constant of shape inn
graph_t::backprop_tensor_t
graph_t::build_grad_term_ew(
  einsummable_t const& e,
  vector<int> inn_ids,
  int which_inn,
  backprop_tensor_t grad)
{
  scalarop_t deri_op = e.join.derivative(which_inn);

  bool constant_deri_op = deri_op.is_constant();
  bool constant_grad = grad.is_constant();

  // Note: remember we need to return a tensor of inn_dtype

  dtype_t out_dtype = e.out_dtype();
  dtype_t inn_dtype = e.inn_dtype(which_inn);

  auto [out_str, inn_strs] = e.str_terms();
  vector<vector<uint64_t>> inn_shapes = e.inn_shapes();
  vector<uint64_t> out_shape = e.out_shape();

  if(constant_deri_op && constant_grad) {
    scalar_t grad_constant = grad.get_constant();
    scalar_t deri_constant = deri_op.eval({});

    scalar_t v = scalarop_t::make_mul(out_dtype).eval({grad_constant, deri_constant});
    v = v.convert(inn_dtype);

    return backprop_tensor_t(fill_t {
      .value = v,
      .shape = inn_shapes[which_inn]
    });
  } else if(constant_deri_op) {
    scalar_t value = deri_op.eval({}).convert(inn_dtype);

    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        scalarop_t::make_constant(value),
        scalarop_t::make_identity(out_dtype)
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    if(new_join.is_constant()) {
      // It could be the case that the new_join is simplified to a constant
      // function. (For example, value == 0.0)
      scalar_t new_value = new_join.eval({});
      return backprop_tensor_t(fill_t {
        .value = new_value,
        .shape = inn_shapes[which_inn]
      });
    }

    string new_str = out_str + "->" + inn_strs[which_inn];
    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, { out_shape });
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank, new_join);
    int term_id = insert_einsummable(new_e, { grad.get_id() });
    return backprop_tensor_t(term_id);
  } else if(constant_grad) {
    scalar_t value = grad.get_constant();

    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        deri_op,
        scalarop_t::make_constant(value)
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    if(new_join.is_constant()) {
      // It could be the case that the new_join is simplified to a constant
      // function. (For example, value == 0.0)
      scalar_t new_value = new_join.eval({});
      return backprop_tensor_t(fill_t {
        .value = new_value,
        .shape = inn_shapes[which_inn]
      });
    }

    vector<string> actual_inn_strs;
    vector<int> new_inn_ids;
    vector<vector<uint64_t>> new_inn_shapes;
    for(int i = 0; i != inn_strs.size(); ++i) {
      if(new_join.is_used(i)) {
        actual_inn_strs.push_back(inn_strs[i]);
        new_inn_ids.push_back(inn_ids[i]);
        new_inn_shapes.push_back(inn_shapes[i]);
      }
    }
    string new_str = actual_inn_strs[0];
    for(int i = 1; i != actual_inn_strs.size(); ++i) {
      new_str += "," + actual_inn_strs[i];
    }
    new_str += "->" + inn_strs[which_inn];

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, new_inn_shapes);
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank, new_join);
    int term_id = insert_einsummable(new_e, new_inn_ids);
    return backprop_tensor_t(term_id);
  } else {
    scalarop_t new_join = scalarop_t::combine(
      scalarop_t::make_mul(out_dtype),
      vector<scalarop_t> {
        deri_op,
        scalarop_t::make_identity(out_dtype)
      }
    );

    new_join = scalarop_t::combine(
      scalarop_t::make_convert_dtype(out_dtype, inn_dtype),
      vector<scalarop_t>{ new_join });

    vector<string> actual_inn_strs;
    vector<int> new_inn_ids;
    vector<vector<uint64_t>> new_inn_shapes;
    for(int i = 0; i != inn_strs.size(); ++i) {
      if(new_join.is_used(i)) {
        actual_inn_strs.push_back(inn_strs[i]);
        new_inn_ids.push_back(inn_ids[i]);
        new_inn_shapes.push_back(inn_shapes[i]);
      }
    }
    if(new_join.is_used(inn_strs.size())) {
      actual_inn_strs.push_back(out_str);
      new_inn_ids.push_back(grad.get_id());
      new_inn_shapes.push_back(out_shape);
    }

    string new_str = actual_inn_strs[0];
    for(int i = 1; i != actual_inn_strs.size(); ++i) {
      new_str += "," + actual_inn_strs[i];
    }
    new_str += "->" + inn_strs[which_inn];

    auto [new_inns, new_out_rank] = einsummable_t::parse_str(new_str);
    vector<uint64_t> new_join_shape = einsummable_t::construct_join_shape(
      inn_shapes[which_inn], new_inns, new_inn_shapes);
    einsummable_t new_e(new_join_shape, new_inns, new_out_rank, new_join);
    int term_id = insert_einsummable(new_e, new_inn_ids);
    return backprop_tensor_t(term_id);
  }
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_select(
  select_t const& select,
  int which_inn,
  backprop_tensor_t grad)
{
  // If grad is zeros, then just return zeros from the input shape
  if(grad.is_zeros()) {
    dtype_t const& dtype = grad.get_constant().dtype;
    return backprop_tensor_t::zeros(dtype, select.inn_shape(which_inn));
  }

  // If which_inn uses the full input tensor:
  bool uses_full_input = true;
  {
    for(auto const& sd: select.inn_regions[which_inn]) {
      if(sd.d_inn != sd.size) {
        uses_full_input = false;
        break;
      }
    }
  }
  if(uses_full_input) {
    if(grad.is_constant()) {
      return backprop_tensor_t(fill_t {
        .value = grad.get_constant(),
        .shape = select.inn_shape(which_inn)
      });
    }
    // Subset the gradient and return that
    int const& grad_id = grad.get_id();
    return backprop_tensor_t(insert_subset(
      select.wrt_output_inn_hrect(which_inn),
      grad_id));
  }

  // Only portion of the input region is getting selected by the output. All other
  // portions are to be set to zero.
  hrect_t inn_grad_hrect = select.wrt_input_inn_hrect(which_inn);
  vector<uint64_t> inn_shape = select.inn_shape(which_inn);
  int rank = inn_shape.size();
  partition_t partition = [&] {
    vector<partdim_t> pds;
    pds.reserve(rank);
    for(int i = 0; i != rank; ++i) {
      auto const& [beg,end] = inn_grad_hrect[i];
      uint64_t const& size = inn_shape[i];
      vector<uint64_t> spans;
      if(beg != 0) {    spans.push_back(beg); }
      if(end != size) { spans.push_back(end); }
      spans.push_back(size);
      pds.push_back(partdim_t::from_spans(spans));
    }
    return partition_t(pds);
  }();

  dtype_t const& dtype = select.dtype;

  vector<int> block_shape = partition.block_shape();
  int num_blocks = product(block_shape);

  vector<int> inn_ids;
  inn_ids.reserve(num_blocks);

  using selectdim_t = select_t::selectdim_t;

  vector<vector<selectdim_t>> inn_regions;
  inn_regions.reserve(num_blocks);

  vector<int> index(rank);
  do {
    hrect_t block_hrect = partition.get_hrect(index);

    // Case 1: this block is the grad block
    //   Case 1a: grad is just a constant, so we need to fill the block with
    //            a constant of that value
    //   Case 1b: grad is not a constant, so we need to use those values
    // Case 2: this block is not the grad block, fill the block with zeros

    // Filling with a constant is mechanically
    // the same code, so separate those out
    optional<scalar_t> maybe_constant_fill = std::nullopt;
    if(block_hrect == inn_grad_hrect) {
      if(grad.is_constant()) {
        maybe_constant_fill = grad.get_constant();
      } else {
        // case 1b
      }
    } else {
      maybe_constant_fill = scalar_t::zero(dtype);
    }

    if(maybe_constant_fill) {
      // Case 1a, Case 2
      inn_regions.emplace_back();
      auto& inn_region = inn_regions.back();
      inn_region.reserve(rank);
      vector<uint64_t> block_shape;
      block_shape.reserve(rank);
      for(int i = 0; i != rank; ++i) {
        auto const& [beg,end] = block_hrect[i];
        uint64_t size = end-beg;
        inn_region.push_back(selectdim_t {
          .d_inn = size,
          .offset_inn = 0,
          .offset_out = beg,
          .size = size
        });
        block_shape.push_back(size);
      }

      inn_ids.push_back(insert_fill(fill_t {
        .value = maybe_constant_fill.value(),
        .shape = block_shape
      }));
    } else {
      // Case 1b
      vector<uint64_t> out_start = select.wrt_output_point(
        vector_mapfst(block_hrect),
        which_inn);
      vector<uint64_t> const& out_shape = select.out_shape;

      inn_regions.emplace_back();
      auto& inn_region = inn_regions.back();
      inn_region.reserve(rank);
      vector<uint64_t> block_shape;
      block_shape.reserve(rank);
      for(int i = 0; i != rank; ++i) {
        auto const& [beg,end] = block_hrect[i];
        uint64_t size = end-beg;
        auto const& o_beg = out_start[i];
        auto const& o_dim = out_shape[i];
        inn_region.push_back(selectdim_t {
          .d_inn = o_dim,
          .offset_inn = o_beg,
          .offset_out = beg,
          .size = size
        });
        block_shape.push_back(size);
      }

      int const& grad_id = grad.get_id();
      inn_ids.push_back(grad_id);
    }
  } while(increment_idxs(block_shape, index));

  select_t new_select(dtype, inn_shape, inn_regions);
  return backprop_tensor_t(insert(op_t(new_select), inn_ids));
}

graph_t::backprop_tensor_t
graph_t::build_grad_term_complexer(
  graph_t::backprop_tensor_t grad)
{
  // The complexer op is a no op, basically...
  // if grad is complex, turn it real
  // if grad is real, turn it complex
  if(grad.is_constant()) {
    vector<uint64_t> shape = grad.get_fill().shape;
    scalar_t value = grad.get_constant();
    dtype_t const& dtype = value.dtype;
    if(dtype_is_real(dtype)) {
      // (v,v,v,v,v,v) -> ( (v,v), (v,v), (v,v) )
      if(dtype != dtype_t::f32) {
        throw std::runtime_error("complexer confusion");
      }
      float v = value.f32();
      std::complex<float> vv(v,v);

      if(shape.back() % 2 != 0) {
        throw std::runtime_error("odd number of last dims in complexer");
      }
      vector<uint64_t> complex_shape = shape;
      complex_shape.back() /= 2;

      return backprop_tensor_t(fill_t {
        .value = scalar_t(vv),
        .shape = complex_shape
      });
    } else {
      // ( (v,v), (v,v), (v,v) ) ->  (v,v,v,v,v,v)
      if(dtype != dtype_t::c64) {
        throw std::runtime_error("complexer confusion");
      }
      std::complex<float> vu = value.c64();
      if(vu.real() == vu.imag()) {
        float v = vu.real();
        vector<uint64_t> real_shape = shape;
        real_shape.back() *= 2;
        return backprop_tensor_t(fill_t {
          .value = scalar_t(v),
          .shape = real_shape
        });
      } else {
        // ( (v,u), (v,u), (v,u) ) cannot be represented as a constant
        // float tensor, so form it and convert it real
        int id = insert_fill(fill_t {
          .value = value,
          .shape = shape
        });
        return backprop_tensor_t(insert_to_real(id));
      }
    }
  } else {
    int const& id = grad.get_id();
    dtype_t dtype = out_dtype(id);
    if(dtype_is_real(dtype)) {
      return backprop_tensor_t(insert_to_complex(id));
    } else {
      return backprop_tensor_t(insert_to_real(id));
    }
  }
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

tuple<
  vector<tuple<int,int>>,
  graph_constructor_t>
create_remap_graph_constructor(
  remap_relations_t const& _remap)
{
  auto const& remap = _remap.remap;

  graph_constructor_t g;

  vector<tuple<int,int>> remap_gid;
  for(auto const& [src,dst]: remap) {
    int gid_src = g.insert_input(src.placement, src.dtype);
    int gid_dst = g.insert_formation(dst.placement, gid_src, true);
    remap_gid.emplace_back(gid_src, gid_dst);
  }

  return {remap_gid, g};
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
    : tensor_t(_shape, vector_iota<int>(_shape.full_rank()), _id, _self)
{}

graph_writer_t::tensor_t::tensor_t(
  full_shape_t const& _shape,
  vector<int> const& _modes,
  int _id,
  graph_writer_t* _self)
  : shape(_shape), modes(_modes), id(_id), self(_self)
{}

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

  auto& op = self->graph.nodes[ret.id].op;
  if(op.has_aggregation()) {
    ret.id = self->graph.insert_formation(ret.id, true);
  } else {
    op.set_save(true);
  }

  return ret;
}

void graph_writer_t::tensor_t::save_inplace() {
  // would it need a permutation?
  if(_has_permutation()) {
    throw std::runtime_error("tensor with virtual permutation can't be saved");
  }

  auto& op = self->graph.nodes[id].op;
  if(op.has_aggregation()) {
    throw std::runtime_error("save inplace: can't save if has agg");
  } else {
    op.set_save(true);
  }
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
  for(int i = 0; i != rank; ++i) {
    auto const& idx = idxs[i];
    hrect.push_back(idx.get(_shape[i]));
    if(idx.is_squeeze()) {
      throw std::runtime_error("squeezes not implemented");
    }
  }
  return self->subset(hrect, *this);
}

bool
graph_writer_t::tensor_t::_has_permutation() const {
  for(int i = 0; i != modes.size(); ++i) {
    if(modes[i] != i) {
      return true;
    }
  }
  return false;
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::physically_permute() const {
  tensor_t ret = *this;

  if(!ret._has_permutation()) {
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
  std::iota(ret.modes.begin(), ret.modes.end(), 0);

  return ret;
}

vector<graph_writer_t::tensor_t>
graph_writer_t::backprop(
  graph_writer_t::tensor_t out,
  vector<graph_writer_t::tensor_t> inns)
{
  // It could be the case that inns do not have a shape that is grouped
  // or that they have a permutation.
  //
  // Here, take the gradient wrt wtvr get_id() gives, but then wrap the resulting
  // gradient with the correct shape + permutation.
  //
  // The output tensor shape grouping or permutations doesn't matter because
  // the gradient is wrt the sum of all the elements.

  vector<int> inn_ids;
  for (auto const& inn : inns) {
    inn_ids.push_back(inn.get_id());
  }
  vector<int> grad_ids = graph.backprop(out.get_id(), inn_ids);

  vector<tensor_t> grads;
  for(int which = 0; which != inns.size(); ++which) {
    auto const& inn = inns[which];
    auto const& grad_id = grad_ids[which];
    grads.push_back(tensor_t(inn.shape, inn.modes, grad_id, this));
  }

  return grads;
}

graph_t::backprop_tensor_t
graph_t::insert_adds(vector<backprop_tensor_t> const& items_)
{
  if(items_.size() == 0) {
    throw std::runtime_error("should not be summing empty list of tensors");
  }

  dtype_t dtype;
  vector<uint64_t> shape;
  int rank;
  {
    auto const& tensor = items_[0];
    dtype = tensor.dtype(*this);
    shape = tensor.shape(*this);
    rank = shape.size();
  }

  scalar_t sum = scalar_t::zero(dtype);
  vector<int> items;
  items.reserve(items_.size());
  for(auto const& tensor: items_) {
    if(tensor.is_constant()) {
      sum += tensor.get_constant();
    } else {
      items.push_back(tensor.get_id());
    }
  }

  if(items.size() == 0) {
    // In this case, all the terms were constants
    return backprop_tensor_t(fill_t {
      .value = sum,
      .shape = shape
    });
  }

  if(sum != scalar_t::zero(dtype)) {
    items.push_back(insert_fill(fill_t {
      .value = sum,
      .shape = shape
    }));
  }

  vector<int> is = vector_iota<int>(rank);
  vector<vector<int>> inns{ is, is };
  einsummable_t e(shape, inns, rank, scalarop_t::make_add(dtype));

  while(items.size() != 1) {
    int n = items.size() / 2;
    int r = items.size() % 2;
    vector<int> next_up;
    next_up.reserve(n + r);
    if(r == 1) {
      next_up.push_back(items.back());
    }
    for(int i = 0; i != n; ++i) {
      next_up.push_back(insert_einsummable(e, {items[2*i], items[2*i+1]}));
    }
    items = next_up;
  }

  return backprop_tensor_t(items[0]);
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

graph_writer_t::tensor_t
graph_writer_t::constant(
  scalar_t value,
  vector<uint64_t> shape,
  dtype_t dtype)
{
  return this->constant(value, full_shape_t::from_full(shape), dtype);
}

graph_writer_t::tensor_t
graph_writer_t::constant(
  scalar_t value,
  full_shape_t shape,
  dtype_t dtype)
{
  int id = graph.insert_fill(fill_t {
    .value = value,
    .shape = shape.full()
  });

  return tensor_t(shape, id, this);
}

graph_writer_t::tensor_t
graph_writer_t::constant(
  scalar_t value,
  vector<vector<uint64_t>> const& shape,
  dtype_t dtype)
{
  return this->constant(value, full_shape_t::from_vecvec(shape), dtype);
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
    castable);

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

  vector<vector<uint64_t>> out_vecvec;

  for(int i = 0; i != rank; ++i) {
    auto const& [b,e] = breaks[i];

    if(is_subset_dim(i)) {
      full_hrect.push_back(hrect[i]);
      auto const& [_b,_e] = hrect[i];
      out_vecvec.push_back({ _e-_b});
    } else {
      out_vecvec.emplace_back();
      for(int f = b; f != e; ++f) {
        uint64_t const& d = inn_full_shape[f];
        full_hrect.emplace_back(0, d);
        out_vecvec.back().push_back(d);
      }
    }
  }

  int out_id = graph.insert_subset(full_hrect, inn.get_id());

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
  string ewbstr = ha + "," + h + "->" + ha;

  tensor_t x = inn;

  tensor_t c = reduction(
    redstr,
    castable_t::max,
    x);

  // x = x - c
  x = ew(
    ewbstr,
    scalarop_t::make_sub(dtype),
    x, c);

  // ex = exp(x)
  tensor_t ex = ew(
    scalarop_t::make_exp(dtype),
    x);

  tensor_t sum_ex = reduction(
    redstr,
    castable_t::add,
    ex);

  return ew(
    ewbstr,
    scalarop_t::make_div(dtype),
    ex, sum_ex);
}

graph_writer_t::tensor_t
graph_writer_t::broadcast(
  graph_writer_t::full_dim_t b_dim,
  graph_writer_t::tensor_t const& tensor)
{
  string str;
  {
    int n_bro = b_dim.parts.size();

    vector<char> letters(tensor.modes.size() + n_bro);
    std::iota(letters.begin(), letters.end(), 'a');

    string inn;
    for(auto const& m: tensor.modes) {
      inn.push_back(letters[n_bro + m]);
    }

    string out(letters.begin(), letters.end());

    str = inn + "->" + out;
  }

  full_shape_t new_shape(vector_concatenate(
    {b_dim}, tensor.get_shape().parts));

  einsummable_t e = [&] {
    auto [inns, out_rank] = einsummable_t::parse_str(str);
    return einsummable_t(
      new_shape.full(), inns, out_rank,
      scalarop_t::make_identity(tensor.get_dtype()));
  }();

  int id = graph.insert_einsummable(e, { tensor.get_id() });

  return tensor_t(new_shape, id, this);
}

graph_writer_t::tensor_t
graph_writer_t::broadcast(
  uint64_t sz,
  graph_writer_t::tensor_t const& inn)
{
  return broadcast(full_dim_t::singleton(sz), inn);
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


