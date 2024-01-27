#include "gwriter.h"

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
  set<int> squeeze_dims;
  vector<tuple<uint64_t, uint64_t>> hrect;
  hrect.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    auto const& idx = idxs[i];
    hrect.push_back(idx.get(_shape[i]));
    if(idx.is_squeeze()) {
      squeeze_dims.insert(i);
    }
  }

  tensor_t ret = self->subset(hrect, *this);

  if(squeeze_dims.size() > 0) {
    if(ret._has_permutation()) {
      // this is super subtle: subset may have been a no op and did nothing.
      ret = ret.physically_permute();
    }

    vector<full_dim_t> new_parts;
    auto const& prev_parts = ret.get_shape().parts;
    for(int i = 0; i != prev_parts.size(); ++i) {
      if(squeeze_dims.count(i) == 0) {
        new_parts.push_back(prev_parts[i]);
      }
    }
    full_shape_t new_shape(new_parts);

    int new_id = self->graph.insert_squeezer(new_shape.full(), ret.get_id());

    ret = tensor_t(new_shape, new_id, self);
  }

  return ret;
}

graph_writer_t::tensor_t
graph_writer_t::tensor_t::squeeze(int which_dim) const
{
  vector<uint64_t> shape = get_shape()();
  int rank = shape.size();

  if(which_dim < 0) {
    which_dim = rank + which_dim;
  }

  if(which_dim < 0 || which_dim >= rank) {
    throw std::runtime_error("cannot squeeze this dimension: invalid which_dim");
  }
  if(shape[which_dim] != 1) {
    throw std::runtime_error("cannot squeez this dimension: size > 1");
  }

  vector<idx_t> idxs;
  idxs.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    if(i == which_dim) {
      idxs.emplace_back(idx_t::idx{ 0 });
    } else {
      idxs.emplace_back(idx_t::rng{ 0, int64_t(shape[i]) });
    }
  }
  return subset(idxs);
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
  int id = graph.insert_constant(value, shape.full());
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
graph_writer_t::exp(
  graph_writer_t::tensor_t const& inn)
{
  return ew(
    scalarop_t::make_exp(inn.get_dtype()),
    inn);
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
    throw std::runtime_error("graph writer straight bew : invalid shapes");
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
