#include "einsummable.h"

int _compute_join_rank(
  vector<vector<int>> const& inns,
  int out_rank)
{
  int almost_join_rank = 0;
  for(auto const& inn: inns) {
    if(inn.size() == 0) {
      throw std::runtime_error("compute_join_rank: empty inn given");
    }

    almost_join_rank = std::max(
      almost_join_rank,
      *std::max_element(inn.begin(), inn.end()));
  }
  almost_join_rank++;

  // Example:
  //   ij->ijk
  //   01->012
  //
  //   alomst_join_rank = 2
  //   out_rank = 3
  //   join_rank = 3
  //
  //   ij->ik
  //   02->01
  //
  //   almost_join_rank = 3
  //   out_rank = 2
  //   join_rank = 3
  return std::max(almost_join_rank, out_rank);
}

einsummable_t::einsummable_t(
  vector<uint64_t> join_shape,
  vector<vector<int>> inns,
  int out_rank,
  scalarop_t join,
  optional<castable_t> castable)
  : join_shape(join_shape),
    inns(inns),
    out_rank(out_rank),
    join(join),
    castable(castable)
{
  if(has_aggregation() && !castable) {
    throw std::invalid_argument("Must have castable when aggregating.");
  }

  if(join.num_inputs() != inns.size()) {
    throw std::runtime_error("einsummable inns size not same as scalarop join");
  }
  // Consider batched matrix multiply into
  // the same output:
  //   bij,bjk->ik
  // There are two possible einsummable representations:
  //   203 231->01 and
  //   302 321->01
  // This constructor forces the first version.
  {
    int agg_cnt = out_rank;
    map<int,int> fix;
    for(vector<int>& inn: inns) {
      for(int& i: inn) {
        if(i >= out_rank) {
          if(fix.count(i) == 0) {
            fix.insert({i, agg_cnt});
            agg_cnt += 1;
          }
          i = fix.at(i);
        }
      }
    }
  }

  int join_rank = _compute_join_rank(inns, out_rank);
  if(join_rank != join_shape.size()) {
    throw std::runtime_error("invalid join shape size");
  }

  for(uint64_t const& d: join_shape) {
    if(d == 0) {
      throw std::runtime_error("einsummable: cannot have zero join size");
    }
  }

  if(!valid_inns_out(inns, out_rank)) {
    throw std::runtime_error("einsummable: invalid inns, out rank");
  }
}

int _count_adjacent(
  vector<int>::const_iterator iter,
  vector<int>::const_iterator end)
{
  int cnt = 0;
  int v = *iter;
  while(iter != end && *iter++ == v++) {
    cnt++;
  }
  return cnt;
}

int update_count_adjacent(
  int score,
  int r,
  vector<int> const& is)
{
  auto iter = std::find(is.begin(), is.end(), r);
  if(score == 0) {
    return _count_adjacent(iter, is.end());
  } else {
    if(iter == is.end()) {
      int e = score + r;
      for(auto const& i: is) {
        if(i >= r && i < e) {
          e = i;
        }
      }
      return e-r;
    } else {
      return std::min(score, _count_adjacent(iter, is.end()));
    }
  }
}

vector<tuple<int,int>> build_adjacents(
  vector<vector<int>> const& inns)
{
  int r = 0;
  vector<tuple<int,int>> ret;

  int m = 0;
  for(auto const& inn: inns) {
    m = std::max(m, *std::max_element(inn.begin(), inn.end()));
  }
  m += 1;

  while(r != m) {
    // find the first place where r is in inns (so score > 0)
    // then loop through all the others and update the score
    // Example: r = mode d, inns = ae,abcde,abc
    //          then score = 2 at i = 1,
    //          then score stays two after i = 2
    //          but at i = 0, score gets chopped to 1
    //          since "de" can't be grouped together
    int score = 0;
    int i = 0;
    for(; i != inns.size() && score == 0; ++i) {
      score = update_count_adjacent(0, r, inns[i]);
    }
    // i is now one past the location that set score to be nonzero
    for(int j = 0; j != inns.size()-1; ++j) {
      int which = (i + j) % inns.size();
      score = update_count_adjacent(score, r, inns[which]);
    }

    if(score <= 0) {
      throw std::runtime_error("impl err");
    }

    ret.emplace_back(r, r+score);
    r += score;
  }

  return ret;
}

einsummable_t einsummable_t::merge_adjacent_dims() const {
  int join_rank = join_shape.size();

  auto _inns = inns;
  {
    vector<int> is(out_rank);
    std::iota(is.begin(), is.end(), 0);
    _inns.push_back(is);
  }

  vector<tuple<int,int>> merges = build_adjacents(_inns);

  if(merges.size() == join_rank) {
    return *this;
  }

  map<int,int> backwards;
  vector<uint64_t> new_join_shape;
  {
    for(int w = 0; w != merges.size(); ++w) {
      auto const& [b,e] = merges[w];

      backwards.insert({b,w});

      uint64_t sz = 1;
      for(int i = b; i != e; ++i) {
        sz *= join_shape[i];
      }
      new_join_shape.push_back(sz);
    }
  }

  vector<vector<int>> new_inns;
  for(auto const& inn: _inns) {
    new_inns.emplace_back();
    vector<int>& new_inn = new_inns.back();

    int i = 0;
    while(i != inn.size()) {
      auto const& w = backwards.at(inn[i]);
      new_inn.push_back(w);

      auto const& [b,e] = merges[w];
      i += (e-b);
    }
  }

  int new_out_rank = new_inns.back().size();
  new_inns.resize(inns.size());

  return einsummable_t(
    new_join_shape, new_inns, new_out_rank,
    join, castable);
}

einsummable_t einsummable_t::from_proto(es_proto::Einsummable const& e) {
  vector<uint64_t> join_shape;
  {
    auto const& js = e.join_shape();
    join_shape = vector<uint64_t>(js.begin(), js.end());
  }

  vector<vector<int>> inns;
  inns.reserve(e.inns_size());
  for(int i = 0; i != e.inns_size(); ++i) {
    auto const& is = e.inns(i).idxs();
    inns.emplace_back(is.begin(), is.end());
  }

  int out_rank = e.out_rank();

  scalarop_t join = parse_with_ss<scalarop_t>(e.join());

  optional<castable_t> castable = std::nullopt;
  if(e.has_castable()) {
    castable = parse_with_ss<castable_t>(e.castable());
  }

  return einsummable_t(join_shape, inns, out_rank, join, castable);
}

einsummable_t einsummable_t::from_wire(string const& str)
{
  es_proto::Einsummable e;
  if(!e.ParseFromString(str)) {
    throw std::runtime_error("could not parse einsummable!");
  }
  return from_proto(e);
}

void einsummable_t::to_proto(es_proto::Einsummable& e) const {
  // join shape
  for(uint64_t const& n: join_shape) {
    e.add_join_shape(n);
  }

  // inns
  for(vector<int> const& inn: inns) {
    es_proto::EsInn* ei = e.add_inns();
    for(int const& i: inn) {
      ei->add_idxs(i);
    }
  }

  // out rank
  e.set_out_rank(out_rank);

  // join
  e.set_join(write_with_ss(join));

  // castable
  if(castable) {
    e.set_castable(write_with_ss(castable.value()));
  }
}

string einsummable_t::to_wire() const
{
  es_proto::Einsummable e;
  to_proto(e);
  string ret;
  e.SerializeToString(&ret);
  return ret;
}

inline
einsummable_t _einsummable_matmul_helper(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype,
  string str)
{
  auto [inns,_] = einsummable_t::parse_str(str);

  auto const& einsum = einsummable_t(
    {di, dk, dj}, inns, 2,
    scalarop_t::make_mul(dtype),
    castable_t::add);

  std::cout << "\t This is join_shape for the created einsumm " << std::endl << "\t\t";

  for(auto i = einsum.join_shape.begin(); i != einsum.join_shape.end(); i++) {
    std::cout << (*i) << " ";
  } 

  std::cout << std::endl;
  if (einsum.join.is_mul()) std::cout << "IT'S A MULLLLLL" << std::endl;
  std::cout << einsum.str() << std::endl;

  return einsum;
}

einsummable_t einsummable_t::from_matmul(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype)
{
  return from_matmul_ss(di, dj, dk, dtype);
}

einsummable_t einsummable_t::from_matmul_ss(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype)
{
  return _einsummable_matmul_helper(di, dj, dk, dtype, "ij,jk->ik");
}

einsummable_t einsummable_t::from_matmul_ts(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype)
{
  return _einsummable_matmul_helper(di, dj, dk, dtype, "ji,jk->ik");
}

einsummable_t einsummable_t::from_matmul_st(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype)
{
  return _einsummable_matmul_helper(di, dj, dk, dtype, "ij,kj->ik");
}

einsummable_t einsummable_t::from_matmul_tt(
  uint64_t di, uint64_t dj, uint64_t dk,
  dtype_t dtype)
{
  return _einsummable_matmul_helper(di, dj, dk, dtype, "ji,kj->ik");
}

einsummable_t einsummable_t::with_new_shape(
  einsummable_t const& e,
  vector<uint64_t> const& new_join_shape)
{
  if(e.join_shape.size() != new_join_shape.size()) {
    throw std::runtime_error("einsummable_t::with_new_shape");
  }
  // copying the previous einsummable and modifying it
  // is much faster than calling the constructor which
  // performs lots of checks.
  einsummable_t ret = e;
  ret.join_shape = new_join_shape;
  return ret;
}

tuple<vector<vector<int>>, int>
einsummable_t::parse_str(string str)
{
  auto iter = str.begin();

  vector<vector<char>> inns(1);
  for(; iter != str.end(); ++iter) {
    char const& c = *iter;
    if(c == '-') {
      ++iter;
      if(iter == str.end() || *iter != '>') {
        throw std::runtime_error("invalid einsummable parse: no arrow");
      }
      ++iter;
      break;
    } else if(c == ',') {
      if(inns.back().size() == 0) {
        throw std::runtime_error("invalid einsummable parse: empty after comma");
      }
      inns.emplace_back();
    } else {
      inns.back().push_back(c);
    }
  }

  if(inns.back().size() == 0) {
    throw std::runtime_error("invalid einsummable parse: empty inn after arrow");
  }

  vector<char> out;
  for(; iter != str.end(); ++iter) {
    char const& c = *iter;
    if(c == '-') {
      throw std::runtime_error("invalid parse: cannot have '-' in output");
    } else if(c == ',') {
      throw std::runtime_error("invalid parse: cannot have ',' in output");
    }
    out.push_back(c);
  }
  if(out.size() == 0) {
    throw std::runtime_error("invalid parse: empty out");
  }

  map<char, int> _get;
  int _cnt = 0;

  auto get = [&](char const& c) {
    if(_get.count(c) == 0) {
      _get.insert({c, _cnt});
      _cnt += 1;
    }
    return _get.at(c);
  };

  // this sets the output mapping to 0,1,...,out_rank-1
  for(char const& c: out) {
    get(c);
  }
  if(_get.size() != out.size()) {
    throw std::runtime_error("output in einsummable cannot have duplicates");
  }

  vector<vector<int>> ret;
  ret.reserve(inns.size());
  for(vector<char> const& inn: inns) {
    ret.emplace_back();
    vector<int>& r = ret.back();
    r.reserve(inn.size());
    for(char const& c: inn) {
      r.push_back(get(c));
    }
  }

  if(!valid_inns_out(ret, out.size())) {
    throw std::runtime_error("invalid indices and output rank: parse");
  }

  return {ret, out.size()};
}

string einsummable_t::make_str(
  vector<vector<int>> const& inns,
  int out_rank)
{
  int join_rank = _compute_join_rank(inns, out_rank);

  vector<char> letters(join_rank); // How many different letters are there 
  std::iota(letters.begin(), letters.end(), 'a');

  auto words = get_inputs_from_join_(inns, letters);

  std::ostringstream ss;
  ss << string(words[0].begin(), words[0].end());
  for(int i = 1; i != words.size(); ++i) {
    auto const& word = words[i];
    ss << "," << string(word.begin(), word.end());
  }

  vector<char> outword(letters.begin(), letters.begin() + out_rank);
  ss << "->" << string(outword.begin(), outword.end());

  return ss.str();
}

string
einsummable_t::create_contraction_vjp_string(int which_inn)
{
  map<int, int> mapz;
  vector<vector<int>> innz;
  vector<int> tmp;
  vector<int> tmp2;
  int ctr = 0;

  for (auto i : inns[which_inn]) {
    mapz.insert({i, ctr++});
  }

  for (auto i : inns[which_inn == 0 ? 1 : 0]) {
    auto it = mapz.find(i);
    if (it == mapz.end()) {
      mapz.insert({i, ctr});
      tmp2.push_back(ctr++);
    } else {
      tmp2.push_back(it->second);
    }
  }

  for (int i = 0; i < out_rank; i++) {
    tmp.push_back(mapz.at(i));
  }

  if (which_inn == 0) {
    innz.push_back(tmp);
    innz.push_back(tmp2);
  } else {
    innz.push_back(tmp2);
    innz.push_back(tmp);
  }
  
  return make_str(innz, inns[which_inn].size());
}

string
einsummable_t::create_binary_vjp_string(vector<int> argument_shape, vector<int> other_shape)
{
  auto identity = identity_permutation(argument_shape.size());
  auto const& permuted_inns = {find_permutation(other_shape, argument_shape), find_permutation(identity, argument_shape)};  
  return make_str(permuted_inns, argument_shape.size());
}

string 
einsummable_t::create_unary_vjp_string(vector<int> inn, int rank)
{
  auto identity = identity_permutation(rank);
  auto inverse = find_permutation(identity, inn); 
  return make_str({inverse, identity}, rank);
}

string 
einsummable_t::create_batch_matmul_string(int lhs_rank, int rhs_rank, bool t_lhs, bool t_rhs) 
{
  int nl = lhs_rank;
  int nr = rhs_rank;
  if(nl < 2 || nr < 2) {
    throw std::runtime_error("graph writer matmul: must have atleast rank 2");
  }

  auto make_header = [](int n) {
    string ret(n, ' ');
    std::iota(ret.begin(), ret.end(), 'd');
    return ret;
  };

  string sl = t_lhs ? "ba" : "ab";
  string sr = t_rhs ? "cb" : "bc";
  string so;

  if(nl == 2 && nr == 2) {
    so = "ac";
  } else if(nl == 2)  {
    int n = nr-2;
    string header = make_header(n);
    sr = header + sr;
    so = header + "ac";
  } else if(nr == 2)  {
    int n = nl-2;
    string header = make_header(n);
    sl = header + sl;
    so = header + "ac";
  } else if(nl == nr) {
    int n = nl-2;
    string header = make_header(n);
    sl = header + sl;
    sr = header + sr;
    so = header + "ac";
  } else {
    throw std::runtime_error("graph writer matmul: invalid inputs");
  }

  return sl + "," + sr + "->" + so;
}

string
einsummable_t::normalize_str(string const& str)
{
  auto [inns, out_rank] = parse_str(str);
  return make_str(inns, out_rank);
}

bool einsummable_t::valid_inns_out(
  vector<vector<int>> const& inns,
  int out_rank)
{
  // scalars are not supported
  if(out_rank == 0) {
    return false;
  }
  for(auto const& inn: inns) {
    if(inn.size() == 0) {
      return false;
    }
  }

  // can't have any duplicates in the inns
  for(auto const& inn: inns) {
    int n_unique = set<int>(inn.begin(), inn.end()).size();
    if(n_unique != inn.size()) {
      return false;
    }
  }

  // 2. the aggs must occur in order and be contigous
  //
  //    so 013->01 is illegal and should be
  //       012->01
  //
  //    and 021,012->0 is illegal and should be
  //        012,021->0
  {
    int agg_cnt = out_rank;
    for(auto const& inn: inns) {
      for(int const& i: inn) {
        if(i >= out_rank) {
          if(i == agg_cnt) {
            agg_cnt += 1;
          } else if(i < agg_cnt) {
            // fine, just another occurence of a previously seen agg
          } else {
            // uh oh, the aggs must be in order
            return false;
          }
        }
      }
    }
  }

  return true;
}

optional<vector<uint64_t>>
einsummable_t::construct_join_shape(
  vector<vector<int>> const& inns,
  vector<vector<uint64_t>> const& inn_shapes)
{
  uint64_t d = 0;
  auto ret = construct_join_shape_(inns, inn_shapes, d, std::equal_to<>());
  if(ret) {
    auto const& xs = ret.value();
    for(auto const& x: xs) {
      if(x == 0) {
        return std::nullopt;
      }
    }
    return ret;
  }

  return std::nullopt;
}

vector<uint64_t> einsummable_t::out_shape() const {
  return vector<uint64_t>(
    join_shape.begin(),
    join_shape.begin() + out_rank);
}

vector<dtype_t> einsummable_t::inn_dtypes() const {
  vector<dtype_t> ret;
  ret.reserve(inns.size());
  for(int i = 0; i != inns.size(); ++i) {
    ret.push_back(inn_dtype(i));
  }
  return ret;
}

dtype_t einsummable_t::inn_dtype(int which_inn) const {
  auto maybe_dtype = join.inn_dtype(which_inn);
  if(!maybe_dtype) {
    throw std::runtime_error(
      "einsummable_t::inn_dtype: maybe arg not used?");
  }
  return maybe_dtype.value();
}

dtype_t einsummable_t::out_dtype() const {
  return join.out_dtype();
}

uint64_t einsummable_t::out_size() const {
  return out_nelem() * dtype_size(out_dtype());
}

uint64_t einsummable_t::out_nelem() const {
  return product(out_shape());
}

vector<vector<uint64_t>> einsummable_t::inn_shapes() const {
  vector<vector<uint64_t>> ret(inns.size());
  for(int i = 0; i != inns.size(); ++i) {
    ret[i].reserve(inns[i].size());
    for(int j: inns[i]) {
      ret[i].push_back(join_shape[j]);
    }
  }
  return ret;
}

vector<vector<int>>
einsummable_t::input_idxs(vector<int> const& join_idx) const
{
  return get_inputs_from_join(join_idx);
}

string einsummable_t::str() const {
  return make_str(inns, out_rank);
}

std::size_t einsummable_t::hash() const {
  std::hash<string>   h_str;
  std::hash<int>      h_int;
  std::hash<uint64_t> h_uint;

  std::size_t ret = h_str(str() + write_with_ss(join));

  for(auto const j: join_shape) {
    hash_combine_impl(ret, h_uint(j));
  }
  if(castable) {
    hash_combine_impl(ret, h_int(int(castable.value())));
  }
  return ret;
}

optional<vector<einsummable_t>> einsummable_t::derivative(int which_inn)
{
  if (is_contraction()) {
    return std::nullopt;
  }
  else if (join.is_unary()) {
    return ewu_derivative(which_inn);
  } else if (join.is_binary()) {
    return ewb_derivative(which_inn);
  }
}

bool einsummable_t::deri_depends_on(int which_inn)
{
  auto const& deri_op = join.derivative(which_inn);
  set<int> which_inputs = deri_op.which_inputs();

  if (which_inputs.size() == 1) {
    return which_inn == *(which_inputs.begin());
  }
}

optional<vector<einsummable_t>> einsummable_t::ewu_derivative(int which_inn)
{

  auto deri_op = join.derivative(which_inn);
  vector<einsummable_t> ret;

  if (deri_op.is_constant_of(scalar_t::one(default_dtype()))) {
      return std::nullopt;
    }

    if (deri_op.is_constant()) {
      scalar_t val = deri_op.eval({});
      einsummable_t grad(
        join_shape,
        {inverse_permute(inns[0])},
        out_rank,
        scalarop_t::make_scale(val)
      );
      ret.push_back(grad);
      return ret;
    }

    // (yhat - y)**2 => 2(yhat - y) * node_grad

    scalarop_t vjp = scalarop_t::make_vjp(deri_op);

    auto const& vjp_string = create_unary_vjp_string(inns[0], out_rank);

    auto const& [inns, out_rank] = parse_str(vjp_string);

    auto const& input_shape = backward_permute(inns[0], join_shape);

    auto const& grad_join_shape = construct_join_shape(inns, {join_shape, inn_shapes()[0]});

    if (!grad_join_shape.has_value()) {
      std::runtime_error("Invalid join shape.");
    }

    einsummable_t grad(
      grad_join_shape.value(),
      inns,
      out_rank,
      vjp
    );
    ret.push_back(grad);

    return ret;
}

optional<vector<einsummable_t>> einsummable_t::ewb_derivative(int which_inn)
{
  
  scalarop_t deri_op = join.derivative(which_inn);
  auto arg_inn = inns[which_inn];
  vector<int> jacobian_inn;
  vector<einsummable_t> ret;

  if(deri_op.is_constant_of(scalar_t::one(default_dtype()))) {
    return optional<vector<einsummable_t>>();
  }

  if(deri_op.is_constant()) {
    scalar_t val = deri_op.eval({});
    einsummable_t grad(
      join_shape,
      {inverse_permute(inns[which_inn])},
      out_rank,
      scalarop_t::make_scale(val)
    );
    
    ret.push_back(grad);
    return ret;
  }

  set<int> which_inputs = deri_op.which_inputs();

  if (which_inputs.size() > 2) {
    std::runtime_error("Invalid input size");
  }

  if (which_inputs.size() == 1) {

    int const& which_input = *(which_inputs.begin());
    int inn; 
    scalarop_t deri_fixed = deri_op;

    if (which_input == which_inn) {
      inn = which_inn;
      jacobian_inn = arg_inn;
    } else {
      inn = which_inn == 0 ? 1 : 0;
      jacobian_inn = inns[inn];
      deri_fixed.remap_inputs({ { 1, 0 } });
    }

    scalarop_t vjp = scalarop_t::make_vjp(deri_fixed);

    auto const& vjp_string = einsummable_t::create_binary_vjp_string(arg_inn, jacobian_inn);
    auto const& [inns, out_rank] = einsummable_t::parse_str(vjp_string);
    auto const& input_shape = forward_permute(inns[inn], out_shape());
    auto const& join_shape = einsummable_t::construct_join_shape(inns, {input_shape, out_shape()});

    if (!join_shape.has_value()) {
      std::runtime_error("Invalid join shape.");
    }

    einsummable_t grad (
      join_shape.value(),
      inns,
      out_rank,
      vjp
    );

    ret.push_back(grad);
    return ret;
  }

  jacobian_inn = inns[which_inn == 0 ? 1 : 0];
  auto identity = identity_permutation(jacobian_inn.size());
  auto inverse = find_permutation(jacobian_inn, identity);
  auto vjp_string = einsummable_t::make_str({inverse, identity}, jacobian_inn.size());
  auto const& [inns1, out_rank1] = einsummable_t::parse_str(vjp_string);
  auto const& arg_shape = forward_permute(arg_inn, out_shape());
  auto const& oth_shape = forward_permute(jacobian_inn, out_shape()); 
  auto const& join_shape1 = einsummable_t::construct_join_shape(inns1, {oth_shape, arg_shape});
  
  if (!join_shape1.has_value()) {
    std::runtime_error("Invalid join shape.");
  } 

  einsummable_t tmp (
    join_shape1.value(),
    inns1, 
    out_rank1,
    deri_op
  );

  ret.push_back(tmp);

  vjp_string = einsummable_t::create_binary_vjp_string(arg_inn, arg_inn);

  auto const& [inns2, out_rank2] = einsummable_t::parse_str(vjp_string);
  auto const& join_shape2 = einsummable_t::construct_join_shape(inns2, {join_shape1.value(), out_shape()});

  einsummable_t grad (
    join_shape2.value(),
    inns2, 
    out_rank2, 
    scalarop_t::make_mul()
  );

  // 2*(yhat - y)*node_grad 
  // ijk, jk -> ijk
  // ijk -> ijkl

  ret.push_back(grad);
  return ret;
}

bool einsummable_t::is_straight_elementwise() const {
  if(has_aggregation()) {
    return false;
  }

  vector<int> reference(join_shape.size());
  std::iota(reference.begin(), reference.end(), 0);
  for(auto const& inn: inns) {
    if(reference != inn) {
      return false;
    }
  }
  return true;
}

bool einsummable_t::is_permutation() const {
  return inns.size() == 1 &&
    !has_aggregation()    &&
    !has_broadcast()      &&
    join.is_identity()    ;
}

bool einsummable_t::has_aggregation() const {
  return out_rank < join_shape.size();
}

bool einsummable_t::is_contraction() const {
  return inns.size() == 2               &&
    has_aggregation()                   &&
    !has_broadcast()                    &&
    castable.value() == castable_t::add &&
    join.is_mul()                       ;
}

bool einsummable_t::has_broadcast() const {
  set<int> normal_modes;
  for(auto const& inn: inns) {
    for(auto const& i: inn) {
      if(i < out_rank) {
        normal_modes.insert(i);
      }
    }
  }
  // Example:
  //   ijkm->ijkl
  //   0124->0123
  //
  //   ijk: normal modes
  //   m:   agg mode
  //   l:   broadcast mode
  return normal_modes.size() != out_rank;
}

bool einsummable_t::is_broadcast() const {
  return inns.size() == 1 &&
    has_broadcast()       &&
    !has_aggregation()    &&
    join.is_identity()    ;
}

bool einsummable_t::is_straight_broadcast() const {
  if(!is_broadcast()) {
    return false;
  }

  vector<int> const& inn = inns[0];
  int inn_rank = inn.size();
  int n_broadcast = out_rank - inn_rank;
  for(int i = 0; i != inn_rank; ++i) {
    if(inn[i] != n_broadcast + i) {
      return false;
    }
  }

  return true;
}

std::ostream& operator<<(std::ostream& out, einsummable_t const& e) {
  out << "es[";
  out << e.join_shape[0];
  for(int i = 1; i < e.join_shape.size(); ++i) {
    out << "," << e.join_shape[i];
  }
  out << "]";

  out << e.str();

  out << e.join.to_cppstr();

  return out;
}

bool operator==(einsummable_t const& lhs, einsummable_t const& rhs) {
  if(!vector_equal(lhs.join_shape, rhs.join_shape)) {
    return false;
  }
  if(lhs.inns.size() != rhs.inns.size()) {
    return false;
  }
  for(int i = 0; i != lhs.inns.size(); ++i) {
    if(!vector_equal(lhs.inns[i], rhs.inns[i])) {
      return false;
    }
  }
  if(lhs.out_rank != rhs.out_rank) {
    return false;
  }
  return lhs.join == rhs.join && lhs.castable == rhs.castable;
}

bool operator!=(einsummable_t const& lhs, einsummable_t const& rhs) {
  return !(lhs == rhs);
}
