#include "einsummable.h"

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

  if(!valid_inns_out(inns, out_rank)) {
    throw std::runtime_error("einsummable: invalid inns, out rank");
  }
}

einsummable_t einsummable_t::merge_adjacent_dims() const {
  int join_rank = join_shape.size();

  auto count_adjacent = [](int r, vector<int> const& is) {
    auto iter = std::find(is.begin(), is.end(), r);
    if(iter == is.end()) {
      return -1;
    }
    int c = r;
    for(; iter != is.end(); ++iter) {
      if(*iter == c) {
        c++;
      } else {
        break;
      }
    }
    return c-r;
  };

  // for each index, return the new index and the count and the size
  map<int, tuple<int, int>> merges;

  int r = 0;
  int new_r = 0;
  while(r != join_rank) {
    int n_adj = count_adjacent(r, inns[0]);
    for(int i = 1; i != inns.size(); ++i) {
      if(n_adj == 1) {
        break;
      }
      int n = count_adjacent(r, inns[i]);
      if(n != -1) {
        if(n_adj == -1) {
          n_adj = n;
        } else {
          n_adj = std::min(n_adj, n);
        }
      }
    }
    if(r < out_rank) {
      // Consider 0123->012
      // n_adj @ 0 = 4 but it can only be 3
      n_adj = std::min(n_adj, out_rank - r);
    }

    merges.insert({r, {new_r, n_adj}});
    r += n_adj;
    new_r += 1;
  }

  if(merges.size() == join_rank) {
    return *this;
  }

  vector<uint64_t> new_join_shape;
  {
    int r = 0;
    while(r != join_rank) {
      auto const& [_, n_adj] = merges.at(r);

      uint64_t sz = 1;
      for(int i = r; i != r + n_adj; ++i) {
        sz *= join_shape[r];
      }

      new_join_shape.push_back(sz);

      r += n_adj;
    }
  }

  int new_out_rank = std::get<0>(merges.at(out_rank));

  vector<vector<int>> new_inns;
  for(auto const& inn: inns) {
    new_inns.emplace_back();
    vector<int>& new_inn = new_inns.back();

    int i = 0;
    while(i != inn.size()) {
      auto const& [new_r, n_adj] = merges.at(inn[i]);
      new_inn.push_back(new_r);
      i += n_adj;
    }
  }

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
  string str)
{
  auto [inns,_] = einsummable_t::parse_str(str);
  return einsummable_t({di, dk, dj}, inns, 2, scalarop_t::make_mul(), castable_t::add);
}

einsummable_t einsummable_t::from_matmul(uint64_t di, uint64_t dj, uint64_t dk) {
  return from_matmul_ss(di, dj, dk);
}

einsummable_t einsummable_t::from_matmul_ss(uint64_t di, uint64_t dj, uint64_t dk) {
  return _einsummable_matmul_helper(di, dj, dk, "ij,jk->ik");
}

einsummable_t einsummable_t::from_matmul_ts(uint64_t di, uint64_t dj, uint64_t dk) {
  return _einsummable_matmul_helper(di, dj, dk, "ji,jk->ik");
}

einsummable_t einsummable_t::from_matmul_st(uint64_t di, uint64_t dj, uint64_t dk) {
  return _einsummable_matmul_helper(di, dj, dk, "ij,kj->ik");
}

einsummable_t einsummable_t::from_matmul_tt(uint64_t di, uint64_t dj, uint64_t dk) {
  return _einsummable_matmul_helper(di, dj, dk, "ji,kj->ik");
}

einsummable_t einsummable_t::with_new_shape(
  einsummable_t const& e,
  vector<uint64_t> const& new_join_shape)
{
  if(e.join_shape.size() != new_join_shape.size()) {
    throw std::runtime_error("einsummable_t::with_new_shape");
  }
  return einsummable_t(new_join_shape, e.inns, e.out_rank, e.join, e.castable);
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

bool einsummable_t::valid_inns_out(
  vector<vector<int>> const& inns,
  int out_rank)
{
  if(out_rank == 0) {
    // output scalars not supported
    return false;
  }

  set<int> all_ids;

  int join_rank = -1;
  for(vector<int> const& inn: inns) {
    if(inn.size() == 0) {
      // scalars not supported
      return false;
    }

    all_ids.insert(inn.begin(), inn.end());

    {
      set<int> _inns(inn.begin(), inn.end());
      if(_inns.size() != inn.size()) {
        // inn index cannot contain duplicates
        return false;
      }
    }
    for(int const& i: inn) {
      join_rank = std::max(join_rank, i);
      if(i < 0) {
        // inn index out of range
        return false;
      }
    }
  }
  join_rank += 1;

  for(int i = 0; i != join_rank; ++i) {
    if(all_ids.count(i) == 0) {
      // all index from 0, ... join rank - 1 must be included
      return false;
    }
  }

  int agg_cnt = out_rank - 1;
  for(vector<int> const& inn: inns) {
    for(int const& i: inn) {
      if(i >= out_rank) {
        if(i == agg_cnt + 1) {
          agg_cnt += 1;
        } else if(i <= agg_cnt) {
          // good
        } else {
          // the aggs must be in order!
          return false;
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
  return construct_join_shape_(inns, inn_shapes, d, std::equal_to<>());
}

vector<uint64_t> einsummable_t::out_shape() const {
  return vector<uint64_t>(
    join_shape.begin(),
    join_shape.begin() + out_rank);
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
  if(join_shape.size() > 26) {
    throw std::runtime_error("not enough letters; what are you doing");
  }

  vector<char> letters(join_shape.size());
  std::iota(letters.begin(), letters.end(), 'a');

  auto words = get_inputs_from_join(letters);

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
};

bool einsummable_t::is_straight_elementwise() const {
  vector<int> reference(join_shape.size());
  std::iota(reference.begin(), reference.end(), 0);
  for(auto const& inn: inns) {
    if(reference != inn) {
      return false;
    }
  }
  return true;
}

bool einsummable_t::has_aggregation() const {
  return out_rank < join_shape.size();
}

bool einsummable_t::is_contraction() const {
  return inns.size() == 2               &&
    has_aggregation()                   &&
    castable.value() == castable_t::add &&
    join.is_mul();
}

std::ostream& operator<<(std::ostream& out, einsummable_t const& e) {
  out << "es[";
  out << e.join_shape[0];
  for(int i = 1; i < e.join_shape.size(); ++i) {
    out << "," << e.join_shape[i];
  }
  out << "]";

  out << e.str();

  return out;
}

std::ostream& operator<<(std::ostream& out, castable_t const& c) {
  if(c == castable_t::add) {
    out << "+";
  } else if(c == castable_t::mul) {
    out << "x";
  } else if(c == castable_t::min) {
    out << "v";
  } else if(c == castable_t::max) {
    out << "^";
  } else {
    throw std::runtime_error("should not reach");
  }

  return out;
}

std::istream& operator>>(std::istream& inn, castable_t& castable) {
  char c;
  inn.read(&c, 1);

  if(c == '+') {
    castable = castable_t::add;
  } else if(c == 'x') {
    castable = castable_t::mul;
  } else if(c == 'v') {
    castable = castable_t::min;
  } else if(c == '^') {
    castable = castable_t::max;
  } else {
    throw std::runtime_error("should not reach");
  }

  return inn;
}

std::ostream& operator<<(std::ostream& out, optional<castable_t> const& maybe_c) {
  if(maybe_c) {
    out << maybe_c.value();
  } else {
    out << ":";
  }
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
