#include "einsummable.h"

inline
einsummable_t _einsummable_matmul_helper(
  uint64_t di, uint64_t dj, uint64_t dk,
  vector<vector<int>> const& inns)
{
  return einsummable_t {
    .join_shape = {di, dk, dj},
    .inns = inns,
    .out_rank = 2,
    .join = scalarop_t::make_mul(),
    .castable = castable_t::add
  };
}

einsummable_t einsummable_t::from_matmul(uint64_t di, uint64_t dj, uint64_t dk) {
  return from_matmul_ss(di, dj, dk);
}

einsummable_t einsummable_t::from_matmul_ss(uint64_t di, uint64_t dj, uint64_t dk) {
  // ij,jk->ik
  // 02 21  01
  return _einsummable_matmul_helper(di, dj, dk, { {0, 2}, {2, 1} });
}

einsummable_t einsummable_t::from_matmul_ts(uint64_t di, uint64_t dj, uint64_t dk) {
  // ji,jk->ik
  // 20 21  01
  return _einsummable_matmul_helper(di, dj, dk, { {2, 0}, {2, 1} });
}

einsummable_t einsummable_t::from_matmul_st(uint64_t di, uint64_t dj, uint64_t dk) {
  // ij,kj->ik
  // 02,12  01
  return _einsummable_matmul_helper(di, dj, dk, { {0, 2}, {1, 2} });
}

einsummable_t einsummable_t::from_matmul_tt(uint64_t di, uint64_t dj, uint64_t dk) {
  // ji,kj->ik
  // 20 12  01
  return _einsummable_matmul_helper(di, dj, dk, { {2, 0}, {1, 2} });
}


einsummable_t einsummable_t::with_new_shape(
  einsummable_t const& e,
  vector<uint64_t> const& new_join_shape)
{
  if(e.join_shape.size() != new_join_shape.size()) {
    throw std::runtime_error("einsummable_t::with_new_shape");
  }
  return einsummable_t {
    .join_shape = new_join_shape,
    .inns       = e.inns,
    .out_rank   = e.out_rank,
    .join       = e.join,
    .castable   = e.castable
  };
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

std::ostream& operator<<(std::ostream& out, einsummable_t const& e) {
  out << "es[";
  out << e.join_shape[0];
  for(int i = 1; i < e.join_shape.size(); ++i) {
    out << "," << e.join_shape[i];
  }
  out << "]";

  out << "\"" << e.str() << "\"";

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

std::ostream& operator<<(std::ostream& out, optional<castable_t> const& maybe_c) {
  if(maybe_c) {
    out << maybe_c.value();
  } else {
    out << ":";
  }
  return out;
}

