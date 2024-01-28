#include "fill.h"

std::variant<fill_t::constant_t, fill_t::lowertri_t>
fill_t::build_from_lowertri(fill_t::lowertri_t const& l)
{
  if(l.lower.dtype != l.upper.dtype) {
    throw std::runtime_error("invalid lowertri values");
  }

  if(l.lower == l.upper) {
    // Unlikely that l.lower == l.upper, but just in case
    return constant_t {
      .value = l.lower,
      .shape = vector<uint64_t>{l.nrow, l.ncol}
    };
  } else if(l.start <= 1 - int64_t(l.ncol)) {
    return constant_t {
      .value = l.lower,
      .shape = vector<uint64_t>{l.nrow, l.ncol}
    };
  } else if(l.start >= int64_t(l.nrow)) {
    return constant_t {
      .value = l.upper,
      .shape = vector<uint64_t>{l.nrow, l.ncol}
    };
  } else {
    return l;
  }
}

fill_t fill_t::make_constant(scalar_t value, vector<uint64_t> const& shape) {
  return fill_t(constant_t { .value = value, .shape = shape });
}

fill_t fill_t::make_square_lowertri(dtype_t d, uint64_t n) {
  return fill_t(lowertri_t {
    .lower = scalar_t::one(d),
    .upper = scalar_t::zero(d),
    .nrow = n,
    .ncol = n,
    .start = 0
  });
}

fill_t::lowertri_t
fill_t::lowertri_t::select(hrect_t const& hrect) const
{
  if(hrect.size() != 2) {
    throw std::runtime_error("invalid lowertri");
  }
  auto const& [i_beg, i_end] = hrect[0];
  auto const& [j_beg, j_end] = hrect[1];
  if(i_beg >= i_end || i_end > nrow) {
    throw std::runtime_error("invalid row selection");
  }
  if(j_beg >= j_end || j_end > ncol) {
    throw std::runtime_error("invalid col selection");
  }
  return lowertri_t {
    .lower = lower,
    .upper = upper,
    .nrow = i_end - i_beg,
    .ncol = j_end - j_beg,
    .start = start - int64_t(i_beg)
  };
}

fill_t fill_t::select(hrect_t const& hrect) const {
  // Assumption: hrect is a valid region of this fill's shape
  if(is_constant()) {
    auto const& [value,_] = get_constant();
    return fill_t(constant_t {
      .value = value,
      .shape = hrect_shape(hrect)
    });
  } else if(is_lowertri()) {
    return fill_t(get_lowertri().select(hrect));
  } else {
    throw std::runtime_error("missing fill case");
  }
}

dtype_t fill_t::dtype() const {
  if(is_constant()) {
    return get_constant().value.dtype;
  } else if(is_lowertri()) {
    return get_lowertri().lower.dtype;
  } else {
    throw std::runtime_error("missing fill case");
  }
}
vector<uint64_t> fill_t::shape() const {
  if(is_constant()) {
    return get_constant().shape;
  } else if(is_lowertri()) {
    auto const& l = get_lowertri();
    return vector<uint64_t>{l.nrow, l.ncol};
  } else {
    throw std::runtime_error("missing fill case");
  }
}

string fill_t::to_wire() const {
  es_proto::Fill f;
  to_proto(f);
  string ret;
  f.SerializeToString(&ret);
  return ret;
}

void fill_t::to_proto(es_proto::Fill& f) const {
  if(is_constant()) {
    auto const& constant = get_constant();

    es_proto::Constant* c = f.mutable_constant();
    c->set_value(write_with_ss(constant.value));
    for(auto const& dim: constant.shape) {
      c->add_shape(dim);
    }
  } else if(is_lowertri()) {
    auto const& lowertri = get_lowertri();
    es_proto::Lowertri* c = f.mutable_lowertri();
    c->set_lower(write_with_ss(lowertri.lower));
    c->set_upper(write_with_ss(lowertri.upper));
    c->set_ncol(lowertri.ncol);
    c->set_nrow(lowertri.nrow);
    c->set_start(lowertri.start);
  } else {
    throw std::runtime_error("should not reach: missing fill case..");
  }
}

fill_t fill_t::from_wire(string const& str) {
  es_proto::Fill f;
  if(!f.ParseFromString(str)) {
    throw std::runtime_error("could not parse fill");
  }
  return from_proto(f);
}

fill_t fill_t::from_proto(es_proto::Fill const& p) {
  if(p.has_constant()) {
    auto const& c = p.constant();
    auto ds = c.shape();
    return fill_t(constant_t {
      .value = parse_with_ss<scalar_t>(c.value()),
      .shape = vector<uint64_t>(ds.begin(), ds.end())
    });
  } else if(p.has_lowertri()) {
    auto const& l = p.lowertri();
    return fill_t(lowertri_t {
      .lower = parse_with_ss<scalar_t>(l.lower()),
      .upper = parse_with_ss<scalar_t>(l.upper()),
      .nrow = l.ncol(),
      .ncol = l.nrow(),
      .start = l.start()
    });
  } else {
    throw std::runtime_error("should not reach");
  }
}

