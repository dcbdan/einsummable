#include "fill.h"

fill_t fill_t::make_constant(scalar_t value, vector<uint64_t> const& shape) {
  return fill_t(constant_t { .value = value, .shape = shape });
}

fill_t fill_t::make_square_lowertri(dtype_t d, uint64_t n) {
  return fill_t(lowertri_t {
    .dtype = d,
    .nrow = n,
    .ncol = n,
    .start = 0
  });
}

fill_t fill_t::select(vector<tuple<uint64_t, uint64_t>> const& hrect) const {
  if(is_constant()) {
    return *this;
  } else if(is_lowertri()) {
    // TODO
    throw std::runtime_error("fill_t::select for lowertri: not implemented");
  } else {
    throw std::runtime_error("missing fill case");
  }
}

dtype_t fill_t::dtype() const {
  if(is_constant()) {
    return get_constant().value.dtype;
  } else if(is_lowertri()) {
    return get_lowertri().dtype;
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
    c->set_dtype(write_with_ss(lowertri.dtype));
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
      .dtype = parse_with_ss<dtype_t>(l.dtype()),
      .nrow = l.ncol(),
      .ncol = l.nrow(),
      .start = l.start()
    });
  } else {
    throw std::runtime_error("should not reach");
  }
}

