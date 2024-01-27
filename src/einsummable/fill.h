#pragma once
#include "../base/setup.h"

#include "einsummable.pb.h"
#include "scalarop.h"

// fill_t is used to describe a variety of constant tensors.
// (for now, only tensors fill with a single constant value are supported)
struct fill_t {
  struct constant_t {
    scalar_t value;
    vector<uint64_t> shape;
  };

  // for i in range(nrow):
  //   for j in range(ncol):
  //     ret[i,j] = 1 if (i <= j + start) else 0
  struct lowertri_t {
    dtype_t dtype;
    uint64_t nrow;
    uint64_t ncol;
    int64_t start;
  };

  fill_t(constant_t const& c): op(c) {}
  fill_t(lowertri_t const& l): op(l) {}

  static fill_t make_constant(scalar_t value, vector<uint64_t> const& shape);
  static fill_t make_square_lowertri(dtype_t d, uint64_t n);

  fill_t select(vector<tuple<uint64_t, uint64_t>> const& hrect) const;

  dtype_t dtype() const;
  vector<uint64_t> shape() const;

  bool is_constant() const { return std::holds_alternative<constant_t>(op); }
  bool is_lowertri() const { return std::holds_alternative<lowertri_t>(op); }

  constant_t const& get_constant() const { return std::get<constant_t>(op); }
  lowertri_t const& get_lowertri() const { return std::get<lowertri_t>(op); }

  uint64_t size() const { return nelem() * dtype_size(dtype()); }
  uint64_t nelem() const { return product(shape()); }

  string to_wire() const;
  void to_proto(es_proto::Fill& f) const;

  static fill_t from_wire(string const& str);
  static fill_t from_proto(es_proto::Fill const& p);

private:
  std::variant<constant_t, lowertri_t> op;
};


