#pragma once
#include "../base/setup.h"

#include "scalarop.h"

#include "einsummable.pb.h"

struct touchdim_t {
  uint64_t d_inn;
  uint64_t d_out;
  uint64_t offset_inn;
  uint64_t offset_out;
  uint64_t size;
};

struct touch_t {
  vector<touchdim_t> selection;
  optional<castable_t> castable;
  dtype_t dtype;

  // Merge out dimensions that are fully copied.
  // So
  //   selection = [10,10,3,3,2],[20,20,0,0,20]
  //   (X[3:5,0:20] = Y[3:5,0:20] for two (10,20) matrices)
  // Becomes
  //   selection = [200,200,60,60,40]
  //   (X.flatten()[60:100] = Y.flatten()[60:100])
  touch_t simplify() const;

  static touch_t from_wire(string const& str);
  static touch_t from_proto(es_proto::Touch const& r);

  string to_wire() const;
  void to_proto(es_proto::Touch& r) const;
};

// Note: this is basically hrect_center but it
//       returns a touchdim_t
vector<touchdim_t> make_touch_selection_from_full_small(
  vector<tuple<uint64_t, uint64_t>> const& full,
  vector<tuple<uint64_t, uint64_t>> const& small);

// a = x -> y
// b = y -> z
// return (x -> z)
//
// If no part of x maps to z, none is returned
optional<touch_t> touch_compose(touch_t const& a, touch_t const& b);


