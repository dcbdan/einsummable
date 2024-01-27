#pragma once
#include "../base/setup.h"

#include "scalarop.h"
#include "touch.h"

struct select_t {
  struct selectdim_t {
    uint64_t d_inn;
    uint64_t offset_inn;
    uint64_t offset_out;
    uint64_t size;
  };

  using inn_region_t = vector<selectdim_t>;

  dtype_t dtype;
  vector<uint64_t> out_shape;
  vector<inn_region_t> inn_regions;

  select_t(
    dtype_t dtype,
    vector<uint64_t> const& out_shape,
    vector<inn_region_t> const& inn_regions);

  vector<touch_t> as_touches() const;
  touch_t as_touch(int which) const;

  vector<uint64_t> // a point with respect to the output tensor
  wrt_output_point(
    vector<uint64_t> const& inn_point, // a point with respect to an input tensor
    int which_inn) const; // which input tensor

  hrect_t wrt_output_hrect(hrect_t const& inn_hrect, int which_inn) const;

  // For each input that touches into the out_hrect,
  //   return the hrect portion of the input tensor and which input
  vector<tuple<hrect_t, int>>
  collect(hrect_t out_hrect) const;

  hrect_t wrt_output_inn_hrect(int which_input) const;
  hrect_t wrt_input_inn_hrect(int which_input) const;

  vector<uint64_t> inn_shape(int which_input) const;
};

select_t make_concat(
  int dim,
  dtype_t dtype,
  vector<vector<uint64_t>> const& input_shapes);

select_t make_subset(
  dtype_t dtype,
  vector<tuple<uint64_t, uint64_t>> const& hrect,
  vector<uint64_t> inn_shape);

