#pragma once
#include "setup.h"

// -------------------
// |A   |A           |
// |----|------------|
// |A   |A  |A       |
// |    |   | B      |
// |    |   |        |
// -------------------
// A: full
// B: small
//
//  --------------
//  |   |Output  |
//  |   |        |
//  |   |        |
//  --------------
// It is an error if the small is not within the big
vector<tuple<uint64_t, uint64_t>>
hrect_center(
  vector<tuple<uint64_t, uint64_t>> const& full,
  vector<tuple<uint64_t, uint64_t>> const& small);

vector<uint64_t> hrect_shape(
  vector<tuple<uint64_t, uint64_t>> const& hrect);

// This is an error on empty intersection
vector<tuple<uint64_t, uint64_t>>
hrect_intersect(
  vector<tuple<uint64_t, uint64_t>> const& lhs,
  vector<tuple<uint64_t, uint64_t>> const& rhs);

optional<tuple<uint64_t, uint64_t>>
interval_intersect(
  tuple<uint64_t, uint64_t> const& lhs,
  tuple<uint64_t, uint64_t> const& rhs);

