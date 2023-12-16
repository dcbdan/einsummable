#pragma once
#include "../base/setup.h"

struct agg_plan_t {
  agg_plan_t()
    : agg_plan_t(0,1)
  {}

  agg_plan_t(int bloc, int eloc)
    : bloc(bloc), eloc(eloc)
  {}

  set<int> source_locs() const;

  int loc_at(int i) const;

  vector<int> as_vector(int nagg) const;

  int bloc;
  int eloc;
};

// Example:
//   nloc = 4, nagg = 4
//     [0,0,0,0]
//     [1,1,1,1]
//     [2,2,2,2]
//     [3,3,3,3]
//     [0,1,0,1]
//     [2,3,2,3]
//     [0,1,2,3]
//   nloc = 2, nagg = 4
//     [0,0,0,0]
//     [1,1,1,1]
//     [0,1,0,1]
//   nloc = 4, nagg = 2
//     [0,0]
//     [1,1]
//     [2,2]
//     [3,3]
//     [0,1]
//     [2,3]
vector<agg_plan_t> gen_agg_plans(int nloc, int nagg);
