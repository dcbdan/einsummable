#pragma once
#include "../base/setup.h"

#include "aggplan.h"
#include "relationwise.h"

// Note: Group all formation nodes with preceeding joins.
// For each node or node+formation in graph order:
//    If it is an input node:
//      round-robin assign locations
//    Otherwise:
//      For each agg,
//        Pick the location that leads to the lowest communication
//        across all agg plans
// Possible agg plans:
//   See aggplan.h
// Cost of an agg plan:
//   The cost to move inputs to site + flops.
//   Here, a move cost is flops_per_byte_moved * bytes moved
vector<placement_t> alocate01(graph_t const&             graph,
                              vector<partition_t> const& parts,
                              int                        nlocs,
                              uint64_t                   flops_per_byte_moved);

vector<placement_t> alocate02(graph_t const&                  graph,
                              vector<partition_t> const&      parts,
                              int                             nlocs,
                              uint64_t                        flops_per_byte_moved,
                              map<int, vtensor_t<int>> const& fixed_pls,
                              vector<tuple<int, int>> const&  equal_pls);
