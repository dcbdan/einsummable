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
vector<placement_t> alocate01(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved);

vector<placement_t> alocate02(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved,
  map<int, vtensor_t<int>> const& fixed_pls,
  vector<tuple<int,int>> const& equal_pls);

// contractions:    load balance across all locs
// everything else: greedily pick cheapest
// cost:            number of bytes moved
vector<placement_t> alocate03(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool with_goofy_topology = false);

// The idea is that we can add a one-directional link
// between any pair of locations. For locations that don't
// have a direct connection, we can add a routing path
// through a single intermediate location.
//
// A link has a width which is proportional to that links
// bandwidth. Larger bandwidth = less time spent moving
// a given amount of bytes.
//
// The cost of a series of moves is the maximum time
// spent moving across all links.
struct topology_t {
  topology_t();

  // A width of 5 is 5x more bandwidth than a width of 1
  void insert_link(int src, int dst, int width);
  void insert_path(int src, int dummy, int dst);

  void add_move(int src, int dst, uint64_t bytes);

  uint64_t cost_moves(vector<tuple<int, int, uint64_t>> const& moves) const;

  uint64_t cost() const;
  void reset();
private:
  int width_factor;

  struct link_t {
    int src;
    int dst;
    int width;
  };
  struct path_t {
    int src;
    int mid;
    int dst;
  };
  vector<uint64_t> cost_per_link;

  vector<link_t> links;
  map<int, map<int,int>> src_to;
  map<int, map<int,int>> dst_to;

  vector<path_t> paths;

  // return index into `links`
  optional<int> get_link(int src, int dst) const;

  // return the intermediate location
  optional<int> get_path(int src, int dst) const;

  int get_link_width(int which_link) const;
  int get_link_width(int src, int dst) const;

  // get a list of links
  vector<int> get_links_for_move(int src, int dst) const;

  uint64_t cost_at_link(int which_link, uint64_t bytes) const;

  bool is_costing() const;

  void update_width_factor(int width);
};

// Always load balance everything, considering all permutations,
//   similar to alocate03.
// Cost: time to do all moves for one relation according
//       to topology
vector<placement_t> alocate04(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  topology_t topology);


