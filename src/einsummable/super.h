#pragma once
#include "memgraph.h"

#include <unordered_map>

struct super_graph_t {
  struct node_t {
    vector<int> ops;
    set<int> inns;
    set<int> outs;
  };

  vector<node_t> nodes;
  std::unordered_map<int, int> mid_to_sid;

  int insert(
    memgraph_t const& memgraph,
    vector<int> op_mids);

  void print_graphviz(std::ostream& out);
};

super_graph_t
create_super_graph(memgraph_t const& mg);

