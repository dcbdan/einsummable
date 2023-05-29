#pragma once
#include "../base/setup.h"

#include "graph.h"

struct twolayergraph_t {
  // Return
  //   1. The mapping from graph id to twolayer join ids
  //   2. A suggestion for which join ids should have the same placement
  //   3. The two layer graph constructed from the graph and it's partitions
  static
  tuple<
    vector<tensor_t<int>>,
    equal_items_t<int>,
    twolayergraph_t>
  make(graph_t const& graph);

  using rid_t = int; // refinement ids
  using jid_t = int; // join ids

  // An agg unit is something that will get summed.
  // So if Y = X1 + X2 + X3 + X4 at locations
  //       0    0   1    1    2
  // then X1 is not moved,
  //      X2 and X3 are summed at location 1 and moved
  //      X4 is moved.
  // An agg unit depends on a set of join ids (X1,X2,X3,X4)
  // but not neccessarily on all elements of the join ids.
  // The size variable how much of each input it takes.
  struct agg_unit_t {
    uint64_t size;
    vector<jid_t> deps;
  };

  // A refinement is called such because when constructing from
  // a graph object, this is one block of the refinement partition
  // of all usages of some graph node.
  //
  // Consider Y = UnaryElementwiseOp(X) where X is partitioned into 2x2 blocks
  // and Y is a formation and the only usage, partitioned into 3x3 blocks.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   just Unary(X[0,0]) and the size is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit has one of the
  //   Unary(X[i,j]) blocks for i,j in [0,1]x[0,1]. The size of each agg unit block
  //   is roughly 1/4 the size of the Y[1,1] block.
  // Since this is an elementwise op, each agg unit is just a copy and does not
  //   actual summation.
  //
  // Consider instead Y = (ijk->ij, X) where X is partition into 2x2x3 blocks
  // and Y is partitioned again into 3x3 blocks. Everything holds the same
  // as before except each agg unit has 3 inputs.
  // The refinement of Y[0,0] has one agg unit. That agg unit has as dependency
  //   (ijk->ij, X[0,0,k]) for k=0,1,2 and the size is the size of Y[0,0] block.
  // The refinement of Y[1,1] has four agg units. Each agg unit represents some i,j
  //   and that agg unit has blocks (ijk->ij, X[i,j,k]) for k=0,1,2.
  //   The size of each agg unit block is roughly 1/4 the size of the Y[1,1] block.
  struct refinement_t {
    vector<agg_unit_t> units;
    set<jid_t> outs;
  };

  // A join cannot complete until each agg in each dependent refinement
  // is completed.
  struct join_t {
    uint64_t flops;
    vector<rid_t> deps;
    set<rid_t> outs;
  };

  vector<join_t> joins;
  vector<refinement_t> refinements;

  struct twolayerid_t {
    int id;       // either join or refinemet id
    bool is_join; // depending on is_join
  };
  vector<twolayerid_t> order;

  // Count the number of elements moved if jid is
  // set to have location loc. Only jids dependent
  // on id will be accessed in locations.
  uint64_t count_elements_to(
    vector<int> const& locations,
    int jid,
    int loc) const;
  // Note: I don't like this method.
  // Ask: What is being moved?

  void print_graphviz(std::ostream&);
  void print_graphviz(
    std::ostream&,
    std::function<string(int)> const& jid_to_color);

  int num_input_joins() const;

private:
  jid_t insert_join(uint64_t flops, vector<rid_t> const& deps);
  rid_t insert_empty_refinement();
  void add_agg_unit(rid_t rid, uint64_t size, vector<jid_t> deps);
};

template <typename T>
struct twolayer_join_holder_t {
  // for each graph node, the tids
  vector<tensor_t<int>> const g_to_tl;

  // for each two layer join node the graph and block id
  vector<tuple<int,int>> const tl_to_g;

  // An item for every join object
  vector<T> items;

  vector<int> block_shape(int gid) const {
    return g_to_tl[gid].get_shape();
  }
  int block_size(int gid) const {
    return product(block_shape(gid));
  }

  void set_at_gid(int gid, vector<T> const& vs) {
    tensor_t<int> const& to_tl = g_to_tl[gid];
    vector<int> const& to_tl_vec = to_tl.get();
    if(to_tl_vec.size() != vs.size()) {
      throw std::runtime_error("set_at_gid invalid size");
    }
    for(int bid = 0; bid != to_tl_vec.size(); ++bid) {
      int const& jid = to_tl_vec[bid];
      items[jid] = vs[bid];
    }
  }

  tensor_t<T> get_tensor_at_gid(int gid) const {
    tensor_t<int> const& to_tl = g_to_tl[gid];
    vector<int> const& to_tl_vec = to_tl.get();

    tensor_t<T> ret(to_tl.get_shape());
    vector<T>& new_vs_vec = ret.get();

    for(int bid = 0; bid != to_tl_vec.size(); ++bid) {
      int const& jid = to_tl_vec[bid];
      new_vs_vec[bid] = items[jid];
    }

    return ret;
  }

  vector<T> get_vector_at_gid(int gid) const {
    return get_tensor_at_gid(gid).get();
  }

  T& get_at_gid(int gid, int bid) {
    int const& jid = g_to_tl[gid].get()[bid];
    return items[jid];
  }
  T const& get_at_gid(int gid, int bid) const {
    int const& jid = g_to_tl[gid].get()[bid];
    return items[jid];
  }

  T& get_at_gid(int gid, vector<int> const& idxs) {
    int const& jid = g_to_tl[gid].at(idxs);
    return items[jid];
  }
  T const& get_at_gid(int gid, vector<int> const& idxs) const {
    int const& jid = g_to_tl[gid].at(idxs);
    return items[jid];
  }

  vector<tensor_t<T>> as_graph_repr() {
    vector<tensor_t<T>> ret;
    for(int gid = 0; gid != g_to_tl.size(); ++gid) {
      ret.push_back(get_tensor_at_gid(gid));
    }
    return ret;
  }

  static vector<tuple<int,int>> make_tl_to_g(
    vector<tensor_t<int>> const& g_to_tl)
  {
    int nj = 0;
    for(tensor_t<int> const& t: g_to_tl) {
      vector<int> const& v = t.get();
      if(v.size() == 0) {
        throw std::runtime_error("g_to_l must have nonempty tensors");
      }
      nj = std::max(
        nj,
        *std::max_element(v.begin(), v.end()));
    }

    // here: nj is the size of the largest access tl_to_g will have
    nj += 1;
    // here: nj is the size of tl_to_g

    vector<tuple<int,int>> tl_to_g(nj);

    for(int gid = 0; gid != g_to_tl.size(); ++gid) {
      tensor_t<int> const& t = g_to_tl[gid];
      vector<int> const& v = t.get();

      for(int bid = 0; bid != v.size(); ++bid) {
        int const& jid = v[bid];
        tl_to_g[jid] = {gid, bid};
      }
    }

    return tl_to_g;
  }

  static twolayer_join_holder_t make(
    vector<tensor_t<int>> const& g_to_tl,
    T const& default_value)
  {
    auto tl_to_g = make_tl_to_g(g_to_tl);

    return twolayer_join_holder_t {
      .g_to_tl = g_to_tl,
      .tl_to_g = tl_to_g,
      .items = vector<T>(tl_to_g.size(), default_value)
    };
  }

  static twolayer_join_holder_t make(
    vector<tensor_t<int>> const& g_to_tl,
    vector<tensor_t<T>>   const& tensors)
  {
    vector<tuple<int,int>> tl_to_g = make_tl_to_g(g_to_tl);

    vector<T> items(tl_to_g.size());

    for(int jid = 0; jid != tl_to_g.size(); ++jid) {
      auto const& [gid,bid] = tl_to_g[jid];
      items[jid] = tensors[gid].get()[bid];
    }

    return twolayer_join_holder_t {
      .g_to_tl = g_to_tl,
      .tl_to_g = tl_to_g,
      .items = items
    };
  }

  static vector<T> get_items(
    vector<tensor_t<int>> const& g_to_tl,
    vector<tensor_t<T>>   const& tensors)
  {
    return make(g_to_tl, tensors).items;
  }
};

vector<int> graph_locations_to_twolayer(
  graph_t const& graph,
  vector<tensor_t<int>> const& g_to_tl);

void set_locations_from_twolayer(
  graph_t& graph,
  vector<tensor_t<int>> const& g_to_tl,
  vector<int> const& locs);
