#pragma once
#include "../base/setup.h"

#include "../base/placement.h"
#include "einsummable.h"
#include "touch.h" // only for subset_t::as_touch

struct concat_t {
  concat_t(int dim, dtype_t dtype, vector<vector<uint64_t>> const& input_shapes);

  int dim;
  dtype_t dtype;
  vector<vector<uint64_t>> inn_shapes;

  int num_inns() const { return inn_shapes.size(); }

  vector<uint64_t> shape() const;

  vector<uint64_t> dim_parts() const;

  vector<tuple<uint64_t, uint64_t>> get_hrect(int which_inn) const;

  vector<uint64_t> get_offsets() const;

  // return [b,e) for the idxs that [beg,end) reaches
  tuple<int, int> get_inn_region(uint64_t beg, uint64_t end) const;
  tuple<int, int> get_inn_region(tuple<uint64_t,uint64_t> const&) const;
};

struct subset_t {
  struct subsetdim_t {
    uint64_t d_inn;
    uint64_t d_out;
    uint64_t offset;
  };

  subset_t(
    vector<tuple<uint64_t, uint64_t>> const& hrect,
    vector<uint64_t> inn_shape,
    set<int> squeeze = {},
    dtype_t dtype = default_dtype());

  subset_t(
    vector<subsetdim_t> selection,
    set<int> squeeze = {},
    dtype_t dtype = default_dtype());

  dtype_t dtype;
  vector<subsetdim_t> selection;
  set<int> squeeze;

  vector<uint64_t> inn_shape() const;
  vector<uint64_t> out_shape() const;

  vector<tuple<uint64_t, uint64_t>> get_hrect() const;

  touch_t as_touch() const;

private:
  static vector<subsetdim_t> make_selection(
    vector<tuple<uint64_t, uint64_t>> const& hrect,
    vector<uint64_t> inn_shape);
};

struct graph_t {
  // Methods to construct a graph object
  // {{{
  int insert_input(
    vector<uint64_t> shape,
    dtype_t dtype = default_dtype());

  int insert_einsummable(
    einsummable_t e,
    vector<int> inns);

  int insert_formation(
    int inn,
    bool is_save = true);

  int insert_to_complex(int inn);

  int insert_to_real(int inn);

  int insert_concat(
    int dim,
    vector<int> inns);

  int insert_subset(
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn,
    set<int> squeeze = {});

  // For each non-save node, make sure it gets marked as save if it isn't used
  // elsewhere.
  // For non-formation nodes, if the node is not used elsewhere, insert an outgoing
  // formation node with is_save = true.
  // For formation nodes with is_save = false, if there are no outgoing edges,
  // flip is_save to true.
  void set_saves();

  // TODO: Implement a way to intelligently insert formation nodes.
  //       (any non elementwise einsummable should have one output of
  //        a formation node)
  // Why would you want a formation node that is not also an output node?
  // The reason is to avoid broadcasting partial aggregation results.
  // For instance
  //   X -> A
  //     -> B
  //     -> C
  // If X is partition alot and A,B,C all use blocks at varying locations,
  // each partial will get sent to all the locations the partial gets used
  // at. Doing this instead will leaf to less communication
  //   X -> Formation Node -> A
  //                       -> B
  //                       -> C
  // as the formation node will aggregate partials of X into a
  // single location

  // }}}

  // create a partition where every node is unpartitioned
  vector<partition_t> make_singleton_partition() const;
  // create a placement where every node is unpartitioned
  // and every block is at loc 0
  vector<placement_t> make_singleton_placement() const;

  vector<uint64_t> out_shape(int id) const;

  dtype_t out_dtype(int id) const;

  vector<int> get_order() const;

  void print() const;

  void print_graphviz(std::ostream& out) const;

  vector<int> get_inputs() const;

public:

  struct input_t {
    dtype_t dtype;
    vector<uint64_t> shape;
  };

  struct formation_t {
    dtype_t dtype;
    vector<uint64_t> shape;
    bool is_save; // if this is false, it is a temporary
  };

  // An op for converting complex <-> float
  // The idea is that the last dimension is either
  // double (complex -> float) or halved (float -> complex)
  struct complexer_t {
    dtype_t dtype;
    vector<uint64_t> shape;
    // these are the out shape and out dtypes which is the join shape
    // (The join shape == the shape for annotated partitions)

    dtype_t inn_dtype() const;
    vector<uint64_t> inn_shape() const;

    bool is_to_real()   const { return dtype_is_real(dtype); }
    bool is_to_complex() const { return dtype_is_complex(dtype); }
  };
  // This has to be an op because a (5,3) partition of reals
  // can't be converted into complexes without a change of partition.
  // That is, (4,4) real parts can be viewed as (2,2) complex parts
  // for free.

  struct op_t {
  private:
    using _op_t = std::variant<
      input_t, formation_t, complexer_t,
      concat_t, subset_t, einsummable_t>;

  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t       x): op_t(_op_t(x)) {}
    op_t(formation_t   x): op_t(_op_t(x)) {}
    op_t(complexer_t   x): op_t(_op_t(x)) {}
    op_t(concat_t      x): op_t(_op_t(x)) {}
    op_t(subset_t      x): op_t(_op_t(x)) {}
    op_t(einsummable_t x): op_t(_op_t(x)) {}

    vector<uint64_t> out_shape() const;
    vector<uint64_t> shape() const;

    int out_rank() const { return this->out_shape().size(); }
    int rank() const { return this->shape().size(); }

    dtype_t out_dtype() const;

    bool is_save() const {
      return is_formation() && get_formation().is_save;
    }

    bool is_input()       const { return std::holds_alternative<input_t>(op);     }
    bool is_formation()   const { return std::holds_alternative<formation_t>(op); }
    bool is_complexer()   const { return std::holds_alternative<complexer_t>(op); }
    bool is_concat()      const { return std::holds_alternative<concat_t>(op);    }
    bool is_subset()      const { return std::holds_alternative<subset_t>(op);    }
    bool is_einsummable() const {
      return std::holds_alternative<einsummable_t>(op);
    }

    bool has_aggregation() const {
      return is_einsummable() && get_einsummable().has_aggregation();
    }

    input_t       const& get_input()     const { return std::get<input_t>(op);     }
    formation_t   const& get_formation() const { return std::get<formation_t>(op); }
    formation_t        & get_formation()       { return std::get<formation_t>(op); }
    complexer_t   const& get_complexer() const { return std::get<complexer_t>(op); }
    concat_t      const& get_concat()    const { return std::get<concat_t>(op);    }
    subset_t      const& get_subset()    const { return std::get<subset_t>(op);    }
    einsummable_t const& get_einsummable() const {
      return std::get<einsummable_t>(op);
    }
    castable_t    const& get_castable() const {
      return get_einsummable().castable.value();
    }

    _op_t op;
  };

  struct node_t {
    op_t op;
    vector<int> inns;
    set<int> outs;

    set<int> get_inns_set() const {
      return set<int>(inns.begin(), inns.end());
    }
    int num_distinct_inputs() const {
      return get_inns_set().size();
    }
  };

  vector<node_t> nodes;

private:
  int insert(op_t const& op, vector<int> inns);
};

// graph_constructor_t is for building a graph
// object with associated placements.
// graph_writer is for building a graph object
// but returning a virtual tensor object

struct graph_constructor_t {
  int insert_input(
    placement_t placement,
    dtype_t dtype = default_dtype());
  int insert_input(
    partition_t partition,
    dtype_t dtype = default_dtype());
  int insert_input(
    vector<uint64_t> shape,
    dtype_t dtype = default_dtype());

  int insert_einsummable(
    placement_t placement,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    partition_t partition,
    einsummable_t e,
    vector<int> inns);
  int insert_einsummable(
    einsummable_t e,
    vector<int> inns);

  int insert_formation(
    placement_t placement,
    int inn,
    bool is_save = true);
  int insert_formation(
    partition_t partition,
    int inn,
    bool is_save = true);
  int insert_formation(
    int inn,
    bool is_save = true);

  int insert_to_complex(
    placement_t placement,
    int inn);
  int insert_to_complex(
    partition_t partition,
    int inn);
  int insert_to_complex(
    int inn);

  int insert_to_real(
    placement_t placement,
    int inn);
  int insert_to_real(
    partition_t partition,
    int inn);
  int insert_to_real(
    int inn);

  int insert_concat(
    placement_t placement,
    int dim,
    vector<int> inns);
  int insert_concat(
    partition_t partition,
    int dim,
    vector<int> inns);
  int insert_concat(
    int dim,
    vector<int> inns);

  int insert_subset(
    placement_t placement,
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn,
    set<int> squeeze = {});
  int insert_subset(
    partition_t partition,
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn,
    set<int> squeeze = {});
  int insert_subset(
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn,
    set<int> squeeze = {});

  vector<placement_t> get_placements() const;

  graph_t graph;
  map<int, placement_t> placements;
};

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_constructor_t three_dimensional_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk,
  int num_processors);

// Same tensor dimensions as 3d matmul
// Does not explicitly set locations
graph_constructor_t straight_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk);

struct graph_writer_t {
  struct idx_t {
    struct rng {
      int64_t beg;
      int64_t end;
    };
    struct idx {
      int64_t v;
    };
    struct all {};

    idx_t(rng x): op(x) {};
    idx_t(idx x): op(x) {};
    idx_t(all x): op(x) {};

    std::variant<rng, idx, all> op;

    tuple<uint64_t, uint64_t> get(uint64_t d) const;
    bool is_squeeze() const { return std::holds_alternative<idx>(op); }
  private:
    static uint64_t to_index(uint64_t total, int64_t held);
  };

  // A full_shape_t is basically just a vector<vector<uint64_t>>.
  // It is used to store how tensor_t's are shaped. That is,
  // a tensor_t has a full_shape, which is the flattened
  // vector-of-vectors, which is how the tensor
  // appears to graph_t, and the shape, which is a vector
  // of the product of each inner-vector.
  //
  // Example:
  //   full_shape = { {50}, {10,5} }
  //   To the graph, this is always shape {50, 10, 5}. However,
  //   the tensor_t can be viewed as {50,10,5}, {50*10, 5}, {50, 10*5},
  //   {50*10*5}.
  //
  // Views in this way are a convenience to construct higher-order einsummables
  // and are not particularly flexible.
  //
  // The full_dim_t must lineup when doing einsummables.
  //
  // Example:
  //   To do ij,jk->ik with { {50}, {10,5} } left input,
  //   then the second input must have as a first full_dim_t {10,5}.
  //   If k = {30},   then to the graph, this amounts to iab,abk ->ik.
  //   If k = {15,2}, then                               iab,abcd->icd
  //
  //   ij,jk->ik would fail for
  //     { {50}, {10,5} } * { {5,10}, {30} } or
  //     { {50}, {10,5} } * { {50},   {30} }
  //   .. The full dims must line up.

  // TODO: Can views be post-processed in?
  // It would be nice if this was valid:
  //   graph_writer_t w;
  //   auto t = w.input({100,100});
  //   t = t.view({10,10,10,10});
  // The reason it is not valid is because all inputs must
  // currently be declared with the finest used dimension sizes.
  // If instead at t.view(...), it is determined
  // which dimensions need to be set finer and that should be
  // somehow propagated accross the graph.
  //
  // It could still be the case that something like {4,3} and {3,4} full_dims
  // line up, which would have to error out or a proper graph_t reshape would have
  // to be implemented...

  struct full_dim_t {
    full_dim_t(){}
    full_dim_t(vector<uint64_t> const& ps): parts(ps) {}

    vector<uint64_t> parts;
    uint64_t operator()() const { return product(parts); }
    static full_dim_t singleton(uint64_t d);
  };

  struct full_shape_t {
    full_shape_t(){}
    full_shape_t(vector<full_dim_t> const& ps): parts(ps) {}

    vector<full_dim_t> parts;

    vector<uint64_t> full() const;
    vector<uint64_t> operator()() const;

    int full_rank() const { return full().size(); }
    int rank() const { return operator()().size(); }

    vector<tuple<int,int>> get_breaks() const;
    vector<vector<uint64_t>> as_vecvec() const;

    static full_shape_t from_full(vector<uint64_t> const& ss);
    static full_shape_t from_vecvec(vector<vector<uint64_t>> const& ss);
  };

  struct tensor_t {
    tensor_t(): self(nullptr) {}

    tensor_t transpose(int i, int j) const;

    tensor_t view(full_shape_t const&) const;
    tensor_t view(vector<vector<uint64_t>> const&) const;

    tensor_t view_full() const;
    tensor_t view_full(vector<uint64_t> const&) const;

    full_shape_t const& get_shape() const { return shape; }
    int rank() const { return shape.rank(); }

    tensor_t save() const;

    int get_id() const { return id; }

    dtype_t get_dtype() const;

    tensor_t scale(scalar_t const& scalar) const;
    tensor_t scale(string const& str) const;

    tensor_t to_complex() const;
    tensor_t to_real() const;

    tensor_t to_dtype(dtype_t) const;
    tensor_t to_f16() const;
    tensor_t to_f32() const;
    tensor_t to_f64() const;

    tensor_t subset(vector<idx_t> const& idxs) const;

    // after this, the modes are 0,1,...,shape.full_rank()-1
    tensor_t physically_permute() const;

  private:
    friend class graph_writer_t;

    tensor_t(
      full_shape_t const& shape,
      int id,
      graph_writer_t* self);

    full_shape_t shape;

    vector<int> modes;
    // ^ modes.size() == shape.full_rank()

    int id;

    graph_writer_t* self;
  };

  struct to_einsummable_info_t {
    full_shape_t join_shape;

    vector<vector<int>> full_inns;

    int full_out_rank;
    int out_rank;

    full_shape_t get_out_shape() const;

    einsummable_t build_einsummable(
      scalarop_t scalarop,
      optional<castable_t> castable = std::nullopt) const;
  };

  graph_t const& get_graph() const { return graph; }

  // the core ops

  tensor_t input(
    vector<uint64_t> shape,
    dtype_t dtype = default_dtype());
  tensor_t input(
    full_shape_t shape,
    dtype_t dtype = default_dtype());

  tensor_t contraction(
    string str,
    tensor_t const& lhs,
    tensor_t const& rhs);

  tensor_t reduction(
    string str,
    castable_t castable,
    tensor_t const& inn);

  tensor_t ew(
    scalarop_t op,
    tensor_t const& inn);
  tensor_t ew( // ew = elementwise
    string str,
    scalarop_t op,
    tensor_t const& inn);
  tensor_t ew(
    string str,
    scalarop_t op,
    tensor_t const& lhs,
    tensor_t const& rhs);
  tensor_t ew(
    string str,
    scalarop_t op,
    vector<tensor_t> const& inns);

  tensor_t concat(
    int dim,
    vector<tensor_t> const& inns);

  tensor_t subset(
    vector<tuple<uint64_t, uint64_t>> const& hrect,
    set<int> squeeze,
    tensor_t const& inn);

  tensor_t to_real(tensor_t const& inn);
  tensor_t to_complex(tensor_t const& inn);

  tensor_t to_dtype(dtype_t dtype, tensor_t const& inn);
  tensor_t to_f16(tensor_t const& inn);
  tensor_t to_f32(tensor_t const& inn);
  tensor_t to_f64(tensor_t const& inn);

  // helper ops that dispatch to the core ops

  // add and mul two tensors with the same shape
  tensor_t add(
    tensor_t const& lhs,
    tensor_t const& rhs);
  tensor_t mul(
    tensor_t const& lhs,
    tensor_t const& rhs);
  tensor_t straight_bew(
    scalarop_t op,
    tensor_t const& lhs,
    tensor_t const& rhs);

  // straight elementwise scale
  tensor_t scale(
    string val,
    tensor_t const& inn);
  tensor_t scale(
    scalar_t val,
    tensor_t const& inn);

  // supports ij,jk->ik and
  //         ...ij,...jk->...ik and
  //         ...ij,jk->ik       and
  //         ij,...jk->ik
  tensor_t matmul(
    tensor_t const& lhs,
    tensor_t const& rhs);

  // take the softmax over the last dimension
  tensor_t softmax(
    tensor_t const& inn);
private:
  graph_t graph;

  int _insert_elementwise(
    string str,
    scalarop_t op,
    int id);

  // TODO
  optional<to_einsummable_info_t>
  make_einsummable_info(string str, vector<tensor_t> const& inns);

  tensor_t insert_complexer(tensor_t inn);
};

bool operator==(
  graph_writer_t::full_dim_t const& lhs,
  graph_writer_t::full_dim_t const& rhs);
bool operator!=(
  graph_writer_t::full_dim_t const& lhs,
  graph_writer_t::full_dim_t const& rhs);

bool operator==(
  graph_writer_t::full_shape_t const& lhs,
  graph_writer_t::full_shape_t const& rhs);
bool operator!=(
  graph_writer_t::full_shape_t const& lhs,
  graph_writer_t::full_shape_t const& rhs);

std::ostream& operator<<(
  std::ostream&,
  graph_writer_t::full_shape_t const&);
