#pragma once
#include "../base/setup.h"

#include "../base/placement.h"
#include "einsummable.h"
#include "touch.h" // only for subset_t::as_touch
#include "relation.h"

// fill_t is used to describe a variety of constant tensors.
// (for now, only tensors fill with a single constant value are supported)
struct fill_t {
  scalar_t value;
  vector<uint64_t> shape;
};

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

  int insert_fill(fill_t const& fill);

  int insert_concat(
    int dim,
    vector<int> inns);

  int insert_subset(
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn);

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

  vector<int> backprop(int out, vector<int> weights);

  // create a partition where every node is unpartitioned
  vector<partition_t> make_singleton_partition() const;
  // create a placement where every node is unpartitioned
  // and every block is at loc 0
  vector<placement_t> make_singleton_placement() const;

  vector<uint64_t> out_shape(int id) const;

  dtype_t out_dtype(int id) const;

  vector<int> get_order() const;

  vector<int> get_reverse_order() const;

  void print() const;

  void print_graphviz(
    std::ostream& out,
    map<int, string> get_color = map<int, string>{}) const;
  void print_graphviz(
    std::ostream& out,
    vector<partition_t> const& p,
    map<int, string> get_color = map<int,string>{}) const;

  vector<int> get_inputs() const;

public:

  struct input_t {
    dtype_t dtype;
    vector<uint64_t> shape;
  };

  struct formation_t {
    dtype_t dtype;
    vector<uint64_t> shape;
  };

  // An op for converting complex <-> float
  // The idea is that the last dimension is either
  // doubled (complex -> float) or halved (float -> complex)
  struct complexer_t {
    dtype_t dtype; // out dtype
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
      fill_t, select_t, einsummable_t>;

  public:
    op_t(_op_t op, bool is_save);

    op_t(input_t       x, bool s = false): op_t(_op_t(x), s) {}
    op_t(formation_t   x, bool s = false): op_t(_op_t(x), s) {}
    op_t(complexer_t   x, bool s = false): op_t(_op_t(x), s) {}
    op_t(fill_t        x, bool s = false): op_t(_op_t(x), s) {}
    op_t(select_t      x, bool s = false): op_t(_op_t(x), s) {}
    op_t(einsummable_t x, bool s = false): op_t(_op_t(x), s) {}

    vector<uint64_t> out_shape() const;
    vector<uint64_t> shape() const;

    int out_rank() const { return this->out_shape().size(); }
    int rank() const { return this->shape().size(); }

    dtype_t out_dtype() const;

    bool is_save() const { return is_save_; }
    void set_save(bool s);

    bool is_input()       const { return std::holds_alternative<input_t>(op);     }
    bool is_formation()   const { return std::holds_alternative<formation_t>(op); }
    bool is_complexer()   const { return std::holds_alternative<complexer_t>(op); }
    bool is_fill()        const { return std::holds_alternative<fill_t>(op);      }
    bool is_select()      const { return std::holds_alternative<select_t>(op);    }
    bool is_einsummable() const {
      return std::holds_alternative<einsummable_t>(op);
    }

    bool has_aggregation() const {
      return is_einsummable() && get_einsummable().has_aggregation();
    }
    bool is_contraction() const {
      return is_einsummable() && get_einsummable().is_contraction();
    }

    input_t       const& get_input()       const { return std::get<input_t>(op);     }
    formation_t   const& get_formation()   const { return std::get<formation_t>(op); }
    formation_t        & get_formation()         { return std::get<formation_t>(op); }
    complexer_t   const& get_complexer()   const { return std::get<complexer_t>(op); }
    fill_t        const& get_fill()        const { return std::get<fill_t>(op);      }
    select_t      const& get_select()      const { return std::get<select_t>(op);    }
    einsummable_t const& get_einsummable() const {
      return std::get<einsummable_t>(op);
    }
    castable_t    const& get_castable() const {
      return get_einsummable().castable.value();
    }

  private:
    _op_t op;
    bool is_save_;
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

private:
  // autodiff stuff here

  set<int> compute_nodeset(
    vector<int> const& outs,
    vector<int> const& inns,
    bool include_inns_outs) const;

  struct backprop_tensor_t {
    backprop_tensor_t();
    backprop_tensor_t(int id);
    backprop_tensor_t(fill_t const& fill);

    static backprop_tensor_t ones(
      dtype_t const& dtype,
      vector<uint64_t> const& shape);
    static backprop_tensor_t zeros(
      dtype_t const& dtype,
      vector<uint64_t> const& shape);

    using op_t = std::variant<int, fill_t>;
    op_t op;

    int const& get_id() const;
    fill_t const& get_fill() const;
    scalar_t get_constant() const;

    bool is_constant() const;

    bool is_constant_of(scalar_t v) const;
    bool is_zeros() const;
    bool is_ones() const;

    dtype_t dtype(graph_t& self) const;
    vector<uint64_t> shape(graph_t& self) const;
  };

  struct backprop_state_t {
    // get the gradient of this id
    backprop_tensor_t operator[](int id);

    // Add the initial ones at out_id
    void start(int out_id);

    map<int, backprop_tensor_t> grads;
    graph_t& self;
    set<int> nodeset;

    struct out_edge_t {
      int out;
      int which_inn;
    };

    vector<out_edge_t> get_out_edges(int id) const;
  };

  // Compute the VJP term for this edge.
  //
  // Note: grad_id has the same out shape as at id,
  // and the return of this will have the same shape as the out
  // shape of the input at which_inn on id
  backprop_tensor_t
  build_grad_term(int id, int which_inn, backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_einsummable(
    einsummable_t const& e,
    vector<int> const& inn_ids,
    int which_inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  backprop_tensor_aggregate(
    backprop_tensor_t const& tensor,
    vector<int> const& es_inns,
    int out_rank);

  backprop_tensor_t
  build_grad_term_contraction(
    einsummable_t const& e,
    vector<int> const& inn_ids,
    int which_inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_ewu(
    einsummable_t const& e,
    int inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_select(
    select_t const& select,
    int which_inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_complexer(
    backprop_tensor_t grad_id);


  //int build_grad_term_ewb_arg(einsummable_t einsummable, int node_grad, int arg, int other, int which_inn);
  //int build_grad_term_ewb_lhs(einsummable_t einsummable, int node_grad, int arg, int other);
  //int build_grad_term_ewb_rhs(einsummable_t einsummable, int node_grad, int arg, int other);
  //int build_grad_term_reduction(einsummable_t einsummable, int node_grad, int node, int inn);

  // In case that one node has multiple edges
  // We need to sum all of its contributions to output function
  // Let's say that we have f(u,v) where u = u(x,y)  and v = v(x,y)
  // Then df/dx = (df/du)*(du/dx) + (df/dv)*(dv/dx)
  // And in the more complex case of f(u1, u2, ... , un) where
  // u1 = u1(x1, x2, ... , xm) , ... , un = un(x1, x2, ... , xm)
  // df/dxi = sum(df/duj * duj/dxi) where j = 1..n and i= 1..m
  backprop_tensor_t insert_adds(vector<backprop_tensor_t> const& items);
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

  // TODO: add insert_fill

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
    int inn);
  int insert_subset(
    partition_t partition,
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn);
  int insert_subset(
    vector<tuple<uint64_t, uint64_t>> hrect,
    int inn);

  vector<placement_t> get_placements() const;

  graph_t graph;
  map<int, placement_t> placements;
};

tuple<
  vector<tuple<int,int>>,
  graph_constructor_t>
create_remap_graph_constructor(
  remap_relations_t const& _remap);

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

    [[nodiscard]] tensor_t save() const;
    void save_inplace();

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

    tensor_t(
      full_shape_t const& shape,
      vector<int> const& modes,
      int id,
      graph_writer_t* self);

    bool _has_permutation() const;

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

  vector<tensor_t> backprop(tensor_t out, vector<tensor_t> params);

  // the core ops

  tensor_t input(
    vector<uint64_t> shape,
    dtype_t dtype = default_dtype());
  tensor_t input(
    full_shape_t shape,
    dtype_t dtype = default_dtype());
  tensor_t input(
    vector<vector<uint64_t>> const& shape,
    dtype_t dtype = default_dtype());

  tensor_t constant(
    scalar_t value,
    vector<uint64_t> shape,
    dtype_t dtype = default_dtype());
  tensor_t constant(
    scalar_t value,
    full_shape_t shape,
    dtype_t dtype = default_dtype());
  tensor_t constant(
    scalar_t value,
    vector<vector<uint64_t>> const& shape,
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

  // convert j->ij
  tensor_t broadcast(
    full_dim_t size,
    tensor_t const& inn);
  tensor_t broadcast(
    uint64_t sz,
    tensor_t const& inn);

private:
  graph_t graph;

  int _insert_elementwise(
    string str,
    scalarop_t op,
    int id);

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
