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

  int insert_squeezer(vector<uint64_t> const& new_shape, int inn);

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

  // Any combination of inserting and removing singleton dimensions
  // are allowed; otherwise this is a no-op
  struct squeezer_t {
    dtype_t dtype;
    vector<uint64_t> inn_shape;
    vector<uint64_t> out_shape;
  };

  // This has to be an op because a (5,3) partition of reals
  // can't be converted into complexes without a change of partition.
  // That is, (4,4) real parts can be viewed as (2,2) complex parts
  // for free.

  struct op_t {
  private:
    using _op_t = std::variant<
      input_t, formation_t, complexer_t, squeezer_t,
      fill_t, select_t, einsummable_t>;

  public:
    op_t(_op_t op, bool is_save);

    op_t(input_t       x, bool s = false): op_t(_op_t(x), s) {}
    op_t(formation_t   x, bool s = false): op_t(_op_t(x), s) {}
    op_t(complexer_t   x, bool s = false): op_t(_op_t(x), s) {}
    op_t(squeezer_t    x, bool s = false): op_t(_op_t(x), s) {}
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
    bool is_squeezer()    const { return std::holds_alternative<squeezer_t>(op);  }
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
    squeezer_t    const& get_squeezer()    const { return std::get<squeezer_t>(op);  }
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

  vector<int> reverse_order_nodeset(set<int> const& ids) const;

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
    void insert(int id);

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
    int out_id,
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
  build_grad_term_ew(
    einsummable_t const& e,
    vector<int> inn_ids,
    int which_inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_select(
    select_t const& select,
    int which_inn,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_complexer(
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_squeezer(
    vector<uint64_t> const& inn_shape,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_reduction_add(
    vector<uint64_t> const& join_shape,
    vector<int> const& inn,
    int out_rank,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_reduction_mul(
    vector<uint64_t> const& join_shape,
    vector<int> const& inn,
    int out_rank,
    int out_id,
    int inn_id,
    backprop_tensor_t grad_id);

  backprop_tensor_t
  build_grad_term_reduction_maxmin(
    vector<uint64_t> const& join_shape,
    vector<int> const& inn,
    int out_rank,
    int out_id,
    int inn_id,
    backprop_tensor_t grad_id);

  // In case that one node has multiple edges
  // We need to sum all of its contributions to output function
  // Let's say that we have f(u,v) where u = u(x,y)  and v = v(x,y)
  // Then df/dx = (df/du)*(du/dx) + (df/dv)*(dv/dx)
  // And in the more complex case of f(u1, u2, ... , un) where
  // u1 = u1(x1, x2, ... , xm) , ... , un = un(x1, x2, ... , xm)
  // df/dxi = sum(df/duj * duj/dxi) where j = 1..n and i= 1..m
  backprop_tensor_t insert_adds(vector<backprop_tensor_t> const& items);

  // Insert an einsummable and then if there is a reduction, a formation node too.
  //
  // This is used because everywhere a backprop term is added, we want the aggs
  // to be formed.
  int insert_einsummable_for_backprop(
    einsummable_t e,
    vector<int> inns);
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
