#pragma once
#include "../base/setup.h"

#include "graph.h"

struct graph_writer_t {
  struct idx_t {
    struct rng {
      int64_t beg; // TODO: do the negative values ever get used?
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
    tensor_t squeeze(int which_dim) const;

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

  // TODO: The problem with this is that the shapes
  //       cannot be grouped
  tensor_t fill(fill_t const& fill);

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

  tensor_t exp(
    tensor_t const& inn);

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
