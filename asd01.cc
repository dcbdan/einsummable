#include <iostream>
#include <vector>
#include <unordered_map>
#include <variant>
#include <tuple>

#include "tensor.h"

using std::tuple;
using std::vector;
using map = std::unordered_map

struct input_t {
  vector<uint64_t> shape;
};

struct output_t {
  vector<uint64_t> shape;
};

enum class castable_t { add, mul, min, max };

enum class scalar_join_t { mul };

struct einsummable_t {
  // ij,jk->ik
  static from_matmul(uint64_t di, uint64_t dj, uint64_t dk) {
    // ij,jk->ik
    // 02 21  01
    return einsummable_t {
      .join_shape = {di, dk, dj},
      .inns = { {0, 2}, {2, 1} },
      .out_rank = 2,
      .join = scalar_join_t::mul,
      .castable = castable_t::add
    };
  }

  vector<uint64_t> join_shape;

  vector<vector<int>> inns;
  int out_rank;

  scalar_join_t join;
  castable_t castable;
};

using node_t = std::variant<input_t, output_t, einsummable_t>;

struct partdim_t {
  static partdim_t from_sizes(vector<uint64_t> sizes) {
    vector<uint64_t> spans = sizes;
    uint64_t total = spans[0];
    for(int i = 1; i < ret.size(); ++i) {
      spans[i] += total;
      total = spans[i];
    }
    return partdim_t { .spans = spans };
  }
  static partdim_t repeat(int n_repeat, uint64_t sz) {
    return from_sizes(vector<uint64_t>(n_repeat, sz));
  }
  static partdim_t singleton(uint64_t shape) {
    return partdim_t { .spans = {shape} };
  }

  uint64_t total() const { return spans.back(); }

  vector<uint64_t> sizes() const {
    vector<uint64_t> ret = spans;
    for(int i = ret.size(); i > 0; --i) {
      ret[i] -= ret[i-1];
    }
    // spans = [10,20,30,35]
    // ret   = [10,10,10, 5]
    return ret;
  }

  int num_parts() const {
    return spans.size();
  }

  vector<uint64_t> spans;
};

struct partition_t {
  partition_t(partition_t const& p):
    partdims(p)
  {}

  static partition_t singleton(vector<uint64_t> shape) {
    vector<partdim_t> partdims(shape.size());
    for(auto const& sz: shape) {
      partdims.push_back(partdim_t.singleton(sz));
    }
    return partition_t { .partdims = partdims };
  };

  vector<uint64_t> total_shape() const {
    vector<uint64_t> ret(partdims.size());
    for(auto const& pdim: partdims) {
      ret.push_back(pdim.total());
    }
    return ret;
  }

  int num_parts() const {
    return product(this->block_shape());
  }

  vector<int> block_shape() const {
    vector<int> ret;
    ret.reserve(partdims.size());
    for(auto const& p: partdims) {
      ret.push_back(p.num_parts());
    }
    return ret;
  }

  vector<partdim_t> partdims;
};

struct placement_t {
  placement_t(partition_t const& p):
    placement_t(p, tensor_t<int>(p.block_shape()))
  {}

  placement_t(partition_t const& p, tensor_t<int> const& locs):
    partition(p), locations(locs)
  {}

  partition_t const partition;
  tensor_t<int> locations;
};

struct graph_t {
  // Methods to construct a graph object
  // {{{
  int insert_input(
    placement_t placement);
  int insert_input(
    partition_t partition);
  int insert_input(
    vector<uint64_t> shape);

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

  int insert_output(
    placement_t placement,
    int inn);
  int insert_output(
    partition_t partition,
    int inn);
  int insert_output(
    vector<uint64_t> shape,
    int inn);

  // for each input and einsummable ops O, make sure O
  // is used by another op. If not, add an output op that
  // uses O.
  void set_outputs();
  // }}}

private:
  struct info_t {
    node_t node;
    vector<int> inns;
    vector<int> outs;
    partition_t partition;
  };

  vector<info_t> infos;
};

// Construct a 3D mamtul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_t three_dimensional_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk,
  int num_processors)
{
  // The mapping from "procesor" grid to actual processor;
  // this is necessary for when pi*pj*pk > num_processors
  auto to_processor = [&](int i, int j, int k) {
    int index = idxs_to_index({pi,pj,pk}, {i,j,k});
    return index % num_processors;
  };

  // rcp = row, column, row_part
  enum class rcp_t { ijk, jki, ikj };

  // All matrices are partitioned along the rows and then the
  // columns, but then each block is further partitioned.
  //
  // So if A is partitioned rcp_t::ijk, then there are pi rows,
  // pj columns to form Aij. But each Aij is partitioned further
  // along the rows, into pk parts.
  // That means the partition is really (pi*pk, pj).
  auto make_matrix_partition = [&](rcp_t which) {
    int nr;
    int nc;
    int np;
    uint64_t dr;
    uint64_t dc;
    if(which == rcp_t::ijk) {
      nr = pi;
      nc = pj;
      np = pk;
      dr = di;
      dc = dj;
    } else if(which == rcp_t::jki) {
      nr = pj;
      nc = pk;
      np = pi;
      dr = dj;
      dc = dk;
    } else if(which == rcp_t::ikj) {
      nr = pi;
      nc = pk;
      np = pj;
      dr = di;
      dc = dk;
    } else {
      throw std::runtime_error("should not reach");
    }
    vector<uint64_t> part_sizes = divide_evenly(np, dr);
    vector<uint64_t> sizes_row;
    sizes_row.reserve(nr*np);
    for(int i = 0; i != nr; ++i) {
      vector_concatenate(sizes_row, part_sizes);
    }
    partdim_t part_row = partdim_t.from_sizes(sizes_row);

    partdim_t part_col = partdim_t.repeat(nc, dc);

    return partition_t({part_row, part_col});
  };

  // For which == rcp_t::ijk,
  // Aij(k) lives at processor (i,j,k).
  // That is, A is partitioned into pi rows, pj columns.
  // Each Aij is distributed across (i,j,*) and
  // Aij is chopped along it's rows forming Aij(k) for
  // some k in 0,...,pk-1.
  auto make_matrix_locs = [&](rcp_t which) {
    vector<int> shape;
    if(which == rcp_t::ijk) {
      shape = {pi*pk, pj};
    } else if(which == rcp_t::jki) {
      shape = {pj*pi, pk};
    } else if(which == rcp_t::ikj) {
      shape = {pi*pj, pk};
    } else {
      throw std::runtime_error("should not reach");
    }
    tensor_t<int> locs(shape);

    int i;
    int j;
    int k;
    for(int r = 0; i != n_row;      ++r) {
    for(int c = 0; c != n_col;      ++c) {
    for(int p = 0; p != n_row_part; ++p) {
      if(which == rcp_t::ijk) {
        i = r;
        j = c;
        k = p;
      } else if(which == rcp_t::kji) {
        i = p;
        j = r;
        k = c
      } else if(which == rcp_t::ikj) {
        i = r;
        j = p;
        k = c;
      } else {
        throw std::runtime_error("should not reach");
      }

      locs[r*n_row_part + p, c] = to_processor(i,j,k);
    }}}

    return locs;
  };

  auto make_matrix_placement(rcp_t rcp) {
    return placement_t(
      make_matrix_partition(rcp),
      make_matrix_locs(rcp)
    );
  };

  graph_t ret;

  int id_lhs = ret.insert_input(make_matrix_placement(rcp_t::ijk));
  int id_rhs = ret.insert_input(make_matrix_placement(rcp_t::jki));

  int id_op;
  {
    einsummable_t matmul = einsummable.from_matmul(di*pi, dj*pj, dk*pk);
    // Be careful: matmul (ij,jk->ik) has indices {0: i, 1: k, 2: j}

    partition_t part({
      partdim_t.repeat(pi, di),
      partdim_t.repeat(pk, dk),
      partdim_t.repeat(pj, dj)
    });

    tensor_t<int> locs({pi,pk,pj});

    for(int i = 0; i != pi; ++i) {
    for(int j = 0; j != pj; ++j) {
    for(int k = 0; k != pk; ++k) {
      locs[i,k,j] = to_processor(i,j,k);
    }}}

    placement_t placement(part, locs);

    id_op = ret.insert_einsummable(placement, matmul, {id_lhs, id_rhs});
  }

  // the output node
  ret.insert_output(make_matrix_placement(rcp_t::ikj), id_op);
}


