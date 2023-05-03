#pragma once
#include "setup.h"

#include "placement.h"
#include "einsummable.h"

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

  vector<uint64_t> out_shape(int id) const;

  vector<int> get_order() const;

  // Assumption: if location i is used in a placement,
  // then all locations 0, ..., i-1 are also used.
  // Put another way, location i is the ith plus 1 processor.
  // As such, return the maximum location value used plus 1.
  int num_locs() const;

  void print() const;

public:

  struct input_t {
    vector<uint64_t> shape;

    vector<uint64_t> out_shape() const { return shape; }
  };

  struct formation_t {
    vector<uint64_t> shape;
    bool is_save; // if this is false, it is a temporary

    vector<uint64_t> out_shape() const { return shape; }
  };


  struct op_t {
  private:
    using _op_t = std::variant<input_t, formation_t, einsummable_t>;

  public:
    op_t(_op_t op): op(op) {}

    op_t(input_t       x): op_t(_op_t(x)) {}
    op_t(formation_t   x): op_t(_op_t(x)) {}
    op_t(einsummable_t x): op_t(_op_t(x)) {}

    vector<uint64_t> out_shape() const {
      return std::visit([](auto x){ return x.out_shape(); }, op);
    }
    int out_rank() const {
      return this->out_shape().size();
    }

    vector<uint64_t> shape() const {
      if(std::holds_alternative<input_t>(op)) {
        return std::get<input_t>(op).shape;
      }
      if(std::holds_alternative<formation_t>(op)) {
        return std::get<formation_t>(op).shape;
      }
      if(std::holds_alternative<einsummable_t>(op)) {
        return std::get<einsummable_t>(op).join_shape;
      }
      throw std::runtime_error("graph::op_t should not reach");
      return {};
    }

    int rank() const {
      return this->shape().size();
    }

    bool is_save() const {
      return is_formation() && get_formation().is_save;
    }
    bool is_formation() const {
      return std::holds_alternative<formation_t>(op);
    }
    bool is_input() const {
      return std::holds_alternative<input_t>(op);
    }
    bool is_einsummable() const {
      return std::holds_alternative<einsummable_t>(op);
    }

    einsummable_t const& get_einsummable() const {
      return std::get<einsummable_t>(op);
    }
    einsummable_t& get_einsummable() {
      return std::get<einsummable_t>(op);
    }

    formation_t const& get_formation() const {
      return std::get<formation_t>(op);
    }
    formation_t& get_formation() {
      return std::get<formation_t>(op);
    }

    _op_t op;
  };

  struct node_t {
    op_t op;
    vector<int> inns;
    set<int> outs;
    placement_t placement;

    set<int> get_inns_set() const {
      return set<int>(inns.begin(), inns.end());
    }
    int num_distinct_inputs() const {
      return get_inns_set().size();
    }
    int num_locs() const {
      auto const& locs = placement.locations.get();
      return 1 + *std::max_element(locs.begin(), locs.end());
    }
  };

  vector<node_t> nodes;

private:
  int insert(op_t const& op, vector<int> inns, placement_t placement);
};

// Construct a 3D matmul graph, (ij,jk->ik)
//   shape lhs: di*pi x dj*pj
//   shape rhs: dj*pj x dk*pk
//   shape out: di*pi x dk*pk
graph_t three_dimensional_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk,
  int num_processors);

// Same tensor dimensions as 3d matmul
// Does not explicitly set locations
graph_t straight_matrix_multiplication(
  int pi, int pj, int pk,
  uint64_t di, uint64_t dj, uint64_t dk);

// Given a compute graph, create a "good" partition for that
// graph.
//
// Break up einsummable nodes into mmlike nodes and not.
// Mmlike (matmul-like) nodes are einsummable with multiple
// inputs and have an aggregation.
//
// All mmlike nodes get partitioned so that they have a
// sizing of mmlike_sizing, done is such a way that each dimension
// is roughly the same size.
// For example: if mmlike_sizing is 1k^3, then a big matmul
// will be decomposed into 1k by 1k left and right input blocks.
// Note: this behaviour might not be desired, so the caller can
// specify fixed partitionings.
//
// Once all mmlike nodes have been specified, all other nodes
// get a partition whenever all of it's input nodes are partitioned.
// For formation nodes, if the corresponding input has an aggregation,
// then the partition is "3d-matmul-formation-partition"--which is
// to have the same partition as the input node and then cut again
// to have a depth the size as the number of aggregates
// Example: For matmul in 2 by 2 by 2 blocks, there are 2
// aggregates so the formation node would preferably be 4 by 2
// else possibly 2 by 4.
// All non-mmlike einsummable nodes get partitioned like
// the intersection of their inputs. For binary ew, the is the
// intersection'd size of both inputs. For matrix addition,
// if lhs is in row strips and rhs is in col strips, the ew
// op will be partitioned along both the row and cols.
// If the resulting partition falls under the min_sizing,
// then a coarser partition is chosen or an input partition,
// if applicable.
//
// Once no more input nodes are partitioned, it may be the case
// that some nodes still aren't partitioned--i.e. all
// input nodes not explictly partitioned.
// Here, all remaining nodes are partitioned based on the
// intersection of their output usages. If the node has no outputs,
// then as a singleton partition. No aggregation dimensions get
// partitioned in this setting.
// If any resulting partition falls under min_sizing,
// a partitioned deduced from an output is chosen.
//
// The user may also set some nodes to have an explicit partition
// or to be equal to each other. For nodes that are equal to each other,
// whenever one node gets set, the other gets that partition.
vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t mmlike_sizing,
  uint64_t min_sizing,
  // make sure each of these pair have the same partition
  set<tuple<int, int>> const& equal_constraints,
  map<int, partition_t> const& fixed_constraints);

// The same thing with no constriants
vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t mmlike_sizing,
  uint64_t min_sizing);

