#pragma once
#include "setup.h"

#include "graph.h"
#include "twolayergraph.h"
#include "taskgraph.h"

//struct costgraph_t {
//  static costgraph_t make(twolayergraph_t const& twolayer);
//
//  static costgraph_t make_from_taskgraph(taskgraph_t const& taskgraph);
//
//  // compute the cost of this graph over this cluster
//  double operator()(cluster_t const& cluster) const;
//
//  struct compute_t {
//    int loc;
//    uint64_t flops;
//    int util; // worker utilization
//  };
//
//  struct move_t {
//    int src;
//    int dst;
//    uint64_t bytes;
//  };
//
//  struct barrier_t {};
//
//  using op_t = std::variant<compute_t, move_t, barrier_t>;
//
//  struct node_t {
//    int id;
//    set<int> inns;
//    set<int> outs;
//    op_t op;
//
//    int worker_utilization() const {
//      if(std::holds_alternative<compute_t>(op)) {
//        return std::get<compute_t>(op).util;
//      } else if(std::holds_alternative<move_t>(op)) {
//        return 1;
//      } else {
//        throw std::runtime_error("invalid alternative in worker util");
//      }
//    }
//
//    bool is_barrier() const {
//      return std::holds_alternative<barrier_t>(op);
//    }
//    bool is_move() const {
//      return std::holds_alternative<move_t>(op);
//    }
//    bool is_compute() const {
//      return std::holds_alternative<compute_t>(op);
//    }
//
//    int const& move_src() const {
//      return std::get<move_t>(op).src;
//    }
//    int const& move_dst() const {
//      return std::get<move_t>(op).dst;
//    }
//    int const& compute_loc() const {
//      return std::get<compute_t>(op).loc;
//    }
//  };
//
//  vector<node_t> nodes;
//
//  int insert_compute(int loc, uint64_t flops, int util, vector<int> const& deps);
//  int insert_move(int src, int dst, uint64_t bytes, int id);
//  int insert_barrier(vector<int> const& deps);
//  int insert(op_t op, vector<int> const& deps);
//};
//
//std::ostream& operator<<(std::ostream& out, costgraph_t::node_t const& node);
//
//// TODO:
//// 1. may just want to cost taskgraph directly
//// 2. may want to incorporate touches into the cost
////
//// This object takes a graph with a fixed partition but
//// changing locations and costs it. It tracks the graph
//// object by reference.
////
//// Example usage:
////   graph_t graph = create graph with initial partition
////   coster_t coster(graph, cluster);
////   until stop:
////     modify graph's placements' locations
////     current_cost = coster()
//struct coster_t {
//  coster_t(graph_t const& g, cluster_t const& cluster)
//    : cluster(cluster), twolayer(twolayergraph_t::make(g))
//  {}
//
//  double operator()() const {
//    return this->operator()(cluster);
//  }
//
//  double operator()(cluster_t const& cluster) const {
//    auto costgraph = costgraph_t::make(twolayer);
//    return costgraph(cluster);
//  }
//
//private:
//  cluster_t const cluster;
//  twolayergraph_t const twolayer;
//};
