#include "antplacer.h"

void ant_graph_t::insert_edge(int inn, int out, uint64_t cost, int ident) {
  edge_t edge {
    .id = inn,
    .ident = ident
  };

  if(edge_costs.count(edge) > 0) {
    if(edge_costs[edge] != cost) {
      throw std::runtime_error("provided cost is incorrect");
    }
  } else {
    edge_costs.insert({edge, cost});
  }

  nodes[out].inns.insert(edge);
  nodes[inn].outs.insert(out);
}

int ant_graph_t::insert_node(uint64_t cost) {
  nodes.push_back(node_t {
    .cost = cost,
    .inns = {},
    .outs = {},
  });
  return nodes.size() - 1;
}

set<int> ant_graph_t::node_t::get_inputs() const {
  set<int> ret;
  for(auto const& [id,_]: inns) {
    ret.insert(id);
  }
  return ret;
}

void ant_graph_t::print_graphviz(std::ostream& out) const
{
  using std::endl;

  string tab = "  ";
  out << "digraph {" << endl;
  for(int id = 0; id != nodes.size(); ++id) {
    auto const& [cost, inns, _] = nodes[id];
    out << tab
      << "n" << id
      << " [label=\"C[" << cost << "]" << "\"";
    out << "]" << endl;

    for(edge_t const& edge: inns) {
      auto const& [inn, ident] = edge;
      uint64_t cost = edge_costs.at(edge);
      out << tab << "n" << inn << " -> " << "n" << id;
      out << " ";
      out << "[label=\"C[" << cost << "], I[" << ident << "]\"]";
      out << endl;
    }
  }
  out << "}" << endl;
}

vector<ant_graph_t::edge_t> ant_graph_t::outgoing_edges(int id) const {
  vector<edge_t> ret;
  auto const& node = nodes[id];
  for(auto const& out: node.outs) {
    auto const& out_node = nodes[out];
    for(auto const& edge: out_node.inns) {
      if(edge.id == id) {
        ret.push_back(edge);
      }
    }
  }
  return ret;
}

ant_state_t::ant_state_t(
  cluster_t const& c,
  ant_graph_t const& ag,
  equal_items_t<int> const& ecl)
  : cluster(cluster),
    ant_graph(ag),
    equal_compute_locations(ecl),
    apply_workers(c.devices.size()),
    move_workers(c.connections.size()),
    to_move_worker(cluster.to_connection),
    compute_locations(ag.nodes.size(), -1),
    compute_status(ag.nodes.size()),
    num_compute_remaining(ag.nodes.size()),
    time(0.0)
{
  // Set num inns remaining (compute status) and
  // add all inputs to pending location choices
  for(int id = 0; id != ant_graph.nodes.size(); ++id) {
    auto const& node = ant_graph.nodes[id];

    int num_inns = node.inns.size();
    compute_status[id] = num_inns;

    if(num_inns == 0) {
      pending_location_choices.push(id);
    }
  }
}

// What triggers what?
//   apply can start:
//     has location
//     all inputs have completed and moved to this location (if necc)
//   move can start:
//     has been computed
//     all outputs have been assigned a location
//   work can start:
//     corr worker is not busy
//     corr pending is not empty
// The invariant:
//   pending work should never not be in pending because a location
//   wasn't chosen
tuple<double, double, ant_state_t::completed_t>
ant_state_t::step(ant_interface_t const& interface)
{
  // Make sure allids in pending location choices have
  // a location
  while(pending_location_choices.size() != 0) {
    int id_ = pending_location_choices.front();
    pending_location_choices.pop();

    if(compute_locations[id_] >= 0) {
      continue;
    }

    vector<int> ids;
    if(equal_compute_locations.has(id_)) {
      set<int> const& eq_ids = equal_compute_locations.get_at(id_);
      ids = vector<int>(eq_ids.begin(), eq_ids.end());
      std::sort(ids.begin(), ids.end());
    } else {
      ids.push_back(id_);
    }

    int chosen_loc = interface.choose_location(id_);

    for(int const& id: ids) {
      if(compute_locations[id] != -1) {
        throw std::runtime_error(
          "this compute location has already been chosen somehow");
      }
      compute_locations[id] = chosen_loc;

      auto const& node = ant_graph.nodes[id];
      // all input nodes can be computed immediately and
      // won't get triggered anywhere else
      if(node.inns.size() == 0) {
        apply_workers[chosen_loc].add_to_pending(id);
      }
      // setting this location might trigger an inputs
      // move possibility
      for(int const& inn: node.get_inputs()) {
        if(can_be_moved(inn)) {
          add_move_to_pending(inn);
        }
      }
    }
  }

  // Make sure all the workers with something to do
  // are doing something
  for(int loc = 0; loc != apply_workers.size(); ++loc) {
    auto& apply_worker = apply_workers[loc];
    if(!apply_worker.is_in_progress()) {
      vector<int> const& pending = apply_worker.get_pending();

      int which = interface.choose_apply(loc, pending);

      int const& id = pending[which];
      uint64_t const& flops = ant_graph.nodes[id].cost;
      double work_time = cluster.compute(loc, flops);

      apply_worker.start_work(which, time, time + work_time);
    }
  }
  for(auto const& [key,idx]: to_move_worker) {
    auto const& [src,dst] = key;
    auto& move_worker = move_workers[idx];
    if(!move_worker.is_in_progress()) {
      vector<move_t> const& pending = move_worker.get_pending();

      int which = interface.choose_move(src, dst, pending);

      move_t const& edge = pending[which];
      uint64_t const& bytes = ant_graph.edge_costs.at(edge);
      double work_time = cluster.move(src, dst, bytes);

      move_worker.start_work(which, time, time + work_time);
    }
  }

  auto [apply_just_finish, which_finish] = get_next_finish();
  if(apply_just_finish) {
    auto& apply_worker = apply_workers[which_finish];
    auto [start,finish,id] = apply_worker.get_in_progress();

    time = finish;

    num_compute_remaining -= 1;

    compute_status[id] -= 1;
    if(compute_status[id] != -1) {
      throw std::runtime_error("compute status does not have -1 value");
    }

    auto const& node = ant_graph.nodes[id];

    int const& this_loc = compute_locations[id];

    // 1. if all the output locs have been assigned, this
    //    guy can start doing moves
    // 2. if any of the output locs haven't yet been assigned,
    //    they better get assigned quick
    // 3. if any output loc is also computed here, decrement
    //    the count accordingly
    bool will_move = true;
    for(int const& out_id: node.outs) {
      int const& out_loc = compute_locations[out_id];
      if(out_loc == -1) {
        will_move = false;
        pending_location_choices.push(out_id);
      } else if(out_loc == this_loc) {
        int cnt = 0;
        auto const& out_node = ant_graph.nodes[out_id];
        for(auto const& [id_,_]: out_node.inns) {
          if(id == id_) {
            cnt += 1;
          }
        }
        if(cnt == 0) {
          throw std::runtime_error("these edges aren't correct");
        }
        int& status = compute_status[out_id];
        status -= cnt;
        if(status == 0) {
          apply_worker.add_to_pending(out_id);
        }
      }
    }
    if(will_move) {
      add_move_to_pending(id);
    }

    return {start, finish, completed_t(this_loc, id)};
  } else {
    auto& move_worker = move_workers[which_finish];
    auto [start,finish,move] = move_worker.get_in_progress();

    time = finish;

    auto const& connection = cluster.connections[which_finish];
    auto const& src = connection.src;
    auto const& dst = connection.dst;

    auto const& node = ant_graph.nodes[move.id];
    for(auto const& out_id: node.outs) {
      int const& out_loc = compute_locations[out_id];
      if(out_loc == dst) {
        int& status = compute_status[out_id];
        status -= 1;
        if(status == 0) {
          apply_workers[dst].add_to_pending(out_id);
        }
      }
    }

    return {start, finish, completed_t(src, dst, move.id, move.ident)};
  }
}

bool ant_state_t::can_be_moved(int id) const {
  if(compute_status[id] != -1) {
    return false;
  }
  auto const& node = ant_graph.nodes[id];
  for(auto const& out_id: node.outs) {
    if(compute_locations[out_id] == -1) {
      return false;
    }
  }
  return true;
}

void ant_state_t::add_move_to_pending(int id){
  auto const& node = ant_graph.nodes[id];
  int const& src = compute_locations[id];

  // ident, dst
  set<tuple<int, int>> all_moves;

  for(auto const& out_id: node.outs) {
    int const& dst = compute_locations[out_id];
    if(src != dst) {
      auto const& out = ant_graph.nodes[out_id];
      for(auto const& [id_, ident]: out.inns) {
        if(id == id_) {
          if(all_moves.count({ident, dst}) == 0) {
            int which_worker = to_move_worker.at({src,dst});
            auto& move_worker = move_workers[which_worker];

            move_worker.add_to_pending(move_t{ id, ident });

            all_moves.insert({ident, dst});
          }
        }
      }
    }
  }
}

// is_apply,which
tuple<bool,int> ant_state_t::get_next_finish() const {
  vector<tuple<double, bool, int>> items;
  for(int i = 0; i != apply_workers.size(); ++i) {
    auto const& apply_worker = apply_workers[i];
    if(apply_worker.is_in_progress()) {
      items.emplace_back(
        std::get<1>(apply_worker.get_in_progress()),
        true,
        i);
    }
  }

  for(int i = 0; i != move_workers.size(); ++i) {
    auto const& move_worker = move_workers[i];
    if(move_worker.is_in_progress()) {
      items.emplace_back(
        std::get<1>(move_worker.get_in_progress()),
        false,
        i);
    }
  }

  auto const& [_, is_apply, which] = *std::min_element(items.begin(), items.end());
  return tuple<bool,int>{is_apply, which};
}

bool ant_state_t::all_done() const {
  return num_compute_remaining == 0;
}

bool operator==(ant_graph_t::edge_t const& lhs, ant_graph_t::edge_t const& rhs)
{
  return two_tuple_eq(lhs, rhs);
}
bool operator< (ant_graph_t::edge_t const& lhs, ant_graph_t::edge_t const& rhs)
{
  return two_tuple_lt(lhs, rhs);
}

// TODO: The taskgraph locally aggregates before moving. How will
//       that be accounted for here?
//       > what does the twolayer graph do? how does it handle aggregations?

// Here is the idea:
// 1 every node in the graph gets a usage partition which is the intersection
//   between the nodes partition and all of its uses
// 2 there is an edge between each inn block and out block with the ident of
//   the usage block
// Note: formation nodes of the same partition as the input
//       don't actually get formed in the ant graph
// Note: einsummables with one input and no aggregation that have the same
//       partition as the input are set as equal
tuple<
  map<int, tensor_t<int>>,
  equal_items_t<int>,
  ant_graph_t>
ant_graph_t::make(graph_t const& graph)
{
  equal_items_t<int> equal_items;
  map<tuple<int, int>, int> g_to_ag;
  ant_graph_t ant_graph;

  vector<partition_t> usage_partitions;
  // TODO: get all usage partitions

  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      auto const& partition = node.placement.partition;
      vector<int> shape = partition.block_shape();
      vector<int> index(shape.size(), 0);
      do {
        int aid  = ant_graph.insert_node(0);
        int gidx = idxs_to_index(shape, index);
        g_to_ag.insert({ {gid, gidx}, aid });
      } while(increment_idxs(shape, index));
    } else {
      if(node.op.is_formation()) {
        int const& inn_gid = node.inns[0];
        auto const& inn_node = graph.nodes[inn_gid];
        auto const& inn_part = inn_node.placement.partition;
        auto const& part = node.placement.partition;
        if(inn_part == part) {
          // if this partition is the same as the input partition
          // (which means there is no agg) then don't actually add
          // any ant_graph nodes since the placements should be the
          // same
          vector<int> shape = part.block_shape();
          vector<int> index(shape.size(), 0);
          do {
            int gidx = idxs_to_index(shape, index);
            int aid = g_to_ag.at({inn_gid, gidx});
            g_to_ag.insert({ {gid, gidx}, aid });
          } while(increment_idxs(shape, index));
        }
        continue;
      }
      if(node.op.is_einsummable() &&
         node.inns.size() == 1 &&
         !node.op.has_aggregation())
      {
        einsummable_t const& e = node.op.get_einsummable();

        int const& gid_inn = node.inns[0];
        auto const& node_inn = graph.nodes[gid_inn];

        auto const& part     = node.placement.partition;
        auto const& part_inn = node_inn.placement.partition;

        auto partdims_with_respect_to_inn =
          e.get_input_from_join(part.partdims, 0);

        // What about ij->i->i ?
        //   Here, the inn op is ij->i
        //   this op is           i->i
        // Here, the inn op has a rank 2 partition
        // and this op has a rank 1 partition, so
        // the corresponding partitions won't work.
        //
        // What about i->ij->ij ?
        //   This won't happen since i->ij isn't a valid
        //   einsummable op.
        //
        // But ij->ji->ij could work

        if(part_inn.partdims == partdims_with_respect_to_inn) {
          // In this case, reach through the permutation and make sure
          // the locations are equal

          auto inn_shape  = part_inn.block_shape();
          auto join_shape = part.block_shape();
          vector<int> join_index(join_shape.size(), 0);
          do {
            uint64_t cost = product(part.tensor_shape_at(join_index));

            // insert this ant graph node
            int aid = ant_graph.insert(cost);
            int gidx = idxs_to_index(join_shape, join_index);
            g_to_ag.insert({ {gid, gidx}, aid });

            // set this and the corresponding input location to be equal
            vector<int> inn_index = e.get_input_from_join(join_index, 0);
            int inn_idx = idxs_to_index(inn_shape, inn_index);
            int aid_inn = g_to_ag.at( { gid_inn, inn_idx } );
            // TODO: ^getting this should reach past missing formation nodes
            equal_items.insert(aid, aid_inn);

            // now add all the edges between aid_inn and aid.
            // Note: it may not be the case that the inn usage partition
            //       is the same as this node. Go through every block of
            //       the usage partition to create an edge from
            //       aid_inn to aid
            auto const& inn_usage_part = usage_partitions[gid_inn];
            vector<int> inn_usage_shape = inn_usage_part.block_shape();

            auto hrect = part.get_hrect(join_index);
            auto hrect_inn = e.get_input_from_join(hrect, 0);
            auto inn_usage_region = inn_usage_part.get_exact_region(hrect_inn);
            vector<int> inn_usage_idxs = vector_mapfst(inn_usage_region);
            do {
              int ident = idxs_to_index(inn_usage_shape, inn_usage_idxs);
              uint64_t cost = product(inn_usage_part.tensor_shape_at(inn_usage_idxs));
              ant_graph.insert_edge(aid_inn, aid, cost, ident);
            } while(increment_idxs_region(inn_usage_region, inn_usage_idxs);
          } while(increment_idxs(join_shape, join_index));

          continue;
        }
      }

      // This is a non-trivial formation or a non-identical einsummable


      auto const& join_part = node.placement.partition;
      vector<int> join_shape = part.block_shape();
      vector<int> join_idxs(join_shape.size(), 0);

      einsummable_t e;
      if(node.op.is_einsummable()) {
        node.op.get_einsummable();
      } else {
        vector<int> inns(join_shape.size());
        std::iota(inns.begin(), inns.end(), 0);

        // just create a dummy einsummable
        e = einsummable_t {
          .join_shape = join_part.total_shape(),
          .inns = { inns },
          .join = scalarop_t::make_identity(),
          .castable = optional<castable_t>()
        };
      }

      // For each out block:
      //   create aid w/ cost
      //   for each input:
      //     get the input usage partition
      //     get the input hrect across all aggregations of the input
      //     for each ???????????????????????????????????????
      // TODO: how did copy region work? see nasty bits of taskgraph or twolayergraph
      //
      do {
        uint64_t cost = product(join_part.tensor_shape_at(join_idxs));
        int aid = ant_graph.insert_edge(cost);

        auto join_hrect = join_part.get_hrect(join_idxs);
        for(int which_inn = 0; which_inn != e.inns.size(); ++which_inn) {
          vector<int> inn_out_index =
        }

      } while(increment_idxs(join_shape, join_idxs));
    }
  }

  return {g_to_ag, equal_items, ant_graph};
}
