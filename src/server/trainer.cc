#include "trainer.h"

#include <fstream> // TODO: remove

trainer_t::trainer_t(
  server_base_t* server,
  graph_t const& init_graph,
  int loss_id,
  vector<int> const& inspect_ids,
  vector<int> const& data_ids,
  vector<int> const& constant_ids,
  vector<int> const& weight_ids,
  scalarop_t update,  // weight elem , grad elem -> new eight elem
  f_autoplace_t autoplace)
  : server(server)
{
  graph_t graph = init_graph;
  map<int, string> colors;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    graph.nodes[gid].op.set_save(false);
    colors.insert({gid, "yellow"});
  }

  for(int const& inspect_id: inspect_ids) {
    graph.nodes[inspect_id].op.set_save(true);
  }
  for(int const& constant_id: constant_ids) {
    graph.nodes[constant_id].op.set_save(true);
  }

  vector<int> grad_ids = graph.backprop(loss_id, weight_ids);

  vector<int> updated_weights;
  updated_weights.reserve(grad_ids.size());
  for(auto [weight, grad]: vector_zip(weight_ids, grad_ids)) {
    int updated_weight = graph.insert_einsummable(
      make_einsummable_update(update, graph.out_shape(weight)),
      {weight, grad});
    graph.nodes[updated_weight].op.set_save(true);
    updated_weights.push_back(updated_weight);
  }

  {
    std::ofstream out("g.gv");
    graph.print_graphviz(out, colors);
    DOUT("printed g.gv");
  }

  vector<placement_t> placements;
  {
    map<int, placement_t> fixed_pls;
    for(int const& id: inspect_ids) {
      vector<uint64_t> shape = graph.nodes[id].op.shape();
      fixed_pls.insert({id, placement_t(partition_t::singleton(shape))});
    }

    vector<tuple<int,int>> equal_pls = vector_zip(weight_ids, updated_weights);

    placements = autoplace(graph, fixed_pls, equal_pls);
  }

  auto [inn_tids, out_tids, taskgraph_] = taskgraph_t::make(graph, placements);
  taskgraph = std::move(taskgraph_);

  {
    std::ofstream out("tg.gv");
    taskgraph.print_graphviz(out);
    DOUT("printed tg.gv");
  }

  // Make sure that inn_ids == weights ++ data_ids ++ constant_ids
  {
    set<int> inn_gids;
    for(auto const& [inn_gid, _]: inn_tids) {
      inn_gids.insert(inn_gid);
    }
    set<int> inn_gids_(weight_ids.begin(), weight_ids.end());
    set_union_inplace(inn_gids_, set<int>(data_ids.begin(), data_ids.end()));
    set_union_inplace(inn_gids_, set<int>(constant_ids.begin(), constant_ids.end()));
    if(inn_gids != inn_gids_) {
      throw std::runtime_error("invalid input gid set in trainer initialization");
    }
  }

  for(auto const& [inn_gid, tids]: inn_tids) {
    inn_remap.insert({
      inn_gid,
      relation_t {
        .dtype     = graph.out_dtype(inn_gid),
        .placement = placements.at(inn_gid),
        .tids      = tids
      }
    });
  }

  for(auto const& [weight, updated_weight]: vector_zip(weight_ids, updated_weights)) {
    after_execution_map.insert({
      updated_weight,
      relation_t {
        .dtype     = graph.out_dtype(updated_weight),
        .placement = placements.at(weight),
        .tids      = out_tids.at(updated_weight)
      }
    });

    out_remap_rels.insert({
      updated_weight,
      relation_t {
        .dtype     = graph.out_dtype(weight),
        .placement = placements.at(weight),
        .tids      = inn_tids.at(weight)
      }
    });
    out_remap_gids.emplace_back(updated_weight, weight);
  }

  // remap info for constant ids and inspect ids are just to make sure things
  // are not deleted
  for(int const& id: vector_concatenate(constant_ids, inspect_ids)) {
    relation_t rel {
      .dtype     = graph.out_dtype(id),
      .placement = placements.at(id),
      .tids      = out_tids.at(id)
    };

    after_execution_map.insert({id, rel});

    out_remap_rels.insert({id, rel});

    out_remap_gids.emplace_back(id, id);
  }
}

void trainer_t::operator()() {
  // prepare: Verify that data ids, constant ids and weight ids are in the server.
  //          and do a remap to give them the correct placement.
  server->remap(inn_remap);
  // Note: this will delete any inspect tensors from the previous iteration

  // execute: Run the graph
  server->execute(taskgraph, after_execution_map);

  // remap: make sure that
  //        1) the updated weights are at the init weights
  //        2) constant tensors and inspect tensors are not deleted
  server->remap(out_remap_rels);
  server->remap_gids(out_remap_gids);
}

einsummable_t trainer_t::make_einsummable_update(
  scalarop_t update,
  vector<uint64_t> const& shape)
{
  int rank = shape.size();
  return einsummable_t(
    shape,
    { vector_iota<int>(rank), vector_iota<int>(rank) },
    rank,
    update);
}

