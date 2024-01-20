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
  f_autoplace_t autoplace,
  update_type_t update_type)
  : server(server), updater(update_type)
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

  vector<tuple<int, int>> updates = updater.update_weights(
    graph,
    weight_ids,
    grad_ids);

  {
    std::ofstream out("g.gv");
    graph.print_graphviz(out, colors);
    DOUT("printed g.gv");
  }

  map<int, placement_t> fixed_pls;
  for(int const& id: inspect_ids) {
    vector<uint64_t> shape = graph.nodes[id].op.shape();
    fixed_pls.insert({id, placement_t(partition_t::singleton(shape))});
  }

  vector<placement_t> placements = autoplace(graph, fixed_pls, updates);

  auto [inn_tids, out_tids, taskgraph_] = taskgraph_t::make(graph, placements);
  taskgraph = std::move(taskgraph_);

  {
    std::ofstream out("tg.gv");
    taskgraph.print_graphviz(out);
    DOUT("printed tg.gv");
  }

  // Make sure that inn_ids == updates ++ data_ids ++ constant_ids
  {
    set<int> inn_gids;
    for(auto const& [inn_gid, _]: inn_tids) {
      inn_gids.insert(inn_gid);
    }
    vector<int> update_inns = vector_mapfst(updates);
    set<int> inn_gids_(update_inns.begin(), update_inns.end());
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

  // Most likely, update_inn = a weight id and
  //              update_out = the weight after being updated
  // Here, inn, out mean that update_inn is an input to graph_t
  // and update_out is an output of graph_t.
  for(auto const& [update_inn, update_out]: updates) {
    after_execution_map.insert({
      update_out,
      relation_t {
        .dtype     = graph.out_dtype(update_out),
        .placement = placements.at(update_out),
        .tids      = out_tids.at(update_out)
      }
    });

    out_remap_rels.insert({
      update_out,
      relation_t {
        .dtype     = graph.out_dtype(update_inn),
        .placement = placements.at(update_inn),
        .tids      = inn_tids.at(update_inn)
      }
    });
    out_remap_gids.emplace_back(update_out, update_inn);
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

void trainer_t::init() {
  updater.init(*this);
}

void trainer_t::operator()(map<string, scalar_t> const& scalar_vars)
{
  // prepare: Verify that data ids, constant ids and weight ids are in the server.
  //          and do a remap to give them the correct placement.
  server->remap(inn_remap);
  // Note: this will delete any inspect tensors from the previous iteration

  // execute: Run the graph
  server->execute(taskgraph, after_execution_map, scalar_vars);

  // remap: make sure that
  //        1) the updated weights are at the init weights
  //        2) constant tensors and inspect tensors are not deleted
  server->remap(out_remap_rels);
  server->remap_gids(out_remap_gids);
}

trainer_t::update_t::update_t(update_type_t u) {
  if(u == update_type_t::vanilla) {
    op = vanilla_update_t();
  } else if(u == update_type_t::adamw) {
    op = adamw_update_t();
  } else {
    throw std::runtime_error("missing update_type impl");
  }
}

void trainer_t::update_t::init(trainer_t& self) {
  return std::visit([&](auto& u) {
    return u.init(self);
  }, op);
}

vector<tuple<int, int>>
trainer_t::update_t::update_weights(
  graph_t& graph,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  return std::visit([&](auto& u) {
    return u.update_weights(graph, weight_ids, grad_ids);
  }, op);
}

vector<tuple<int, int>>
trainer_t::vanilla_update_t::update_weights(
  graph_t& graph,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  optional<dtype_t> maybe_dtype;

  vector<tuple<int, int>> ret;
  ret.reserve(weight_ids.size());
  for(auto [weight, grad]: vector_zip(weight_ids, grad_ids)) {
    if(maybe_dtype) {
      if(graph.out_dtype(weight) != maybe_dtype.value()) {
        throw std::runtime_error("all weights must have the same dtype");
      }
    } else {
      maybe_dtype = graph.out_dtype(weight);
    }
    dtype_t const& dtype = maybe_dtype.value();

    einsummable_t e = make_einsummable(dtype, graph.out_shape(weight));
    int updated_weight = graph.insert_einsummable(e, {weight, grad});
    graph.nodes[updated_weight].op.set_save(true);
    ret.emplace_back(weight, updated_weight);
  }

  return ret;
}

einsummable_t
trainer_t::vanilla_update_t::make_einsummable(
  dtype_t dtype,
  vector<uint64_t> const& shape) const
{
  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(),
    {
      scalarop_t::make_identity(dtype),
      scalarop_t::make_scale("learning_rate", dtype)
    }
  );

  int rank = shape.size();
  return einsummable_t(
    shape,
    { vector_iota<int>(rank), vector_iota<int>(rank) },
    rank,
    grad_update);
}

void trainer_t::adamw_update_t::init(trainer_t& self) {
  // TODO: for each id in (m_ids, v_ids),
  //          set the trainer to zero at these locations
  throw std::runtime_error("adamw not implemented");
}

vector<tuple<int, int>>
trainer_t::adamw_update_t::update_weights(
  graph_t& graph,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  throw std::runtime_error("adamw not implemented");
//  int n = weight_ids.size();
//
//  m_ids.reserve(n);
//  v_ids.reserve(n);
//  for(auto const& w_id: weight_ids) {
//    auto shape = graph.out_shape(w_id);
//    dtype_t dtype = graph.out_dtype(w_id);
//    m_ids.push_back(graph.insert_input(shape, dtype));
//    v_ids.push_back(graph.insert_input(shape, dtype));
//  }
//
//  vector<tuple<int, int>> ret;
//  ret.reserve(n);
//  for(int i = 0; i != n; ++i) {
//    int const& w_id = weight_ids[i];
//    int const& g_id = grad_ids[i];
//    int const& m_id = m_ids[i];
//    int const& v_id = v_ids[i];
//
//    int m_new = update_portion(graph, params.beta1, m_id, g_id);
//    int v_new = update_portion(graph, params.beta2, v_id, g_id);
//  }
}

