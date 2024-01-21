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
  dtype_t weight_dtype,
  update_type_t update_type)
  : server(server), updater(make_updater(weight_dtype, update_type))
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

  // save all the things on the out side
  for(auto const& [_, out_id]: updates) {
    graph.nodes[out_id].op.set_save(true);
  }

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

trainer_t::update_t trainer_t::make_updater(dtype_t d, update_type_t u)
{
  if(u == update_type_t::vanilla) {
    return update_t { .op = vanilla_update_t(d) };
  }
  if(u == update_type_t::adamw) {
    return update_t { .op = adamw_update_t(d) };
  }
  throw std::runtime_error("missing update_type case");
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
  auto vars = scalar_vars;
  updater.modify_vars(vars);
  server->execute(taskgraph, after_execution_map, vars);

  // remap: make sure that
  //        1) the updated weights are at the init weights
  //        2) constant tensors and inspect tensors are not deleted
  server->remap(out_remap_rels);
  server->remap_gids(out_remap_gids);
}

void trainer_t::update_t::init(trainer_t& self) {
  return std::visit([&](auto& u) {
    return u.init(self);
  }, op);
}

void trainer_t::update_t::modify_vars(map<string, scalar_t>& vars) {
  return std::visit([&](auto& u) {
    return u.modify_vars(vars);
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
  vector<tuple<int, int>> ret;
  ret.reserve(weight_ids.size());
  for(auto [weight, grad]: vector_zip(weight_ids, grad_ids)) {
    einsummable_t e = make_einsummable(dtype, graph.out_shape(weight));
    int updated_weight = graph.insert_einsummable(e, {weight, grad});
    ret.emplace_back(weight, updated_weight);
  }

  return ret;
}

void trainer_t::vanilla_update_t::modify_vars(
  map<string, scalar_t>& vars)
{
  // put a default learning rate if none is already in vars
  vars.insert({"learning_rate", scalar_t(dtype, "1e-3")});
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
  for(int gid: vector_concatenate(m_ids, v_ids)) {
    self.server->insert_constant(gid, self.inn_remap.at(gid), scalar_t::zero(dtype));
  }
}

void trainer_t::adamw_update_t::modify_vars(
  map<string, scalar_t>& vars)
{
  if(vars.count("beta1") == 0 ||
     vars.count("beta2") == 0 ||
     vars.count("eta") == 0)
  {
    throw std::runtime_error("missing vars provided");
  }

  iter++;

  scalar_t one = scalar_t::one(dtype);

  vars.insert_or_assign("beta1_complement", one - vars["beta1"]);
  vars.insert_or_assign("beta2_complement", one - vars["beta2"]);

  scalarop_t power = scalarop_t::make_power(iter, dtype);
  scalarop_t div = scalarop_t::make_div(dtype);

  vars.insert_or_assign("beta1tt", one / (one - power.eval(vars["beta1"])));
  vars.insert_or_assign("beta2tt", one / (one - power.eval(vars["beta2"])));

  vars.insert({"eps", scalar_t(dtype, "1e-8")});
}

vector<tuple<int, int>>
trainer_t::adamw_update_t::update_weights(
  graph_t& graph,
  vector<int> const& weight_ids,
  vector<int> const& grad_ids)
{
  int n = weight_ids.size();

  if(n == 0) {
    return {};
  }

  scalarop_t grad_update = scalarop_t::combine(
    scalarop_t::make_sub(dtype),
    {
      scalarop_t::make_identity(dtype),
      scalarop_t::make_scale("eta", dtype)
    });

  scalarop_t beta1_portion = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_scale("beta1", dtype),
      scalarop_t::make_scale("beta1_complement", dtype)
    });
  scalarop_t scale_square = scalarop_t::combine(
    scalarop_t::make_scale("beta2_complement"),
    { scalarop_t::make_square(dtype) });
  scalarop_t beta2_portion = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_scale("beta2", dtype),
      scale_square
    });
  scalarop_t sqrt_plus_eps = scalarop_t::combine(
    scalarop_t::make_add(dtype),
    {
      scalarop_t::make_sqrt(dtype),
      scalarop_t::make_variable("eps", dtype)
    });

  scalarop_t beta1t_scale = scalarop_t::make_scale("beta1tt", dtype);
  scalarop_t beta2t_scale = scalarop_t::make_scale("beta2tt", dtype);

  m_ids.reserve(n);
  v_ids.reserve(n);
  for(auto const& w_id: weight_ids) {
    auto shape = graph.out_shape(w_id);
    m_ids.push_back(graph.insert_input(shape, dtype));
    v_ids.push_back(graph.insert_input(shape, dtype));
  }

  vector<tuple<int, int>> ret;
  ret.reserve(n);
  for(int i = 0; i != n; ++i) {
    int const& w_id = weight_ids[i];
    int const& g_id = grad_ids[i];
    int const& m_id = m_ids[i];
    int const& v_id = v_ids[i];

    int m_new = insert_einsummable_ew(graph, beta1_portion, {m_id, g_id});
    int v_new = insert_einsummable_ew(graph, beta2_portion, {v_id, g_id});

    int mm = insert_einsummable_ew(graph, beta1t_scale, { m_new });
    int vv = insert_einsummable_ew(graph, beta2t_scale, { v_new });
    vv = insert_einsummable_ew(graph, sqrt_plus_eps, { vv });

    int xx = insert_einsummable_ew(graph, scalarop_t::make_div(dtype), {mm, vv});

    int w_new = insert_einsummable_ew(graph, grad_update, {w_id, xx});

    ret.emplace_back(m_id, m_new);
    ret.emplace_back(v_id, v_new);
    ret.emplace_back(w_id, w_new);
  }

  return ret;
}

int trainer_t::adamw_update_t::insert_einsummable_ew(
  graph_t& graph,
  scalarop_t join,
  vector<int> const& inns)
{
  vector<uint64_t> shape = graph.out_shape(inns[0]);
  int rank = shape.size();

  einsummable_t e(
    shape,
    vector<vector<int>>(inns.size(), vector_iota<int>(rank)),
    rank,
    join);

  return graph.insert_einsummable(e, inns);
}

