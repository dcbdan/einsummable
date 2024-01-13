#pragma once
#include "../base/setup.h"

#include "base.h"

// loss_id:      What to take the backprop against. Need not be saved
// inspect_ids:  tensors that get computed and should be saved so the user
//               can inspect them
// data_ids:     tensors that get inserted by the user before each iteration
//               (e.g. data matrix x and correct results y)
// constant_ids: input tensors that never get changed and must be insert by
//               the user before the first iteration
// weight_ids:   the tensors that get updated via update(weight, grad)
struct trainer_t {
  using f_autoplace_t =
    std::function<vector<placement_t>(
      graph_t const&,
      map<int, placement_t> const&,
      vector<tuple<int,int>> const&)>;

  trainer_t(
    server_base_t* server,
    graph_t const& init_graph,
    int loss_id,
    vector<int> const& inspect_ids,
    vector<int> const& data_ids,
    vector<int> const& constant_ids,
    vector<int> const& weight_ids,
    scalarop_t update,  // weight elem , grad elem -> new eight elem
    f_autoplace_t autoplace);

  void operator()();

  relation_t const& get_input_relation(int id) {
    return inn_remap.at(id);
  }

private:
  static einsummable_t make_einsummable_update(
    scalarop_t update,
    vector<uint64_t> const& shape);

private:
  server_base_t* server;

  taskgraph_t taskgraph;

  map<int, relation_t> inn_remap;

  map<int, relation_t> after_execution_map;

  map<int, relation_t> out_remap_rels;
  vector<tuple<int, int>> out_remap_gids;
};

