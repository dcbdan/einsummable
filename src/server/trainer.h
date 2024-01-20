#pragma once
#include "../base/setup.h"

#include "base.h"

enum class update_type_t {
  vanilla,
  adamw
};

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
    f_autoplace_t autoplace,
    update_type_t);

  void init();

  void operator()(map<string, scalar_t> const& vals);

  relation_t const& get_input_relation(int id) {
    return inn_remap.at(id);
  }

private:
  // Idea: Find the unifying structure of tensors
  //
  // The types of tensors:
  //                     has_input has_same_output has_updated_output at_pl0
  //  weights            T         F               T                  NA
  //  updates            T         F               T                  NA
  //  constants          T         T               F                  NA
  //  inspect            F         NA              NA                 T
  //  data               T         F               F                  NA
  //
  // Does the tensor have an input?
  //   Yes: Does the tensor have an output?
  //     Yes: Is the input and the output the same?
  //       Yes: Constants
  //       No:  Updates
  //     No: Data
  //   No: Does it need to end up at location zero in singleton?
  //     Yes: inspect

  // Note: updaters may add

  struct vanilla_update_t {
    vanilla_update_t() {}

    void init(trainer_t& self) {}

    vector<tuple<int, int>> update_weights(
      graph_t& graph,
      vector<int> const& weight_ids,
      vector<int> const& grad_ids);

  private:
    einsummable_t make_einsummable(
      dtype_t dtype, vector<uint64_t> const& shape) const;
  };

  // https://arxiv.org/pdf/1711.05101.pdf
  struct adamw_update_t {
    adamw_update_t() {}

    void init(trainer_t& self);

    vector<tuple<int, int>> update_weights(
      graph_t& graph,
      vector<int> const& weight_ids,
      vector<int> const& grad_ids);

  private:
    vector<int> m_ids;
    vector<int> v_ids;
  };

  struct update_t {
    update_t(update_type_t u);

    void init(trainer_t& self);

    vector<tuple<int, int>> update_weights(
      graph_t& graph,
      vector<int> const& weight_ids,
      vector<int> const& grad_ids);

    using op_t = std::variant<vanilla_update_t, adamw_update_t>;
    op_t op;
  };

  server_base_t* server;

  update_t updater;

  taskgraph_t taskgraph;

  map<int, relation_t> inn_remap;

  map<int, relation_t> after_execution_map;

  map<int, relation_t> out_remap_rels;
  vector<tuple<int, int>> out_remap_gids;
};

