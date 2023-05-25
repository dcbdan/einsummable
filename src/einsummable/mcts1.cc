#include "mcts1.h"

namespace mcts1_ns {

vector<locset_t> make_locsets(int n) {
  if(n == 1) {
    return { {0,1} };
  }
  if(n == 2) {
    return { {0,2}, {0,1}, {1,2} };
  }
  if(n == 3) {
    return { {0,3}, {0,1}, {2,3} };
  }
  if(n == 4) {
    return { {0,4}, {0,1}, {0,2}, {2,4} };
  }
  if(n == 5) {
    return { {0,5}, {0,1}, {0,3}, {3,5} };
  }
  if(n == 6) {
    return { {0,6}, {0,1}, {0,4}, {0,2}, {2,4}, {4,6} };
  }
  if(n == 7) {
    return { {0,7}, {0,1}, {0,4}, {4,7} };
  }
  if(n == 8) {
    return { {0,8}, {0,1}, {0,4}, {4,8}, {0,2}, {2,4}, {4,6}, {6,8} };
  }
  throw std::runtime_error("not implemented");
}

tree_t::tree_t(
  graph_t const& g,
  cluster_t const& c)
  : graph(g), cluster(c), ordered_gids(g.get_order()),
    locsets(make_locsets(c.devices.size()))
{
  // insert the root node
  {
    int const& gid0 = ordered_gids[0];
    auto choices = partition_choices(gid0, {});
    nodes.push_back(node_t {
      .up = -1,
      .children = {},
      .which = 0,
      .choice = choices,
      .best_makespan = std::numeric_limits<double>::max()
    });
  }

  auto [makespan,leaf] = _step_init();

  best_makespan = makespan;
  best_leaf = leaf;
}

bool tree_t::step() {
  int n_before = nodes.size();

  auto [makespan, leaf] = _step();
  if(makespan < best_makespan) {
    best_makespan = makespan;
    best_leaf = leaf;
  }

  return n_before != nodes.size();
}

double tree_t::get_best_makespan() const {
  return best_makespan;
}

bool tree_t::is_leaf(int id) const {
  node_t const& node = nodes[id];
  return node.is_leaf();
}

// walk down the tree, always selecting
// (1) the default partition and (2) the
// {0,1} locset
tuple<double, int> tree_t::_step_init() {
  int n_gids = ordered_gids.size();

  map<int, partition_t> parts;

  int choice;
  int i = 0;
  for(int w = 0; w != n_gids; ++w) {
    {
      node_t const& node = nodes[i];
      int const& gid = ordered_gids[node.which];
      parts.insert({gid, node.child_parts()[0]});
      choice = 0;
    }
    i = get_part_child(i, choice);

    {
      node_t const& node = nodes[i];
      auto const& choices = node.child_places();
      locset_t l{0,1};
      for(choice = 0; choice != choices.size(); ++choice) {
        if(choices[choice].locset == l) {
          break;
        }
      }
      if(choice == choices.size()) {
        throw std::runtime_error("cannot find locset {0,1}");
      }
    }
    i = get_place_child(i, choice, parts);
  }

  double makespan = simulate(i);

  update_from_leaf(i, makespan);

  return {makespan, i};
}

tuple<double, int> tree_t::_step() {
  int n_gids = ordered_gids.size();

  map<int, partition_t> parts;

  int choice;
  int i = 0;
  for(int w = 0; w != n_gids; ++w) {
    choice = _step_select_which_choice(i);

    {
      auto const& node = nodes[i];
      int const& gid = ordered_gids[node.which];
      parts.insert({gid, node.child_parts()[choice]});
    }
    i = get_part_child(i, choice);

    choice = _step_select_which_choice(i);
    i = get_place_child(i, choice, parts);
  }

  double makespan = simulate(i);

  update_from_leaf(i, makespan);

  return {makespan, i};
}

int tree_t::_step_select_which_choice(int id) {
  // TODO: need a proper mechanism here to handle
  //       exploration and exploitation...
  //       just a quick guestimate for now...
  auto const& node = nodes[id];
  int n_children = node.num_possible_children();

  vector<double> scores;
  scores.reserve(n_children);
  for(int i = 0; i != n_children; ++i) {
    auto iter = node.children.find(i);
    if(iter == node.children.end()) {
      scores.push_back(1.0);
    } else {
      auto const& child_node = nodes[iter->second];
      if(child_node.best_makespan < 2.0 * best_makespan) {
        scores.push_back(0.5);
      } else {
        scores.push_back(2.0);
      }
    }
  }

  return runif(scores);
}

int tree_t::get_part_child(
  int id,
  int choice)
{
  node_t const& node = nodes[id];

  auto iter = node.children.find(choice);
  if(iter != node.children.end()) {
    return iter->second;
  }

  vector<place_choice_t> choices;
  for(auto const& locset: locsets) {
    choices.push_back(place_choice_t {
      .locset = locset,
      .by_agg_group = false,
      .with_load_balance = true
    });
  }

  nodes.push_back(node_t {
    .up = id,
    .children = {},
    .which = node.which,
    .choice = choices,
    .best_makespan = std::numeric_limits<double>::max()

  });

  int ret = nodes.size() - 1;
  nodes[id].children.insert({choice, ret});
  return ret;
}

int tree_t::get_place_child(
  int id,
  int choice,
  map<int,partition_t> const& parts_before)
{
  node_t const& node = nodes[id];

  auto iter = node.children.find(choice);
  if(iter != node.children.end()) {
    return iter->second;
  }

  if(node.which == ordered_gids.size()-1) {
    nodes.push_back(node_t {
      .up = id,
      .children = {},
      .which = -1,
      .choice = leaf_t {},
      .best_makespan = std::numeric_limits<double>::max()
    });
    int ret = nodes.size() - 1;
    node_t& node = nodes[id];
    node.children.insert({choice, ret});
    return ret;
  }

  vector<partition_t> parts;
  {
    vector<partition_t> inn_parts;
    int const& gid = ordered_gids[node.which + 1];
    auto const& gnode = graph.nodes[gid];
    for(int inn: gnode.inns) {
      inn_parts.push_back(parts_before.at(inn));
    }
    parts = partition_choices(gid, inn_parts);
  }

  nodes.push_back(node_t {
    .up = id,
    .children = {},
    .which = node.which + 1,
    .choice = parts,
    .best_makespan = std::numeric_limits<double>::max()
  });

  int ret = nodes.size() - 1;
  nodes[id].children.insert({choice, ret});
  return ret;
}

vector<tuple<int,int>>
tree_t::get_path_from_leaf(int id) const
{
  vector<tuple<int,int>> ret;
  do {
    node_t const& dwn = nodes[id];
    int up = dwn.up;
    node_t const& upp = nodes[up];
    bool found = false;
    for(auto const& [choice,a_dwn]: upp.children) {
      if(a_dwn == id) {
        ret.emplace_back(up, choice);
        found = true;
        break;
      }
    }
    if(!found) {
      throw std::runtime_error("should not reach: not found a choice");
    }
    id = up;
  } while(id != 0);

  std::reverse(ret.begin(), ret.end());

  return ret;
}

double tree_t::simulate(int leaf_id) const {
  vector<tuple<int,int>> choices = get_path_from_leaf(leaf_id);

  forward_state_t state(cluster, graph);

  // assign all partitions
  for(int w = 0; w != ordered_gids.size(); ++w) {
    int const& gid = ordered_gids[w];
    auto const& [id, choice] = choices[2*w];
    auto const& node = nodes[id];
    if(w != node.which) {
      throw std::runtime_error("simulate: w != node.which");
    }
    state.assign_partition(gid, node.child_parts()[choice]);
  }

  // assign all placements
  for(int w = 0; w != ordered_gids.size(); ++w) {
    int const& gid = ordered_gids[w];
    auto const& [id, choice] = choices[2*w + 1];
    auto const& node = nodes[id];
    if(w != node.which) {
      throw std::runtime_error("simulate: w != node.which");
    }
    assign_locations(state, gid, node.child_places()[choice]);
  }

  // do the entire simulation
  double makespan = 0.0;
  while(!state.all_done()) {
    state.enqueue_all();
    double finish = state.pop_work().finish;
    makespan = std::max(makespan, finish);
  }

  return makespan;
}

bool _is_mmlike(graph_t::node_t const& node) {
  if(node.op.is_einsummable()) {
    auto const& e = node.op.get_einsummable();
    return e.inns.size() > 1 && e.out_rank < e.join_shape.size();
  } else {
    return false;
  }
}

void _update_pds_and_choices_from_input_for_nonmmlike(
  vector<vector<partdim_t>>& pds,
  vector<partition_t>& choices,
  vector<int> const& is,
  vector<partdim_t> const& inn_partdims)
{
  if(inn_partdims.size() == pds.size()) {
    vector<partdim_t> choice_partdims(pds.size());
    for(int inn_idx = 0; inn_idx != is.size(); ++inn_idx) {
      int const& join_idx = is[inn_idx];
      auto const& inn_partdim = inn_partdims[inn_idx];
      choice_partdims[join_idx] = inn_partdim;
    }
    choices.emplace_back(choice_partdims);
  }

  for(int inn_idx = 0; inn_idx != is.size(); ++inn_idx) {
    int const& join_idx = is[inn_idx];
    auto const& inn_partdim = inn_partdims[inn_idx];
    pds[join_idx].push_back(inn_partdims[inn_idx]);
  }
}

vector<partition_t>
tree_t::partition_choices(
  int gid,
  vector<partition_t> const& inn_parts) const
{
  auto const& node = graph.nodes[gid];
  auto shape = node.op.shape();
  int const nlocs = cluster.devices.size();

  if(node.op.is_einsummable()) {
    vector<partition_t> ret;
    auto const& e = node.op.get_einsummable();
    vector<vector<partdim_t>> pds(shape.size());
    for(int which_inn = 0; which_inn != e.inns.size(); ++which_inn) {
      auto const& is = e.inns[which_inn];
      auto const& inn_partdims = inn_parts[which_inn].partdims;
      _update_pds_and_choices_from_input_for_nonmmlike(pds, ret, is, inn_partdims);
    }

    vector<partdim_t> join_partdims;
    join_partdims.reserve(shape.size());
    for(auto const& pd: pds) {
      join_partdims.push_back(partdim_t::unions(pd));
    }

    partition_t join_part(join_partdims);
    if(join_part.num_parts() < 4*cluster.devices.size()) {
      ret.push_back(join_part);
      std::swap(ret[0], ret.back());
    } else if(ret.size() == 0) {
      ret.push_back(join_part);
    }

    return vector_uniqueify(ret);
  }

  if(node.op.is_input()) {
    auto default_pds = partition_t::singleton(shape).partdims;

    vector<partition_t> ret;

    {
      vector<partdim_t> pds = default_pds;
      int total = 1;

      if(product(shape) > nlocs) {
        int d = 0;
        while(total < nlocs) {
          auto const& total_size = shape[d];
          int np = pds[d].num_parts();
          if(np != total_size) {
            pds[d] = partdim_t::split(total_size, np + 1);
            total /= np;
            total *= (np + 1);
          }
          d = (d + 1) % shape.size();
        }
      }

      ret.emplace_back(pds);
    }

    for(int d = 0; d != shape.size(); ++d) {
      if(shape[d] > nlocs) {
        auto pds = default_pds;
        pds[d] = partdim_t::split(shape[d], nlocs);
        ret.emplace_back(pds);
      }
    }

    return vector_uniqueify(ret);
  }

  if(node.op.is_formation()) {
    auto const& pds_inn = inn_parts[0].partdims;
    vector<partdim_t> partdims(pds_inn.begin(), pds_inn.begin() + shape.size());
    return vector<partition_t>(1, partition_t(partdims));
  }

  throw std::runtime_error("should not reach: partition_choices");
}

void tree_t::assign_locations(
  forward_state_t& state,
  int gid,
  place_choice_t const& p) const
{
  // TODO: implement this proper

  if(p.by_agg_group == true || p.with_load_balance == false) {
    throw std::runtime_error("no implemented assign locations");
  }

  auto const& [mn,mx] = p.locset;
  int nl = mx-mn;

  int l = 0;
  int nbid = state.num_join_bid(gid).value();
  for(int bid = 0; bid != nbid; ++bid) {
    state.assign_location({gid, bid}, mn+l);
    l = (l + 1) % nl;
  }
}

void tree_t::update_from_leaf(int leaf_id, double found_makespan)
{
  for(auto const& [id,_]: get_path_from_leaf(leaf_id)) {
    node_t& node = nodes[id];
    if(found_makespan < node.best_makespan) {
      node.best_makespan = found_makespan;
    }
  }

  node_t& node = nodes[leaf_id];
  if(found_makespan < node.best_makespan) {
    node.best_makespan = found_makespan;
  }
}

}

bool operator==(mcts1_ns::locset_t const& lhs, mcts1_ns::locset_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}


