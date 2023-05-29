#include "mcts1.h"
#include "locsetter.h"

namespace mcts1_ns {

vector<locset_t> make_locsets(int n) {
  // TODO
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
  cluster_t const& c,
  double pc)
  : graph(g), cluster(c), param_c(pc), ordered_gids(g.get_order()),
    locsets(make_locsets(c.devices.size()))
{
  nodes.reserve(1000000);

  // insert the root node
  {
    int const& gid0 = ordered_gids[0];
    auto choices = partition_choices(gid0, {});
    nodes.push_back(node_t {
      .up = -1,
      .children = {},
      .which = 0,
      .choice = choices,
      .best_makespan = std::numeric_limits<double>::max(),
      .cumul = 0.0,
      .n = 0
    });
  }

  auto [makespan,leaf] = _step_init();

  best_makespan = makespan;
  best_leaf = leaf;
}

tuple<double,bool> tree_t::step() {
  int n_before = nodes.size();

  auto [makespan, leaf] = _step();
  if(makespan < best_makespan) {
    best_makespan = makespan;
    best_leaf = leaf;
  }

  return {makespan, n_before != nodes.size()};
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

  if(n_children != node.children.size()) {
    vector<double> scores;
    scores.reserve(n_children);
    for(int i = 0; i != n_children; ++i) {
      auto iter = node.children.find(i);
      if(iter == node.children.end()) {
        scores.push_back(1.0);
      } else {
        auto const& child_node = nodes[iter->second];
        if(child_node.best_makespan < 2.0 * best_makespan) {
          scores.push_back(0.0);
        } else {
          scores.push_back(0.0);
        }
      }
    }

    return runif(scores);
  }

  int best_choice = -1;
  double best_score = 0.0;
  for(auto const& [choice,child]: node.children) {
    double n_here = 1.0 * node.n;
    double n_child = 1.0 * nodes[child].n;
    double avg_makespan = nodes[child].cumul / n_child;
    double score = avg_makespan / best_makespan + param_c * std::sqrt(
               std::log( n_here ) / n_child );
    if(score > best_score) {
      best_choice = choice;
      best_score = score;
    }
  }

  return best_choice;
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

  vector<char> in_schemes;
  {
    int const& gid = ordered_gids[node.which];
    auto const& gnode = graph.nodes[gid];
    if(gnode.op.is_input()) {
      in_schemes.push_back(0);
    } else if(gnode.outs.size() == 0) {
      in_schemes.push_back(1);
    } else {
      in_schemes.push_back(0);
      in_schemes.push_back(1);
    }
  }

  vector<place_choice_t> choices;
  for(auto const& locset: locsets) {
    auto const& [mn,mx] = locset;
    int nl = mx-mn;
    if(nl == 1) {
      choices.push_back(place_choice_t {
        .locset = locset,
        .in_scheme = true // not used
      });
    } else {
      for(char s: in_schemes) {
        choices.push_back(place_choice_t {
          .locset = locset,
          .in_scheme = bool(s)
        });
      }
    }
  }

  nodes.push_back(node_t {
    .up = id,
    .children = {},
    .which = node.which,
    .choice = choices,
    .best_makespan = std::numeric_limits<double>::max(),
    .cumul = 0.0,
    .n = 0

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
      .best_makespan = std::numeric_limits<double>::max(),
      .cumul = 0.0,
      .n = 0
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
    .best_makespan = std::numeric_limits<double>::max(),
    .cumul = 0.0,
    .n = 0
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
  forward_state_t state = construct_state(leaf_id);

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
    vector<partition_t> ret;

    auto const& pds_inn = inn_parts[0].partdims;
    vector<partdim_t> partdims(pds_inn.begin(), pds_inn.begin() + shape.size());
    ret.emplace_back(partdims);

    if(pds_inn.begin() + shape.size() != pds_inn.end()) {
      int n_aggd;
      {
        int n_total = inn_parts[0].num_parts();
        partition_t _p(vector<partdim_t>(
          pds_inn.begin() + shape.size(),
          pds_inn.end()));
        n_aggd = n_total / _p.num_parts();
        int nloc = cluster.devices.size();
        n_aggd = std::min(n_aggd, nloc);
      }

      if(n_aggd > 1) {
        for(int d = 0; d != shape.size(); ++d) {
          auto const& pd = partdims[d];
          auto szs = pd.sizes();
          uint64_t sz = *std::min_element(szs.begin(), szs.end());
          if(sz >= n_aggd) {
            vector<partdim_t> ps = partdims;
            ps[d] = partdim_t::split_each(pd, n_aggd);
            ret.emplace_back(ps);
            break;
          }
        }
      }
    }

    return ret;
  }

  throw std::runtime_error("should not reach: partition_choices");
}

void tree_t::assign_locations(
  forward_state_t& state,
  int gid,
  place_choice_t const& p) const
{
  // Assumption: all partitions have been assigned to state
  // Assumption: all dependents of gid have been placed

  auto const& [mn,mx] = p.locset;
  int nl = mx-mn;
  int nbid = state.num_join_bid(gid).value();

  if(nl == 1) {
    for(int bid = 0; bid != nbid; ++bid) {
      state.assign_location({gid, bid}, mn);
    }
    return;
  }

  auto const& node = graph.nodes[gid];

  if(node.op.is_input() && p.in_scheme) {
    throw std::runtime_error("is input, must use output scheme");
  }
  if(node.outs.size() == 0 && !p.in_scheme) {
    throw std::runtime_error("is output, must use input scheme");
  }

  //if(node.outs.size() == 0) {
  //  int l = 0;
  //  int nbid = state.num_join_bid(gid).value();
  //  for(int bid = 0; bid != nbid; ++bid) {
  //    state.assign_location({gid, bid}, mn+l);
  //    l = (l + 1) % nl;
  //  }
  //  return;
  //}

  if(!p.in_scheme) {
    auto const& ginfo = state.get_ginfo(gid);
    auto const& joins = ginfo.joins.value();
    auto const& refis = ginfo.refis.value();
    auto const& locs = ginfo.locs.value();

    using jid_t = forward_state_t::jid_t;

    map<jid_t, vector<int>> items;
    for(int bid = 0; bid != nbid; ++bid) {
      auto const& join = joins[bid];
      for(auto const& refi_bid: join.outs) {
        auto const& refi = refis[refi_bid];
        for(auto const& out_jid: refi.outs) {
          items[out_jid].push_back(bid);
        }
      }
    }

    loc_setter_t loc_setter(nbid, nl);
    vector<char> is_set(nbid, 0);
    for(auto const& [_, bids]: items) {
      vector<int> cnts(nl, 0);
      for(int const& bid: bids) {
        int const& loc = locs[bid];
        if(loc >= 0) {
          cnts[loc-mn] += 1;
        }
      }

      vector<std::size_t> locs_order = argsort(cnts.begin(), cnts.end());

      auto iter = locs_order.begin();
      auto get_loc_minus_mn = [&]() {
        while(!loc_setter.is_avail(*iter)) {
          iter++;
        }
        return *iter;
      };

      for(int const& bid: bids) {
        if(locs[bid] == -1) {
          int l = get_loc_minus_mn();
          state.assign_location({gid,bid}, mn+l);
          loc_setter.decrement(l);
        }
      }
    }
  } else {
    loc_setter_t loc_setter(nbid, nl);

    for(int bid = 0; bid != nbid; ++bid) {
      vector<uint64_t> scores;
      scores.reserve(nl);
      for(int l = mn; l != mx; ++l) {
        scores.push_back(state.extra_elems_to({gid, bid}, l));
      }
      vector<std::size_t> locs_order = argsort(
        scores.begin(), scores.end());
      for(auto const& l: locs_order) {
        if(loc_setter.is_avail(l)) {
          state.assign_location({gid,bid}, mn+l);
          loc_setter.decrement(l);
          break;
        }
      }
    }
  }
}

void tree_t::update_from_leaf(int leaf_id, double found_makespan)
{
  for(auto const& [id,_]: get_path_from_leaf(leaf_id)) {
    node_t& node = nodes[id];
    node.cumul += found_makespan;
    node.n += 1;
    if(found_makespan < node.best_makespan) {
      node.best_makespan = found_makespan;
    }
  }

  node_t& node = nodes[leaf_id];
  node.cumul += found_makespan;
  node.n += 1;
  if(found_makespan < node.best_makespan) {
    node.best_makespan = found_makespan;
  }
}

forward_state_t tree_t::construct_best() const {
  return construct_state(best_leaf);
}

forward_state_t tree_t::construct_state(int leaf_id) const {
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

  return state;
}

}

bool operator==(mcts1_ns::locset_t const& lhs, mcts1_ns::locset_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}


