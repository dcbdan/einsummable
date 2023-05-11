#include "loadbalanceplace.h"

struct _lbp_count_t {
  uint64_t bytes;
  int loc;
  int block_id;
};

inline bool operator<(
  _lbp_count_t const& lhs,
  _lbp_count_t const& rhs)
{
  return lhs.bytes < rhs.bytes;
}

bool is_balanced(vector<int> const& locs);

vector<_lbp_count_t> compute_loc_scores(
  vector<int> const& avail_locs,
  twolayergraph_t const& twolayer,
  twolayer_join_holder_t<int> const& placements,
  int gid,
  vector<int> const& bids);

struct _loc_setter_t {
  _loc_setter_t(int n_items, int n_locs) {
    avail_locs = vector<int>(n_locs);
    std::iota(avail_locs.begin(), avail_locs.end(), 0);

    int cnt = (n_items / n_locs) + 1;
    remainder = n_items % n_locs;
    num_remaining = vector<int>(n_locs, cnt);
  }

  // return whether or not this loc is still available
  bool decrement(int loc) {
    auto al_iter = std::find(avail_locs.begin(), avail_locs.end(), loc);
    if(al_iter == avail_locs.end()) {
      throw std::runtime_error("invalid decrement: this loc isn't here");
    }
    int which = al_iter - avail_locs.begin();

    auto nr_iter = num_remaining.begin() + which;

    int& nr = *nr_iter;
    nr -= 1;

    if(nr == 0 && remainder == 0) {
      throw std::runtime_error("should not happen: loc setter");
    } else if(nr == 0 && remainder > 0) {
      remainder -= 1;
      num_remaining.erase(nr_iter);
      avail_locs.erase(al_iter);
      if(remainder == 0) {
        // remove all locs with nr == 1 now that the remainder
        // has run out
        for(int i = avail_locs.size() - 1; i >= 0; --i) {
          if(num_remaining[i] == 1) {
            num_remaining.erase(num_remaining.begin() + i);
            avail_locs.erase(avail_locs.begin() + i);
          }
        }
      }
      return false;
    } else if(nr == 1 && remainder == 0) {
      num_remaining.erase(nr_iter);
      avail_locs.erase(al_iter);
      return false;
    } else {
      return true;
    }
  }

  vector<int> num_remaining;
  int remainder;

  vector<int> avail_locs;
};

vector<tensor_t<int>> load_balanced_placement(
  graph_t const& graph,
  int nlocs,
  bool random_input)
{
  tuple<
    vector<tensor_t<int>>,
    equal_items_t<int>,
    twolayergraph_t> _info = twolayergraph_t::make(graph);
  auto& [to_join_id, _, twolayer] = _info;

  twolayer_join_holder_t<int> placements =
    twolayer_join_holder_t<int>::make(to_join_id, -1);

  for(int graph_id: graph.get_order()) {
    auto& node = graph.nodes[graph_id];

    if(node.op.is_input()) {
      if(random_input) {
        vector<int> const& jids = placements.g_to_tl[graph_id].get();

        for(int const& jid: jids) {
          _loc_setter_t setter(jids.size(), nlocs);
          int which_loc = runif(setter.avail_locs.size());
          int loc = setter.avail_locs[which_loc];
          placements.items[jid] = loc;
          setter.decrement(loc);
        }

        if(!is_balanced(placements.get_vector_at_gid(graph_id))) {
          throw std::runtime_error("random assignment is not balanced");
        }
      } else {
        // Just round robin assign input nodes
        vector<int> const& jids = placements.g_to_tl[graph_id].get();

        int l = 0;
        for(auto const& jid: jids) {
          placements.items[jid] = l;
          l = (l + 1) % nlocs;
        }
      }

      continue;
    }

    if(node.op.is_formation()) {
      int const& id_inn = node.inns[0];
      auto const& node_inn = graph.nodes[id_inn];
      if(node.placement.partition == node_inn.placement.partition) {
        // This is a formation node with an equivalently placed input,
        // so there is no other reasonable location than this one
        vector<int> locs = placements.get_vector_at_gid(id_inn);
        placements.set_at_gid(graph_id, locs);
        continue;
      }
    }

    if(node.op.is_einsummable() &&
       node.inns.size() == 1 &&
       !node.op.has_aggregation())
    {
      auto const& e = node.op.get_einsummable();

      int const& id_inn = node.inns[0];
      auto const& node_inn = graph.nodes[id_inn];

      auto const& part     = node.placement.partition;
      auto const& part_inn = node_inn.placement.partition;

      auto partdims_with_respect_to_inn =
        e.get_input_from_join(part.partdims, 0);

      if(part_inn.partdims == partdims_with_respect_to_inn) {
        // now we have to reach past the permutation
        auto join_shape = part.block_shape();
        vector<int> join_index(join_shape.size(), 0);
        do {
          vector<int> inn_index = e.get_input_from_join(join_index, 0);
          placements.get_at_gid(graph_id, join_index) =
            placements.get_at_gid(graph_id, inn_index);
        } while(increment_idxs(join_shape, join_index));

        continue;
      }
    }

    int num_parts = placements.block_size(graph_id);
    vector<int> locs(num_parts, -1);

    vector<int> remaining(num_parts);
    std::iota(remaining.begin(), remaining.end(), 0);

    _loc_setter_t setter(num_parts, nlocs);

    while(remaining.size() > 0) {
      // compute a three tuple of
      //   1. bytes move if this location is chosen
      //   2. location
      //   3. blockid
      vector<_lbp_count_t> loc_scores = compute_loc_scores(
        setter.avail_locs,
        twolayer,
        placements,
        graph_id,
        remaining);
      std::sort(loc_scores.begin(), loc_scores.end());

      vector<int> new_remaining;
      bool ran_out = false;
      for(auto const& [_, loc, bid]: loc_scores) {
        if(ran_out) {
          new_remaining.push_back(bid);
        } else {
          locs[bid] = loc;
          bool loc_still_avail = setter.decrement(loc);
          ran_out = !loc_still_avail;
        }
      }

      remaining = new_remaining;
    }

    if(!is_balanced(locs)) {
      throw std::runtime_error("is not balanced!");
    }

    placements.set_at_gid(graph_id, locs);
  }

  return placements.as_graph_repr();
}

vector<_lbp_count_t> compute_loc_scores(
  vector<int> const& avail_locs,
  twolayergraph_t const& twolayer,
  twolayer_join_holder_t<int> const& placements,
  int gid,
  vector<int> const& bids)
{
  vector<_lbp_count_t> ret;
  ret.reserve(bids.size());

  vector<int> const& to_jid = placements.g_to_tl[gid].get();

  for(auto const& bid: bids) {
    int const& jid = to_jid[bid];
    vector<uint64_t> scores;
    scores.reserve(avail_locs.size());
    for(auto const& loc: avail_locs) {
      scores.push_back(twolayer.count_bytes_to(
        placements.items,
        jid,
        loc));
    }

    int which = std::min_element(scores.begin(), scores.end()) - scores.begin();
    ret.push_back(_lbp_count_t {
      .bytes = scores[which],
      .loc = avail_locs[which],
      .block_id = bid
    });
  }

  return ret;
}

bool is_balanced(vector<int> const& locs) {
  vector<int> cnts;
  for(auto const& loc: locs) {
    if(cnts.size() <= loc) {
      cnts.resize(loc + 1);
    }
    cnts[loc] += 1;
  }
  int const& mx = *(std::max_element(cnts.begin(), cnts.end()));
  int const& mn = *(std::min_element(cnts.begin(), cnts.end()));
  return mx-mn <= 1;
}


