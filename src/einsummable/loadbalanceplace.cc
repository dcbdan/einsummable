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
  twolayergraph_t const& twolayer,
  map<twolayergraph_t::gid_t, int> const& to_join_id,
  vector<int> const& avail_locs,
  int graph_id,
  vector<int> const& block_ids);

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

void load_balanced_placement(
  graph_t& graph,
  int nlocs)
{
  // twolayer contains graph by const reference
  twolayergraph_t twolayer = twolayergraph_t::make(graph);

  // [graph id, block id] -> join id
  map<twolayergraph_t::gid_t, int> to_join_id;
  for(int jid=0; jid != twolayer.joins.size(); ++jid) {
    auto const& join = twolayer.joins[jid];
    to_join_id.insert({join.gid, jid});
  }

  for(int graph_id: graph.get_order()) {
    auto& node = graph.nodes[graph_id];

    if(node.op.is_input()) {
      // Just round robin assign input nodes

      vector<int>& locs = node.placement.locations.get();

      int l = 0;
      for(auto& loc: locs) {
        loc = l;
        l = (l + 1) % nlocs;
      }

      continue;
    }

    if(node.op.is_formation()) {
      int const& id_inn = node.inns[0];
      auto const& node_inn = graph.nodes[id_inn];
      if(node.placement.partition == node_inn.placement.partition) {
        // This is a formation node with an equivalently placed input,
        // so there is no other reasonable location than this one
        vector<int>& locs = node.placement.locations.get();
        locs = node_inn.placement.locations.get();
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

        auto&       place     = node.placement;
        auto const& place_inn = node_inn.placement;

        auto inn_shape  = part_inn.block_shape();
        auto join_shape = part.block_shape();
        vector<int> join_index(join_shape.size(), 0);
        do {
          vector<int> inn_index = e.get_input_from_join(join_index, 0);
          place.at(join_index) = place_inn.at(inn_index);
        } while(increment_idxs(join_shape, join_index));

        continue;
      }
    }

    vector<int>& locs = node.placement.locations.get();

    int num_parts = node.placement.num_parts();
    vector<int> remaining(num_parts);
    std::iota(remaining.begin(), remaining.end(), 0);

    _loc_setter_t setter(num_parts, nlocs);

    while(remaining.size() > 0) {
      vector<_lbp_count_t> loc_scores = compute_loc_scores(
        twolayer, to_join_id, setter.avail_locs, graph_id, remaining);
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
  }
}

vector<_lbp_count_t> compute_loc_scores(
  twolayergraph_t const& twolayer,
  map<twolayergraph_t::gid_t, int> const& to_join_id,
  vector<int> const& avail_locs,
  int graph_id,
  vector<int> const& block_ids)
{
  vector<_lbp_count_t> ret;
  ret.reserve(block_ids.size());

  for(auto const& bid: block_ids) {
    auto const& jid = to_join_id.at({graph_id, bid});

    vector<uint64_t> scores;
    scores.reserve(avail_locs.size());
    for(auto const& loc: avail_locs) {
      scores.push_back(twolayer.count_bytes_to(jid, loc));
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


