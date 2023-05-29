#include "loadbalanceplace.h"
#include "locsetter.h"

struct _lbp_count_t {
  uint64_t size;
  int loc;
  int block_id;
};

inline bool operator<(
  _lbp_count_t const& lhs,
  _lbp_count_t const& rhs)
{
  return lhs.size < rhs.size;
}

bool is_balanced(vector<int> const& locs);

vector<_lbp_count_t> compute_loc_scores(
  vector<int> const& avail_locs,
  twolayergraph_t const& twolayer,
  twolayer_join_holder_t<int> const& placements,
  int gid,
  vector<int> const& bids);

vector<tensor_t<int>> load_balanced_placement(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool random_input)
{
  tuple<
    vector<tensor_t<int>>,
    equal_items_t<int>,
    twolayergraph_t> _info = twolayergraph_t::make(graph, parts);
  auto& [to_join_id, _, twolayer] = _info;

  twolayer_join_holder_t<int> placements =
    twolayer_join_holder_t<int>::make(to_join_id, -1);

  for(int graph_id: graph.get_order()) {
    auto& node = graph.nodes[graph_id];
    auto const& part = parts[graph_id];

    if(node.op.is_input()) {
      if(random_input) {
        vector<int> const& jids = placements.g_to_tl[graph_id].get();

        for(int const& jid: jids) {
          loc_setter_t setter(jids.size(), nlocs);
          int loc = setter.runif();
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
      if(part == parts[id_inn]) {
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
      auto const& part_inn = parts[id_inn];

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

    loc_setter_t setter(num_parts, nlocs);

    while(remaining.size() > 0) {
      // compute a three tuple of
      //   1. elements move if this location is chosen
      //   2. location
      //   3. blockid
      vector<_lbp_count_t> loc_scores = compute_loc_scores(
        setter.get_avail_locs(),
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
      scores.push_back(twolayer.count_elements_to(
        placements.items,
        jid,
        loc));
    }

    int which = std::min_element(scores.begin(), scores.end()) - scores.begin();
    ret.push_back(_lbp_count_t {
      .size = scores[which],
      .loc = avail_locs[which],
      .block_id = bid
    });
  }

  return ret;
}

