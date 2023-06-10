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

vector<_lbp_count_t> compute_loc_scores(
  vector<int> const& avail_locs,
  // bid,loc->score
  std::function<uint64_t(int,int)> get_score,
  vector<int> const& bids)
{
  using jid_t = forward_state_t::jid_t;

  vector<_lbp_count_t> ret;
  ret.reserve(bids.size());

  for(auto const& bid: bids) {
    vector<uint64_t> scores;
    scores.reserve(avail_locs.size());
    for(auto const& loc: avail_locs) {
      scores.push_back(get_score(bid,loc));
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

vector<placement_t> load_balanced_placement(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  bool random_input)
{
  // the forward_state_t object is just acting as a twolayer graph
  // here, so give it a dummy cluster that won't actually end up
  // being used
  cluster_t dummy_cluster = cluster_t::make(
    { cluster_t::device_t(1000000) },
    {});
  forward_state_t twolayer(dummy_cluster, graph);

  // Assign all the partitions in twolayer and
  // initialize pls
  vector<placement_t> pls;
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    auto const& part = parts[gid];

    twolayer.assign_partition(gid, part);

    tensor_t<int> locations(part.block_shape(), -1);
    pls.emplace_back(part, locations);
  }

  using jid_t = forward_state_t::jid_t;

  auto get_loc = [&pls, &nlocs](jid_t const& jid) {
    auto const& [gid,bid] = jid;
    int ret = pls[gid].locations.get()[bid];
    if(ret < 0 || ret > nlocs) {
      throw std::runtime_error("get_loc: invalid return value");
    }
    return ret;
  };

  for(int gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];

    auto& pl = pls[gid];
    tensor_t<int>& locations = pl.locations;
    vector<int>& locs = locations.get();

    auto const& part = parts[gid];

    if(node.op.is_input()) {
      if(random_input) {
        loc_setter_t setter(locs.size(), nlocs);
        for(int bid = 0; bid != locs.size(); ++bid) {
          int loc = setter.runif();
          locs[bid] = loc;
          setter.decrement(loc);
        }
      } else {
        // Just round robin assign input nodes
        int l = 0;
        for(int bid = 0; bid != locs.size(); ++bid) {
          locs[bid] = l;
          l = (l + 1) % nlocs;
        }
      }

      if(!is_balanced(locs)) {
        throw std::runtime_error("random assignment is not balanced");
      }

      continue;
    }

    if(node.op.is_formation()) {
      int const& id_inn = node.inns[0];
      if(part == parts[id_inn]) {
        // This is a formation node with an equivalently placed input,
        // so there is no other reasonable location than this one
        locations = pls[id_inn].locations;
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
        tensor_t<int> const& inn_locations = pls[id_inn].locations;
        do {
          vector<int> inn_index = e.get_input_from_join(join_index, 0);
          locations.at(join_index) = inn_locations.at(inn_index);
        } while(increment_idxs(join_shape, join_index));

        continue;
      }
    }

    // TODO: this is not taking into account the dtype
    auto get_score = [&](int bid, int loc) {
      return twolayer.count_elements_to(
        get_loc,
        jid_t { gid, bid },
        loc);
    };

    vector<int> remaining(locs.size());
    std::iota(remaining.begin(), remaining.end(), 0);

    loc_setter_t setter(locs.size(), nlocs);

    while(remaining.size() > 0) {
      vector<_lbp_count_t> loc_scores = compute_loc_scores(
        setter.get_avail_locs(),
        get_score,
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
  }

  // do some more checks
  for(auto const& pl: pls) {
    vector<int> const& locs = pl.locations.get();
    int mn = *std::min_element(locs.begin(), locs.end());
    int mx = *std::max_element(locs.begin(), locs.end());
    if(mn < 0 || mn >= nlocs || mx < 0 || mx >= nlocs) {
      throw std::runtime_error("invalid locs");
    }
    if(locs.size() != product(pl.partition.block_shape())) {
      throw std::runtime_error("invalid num locs");
    }
  }

  return pls;
}

