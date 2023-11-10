#pragma once
#include "../base/setup.h"

#include "twolayer.h"

struct builder_t;

struct relationwise3_t {
  struct rinfo_t {
    partition_t partition;
    vector<refinement_t> refis;
    vector<set<int>> locations;
  };
  struct ginfo_t {
    partition_t partition;
    vector<join_t> joins;
    vector<int> locations;

    optional<rinfo_t> rinfo;

    bool has_refinement() const { return bool(rinfo); }
  };
  // Note: all partition and refinement partitions are with respect to
  //       real dtypes

  relationwise3_t(
    int nlocs,
    uint64_t flops_per_byte_moved,
    graph_t const& graph,
    vector<partition_t> const& parts,
    bool set_inputs_everywhere = true);

  vector<placement_t> get_placements() const;
  placement_t get_placement_at(int gid) const;

  std::function<partition_t const&(int)> f_get_partition() const;
  std::function<partition_t const&(int)> f_get_refinement_partition() const;
  std::function<vector<refinement_t>&(int)> f_get_mutable_refis();

  refinement_t const& get_refi(rid_t const& rid) const {
    return ginfos[rid.gid].rinfo.value().refis[rid.bid];
  }
  join_t const& get_join(jid_t const& jid) const {
    return ginfos[jid.gid].joins[jid.bid];
  }

  int get_singleton_refi_loc(rid_t const& rid) const {
    set<int> const& s = get_refi_locs(rid);
    if(s.size() != 1) {
      DLINEOUT(rid << ": " << vector<int>(s.begin(), s.end()));
      throw std::runtime_error("is not singleton");
    }
    return *s.begin();
  }
  set<int> const& get_refi_locs(rid_t const& rid) const {
    return ginfos[rid.gid].rinfo.value().locations[rid.bid];
  }
  int const& get_join_loc(jid_t const& jid) const {
    return ginfos[jid.gid].locations[jid.bid];
  }

  set<int>& get_refi_locs(rid_t const& rid) {
    return ginfos[rid.gid].rinfo.value().locations[rid.bid];
  }
  int& get_join_loc(jid_t const& jid) {
    return ginfos[jid.gid].locations[jid.bid];
  }

  bool has_refinement(int const& gid) const {
    return ginfos[gid].has_refinement();
  }
  int get_num_refis(int const& gid) const {
    return ginfos[gid].rinfo.value().refis.size();
  }
  int get_num_joins(int const& gid) const {
    return ginfos[gid].locations.size();
  }

  void update_cost(
    vector<uint64_t>& cost,
    optional<einsummable_t> const& maybe,
    int loc) const;
  void update_cost(
    vector<uint64_t>& cost,
    uint64_t bytes,
    int src,
    int dst) const;

  void update(builder_t const& builder);

  int const nlocs;
  uint64_t const flops_per_byte_moved;
  graph_t const& graph;
  vector<ginfo_t> ginfos;
};

struct builder_t {
  builder_t(relationwise3_t const& rw);

  builder_t(relationwise3_t const& rw, vector<uint64_t> const& site_costs);

  builder_t(builder_t const& other);
  builder_t& operator=(builder_t const& other);

  vector<uint64_t> site_costs;
  map<jid_t, int> join_locs;
  map<rid_t, set<int>> refi_locs;

  uint64_t max_site_cost() const {
    return *std::max_element(site_costs.begin(), site_costs.end());
  }

  // Recursively get this join at this location. If this join was already set,
  // return the previous location
  int jid_at(jid_t const& jid, int loc, bool assert_ = false);

  // recursively make updates so that this rid has this loc
  void rid_at(rid_t const& rid, int loc);

  relationwise3_t const& rw;

  optional<int> get_join_loc( jid_t const& jid) const;
  set<int>      get_refi_locs(rid_t const& rid) const;

  refinement_t const& get_refi(rid_t const& rid) const { return rw.get_refi(rid); }
  join_t       const& get_join(jid_t const& jid) const { return rw.get_join(jid); }
};

vector<placement_t> autolocate_bipartite(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved);

vector<placement_t> autolocate_agg_at_a_time_from_inns_v2(
  graph_t const& graph,
  vector<partition_t> const& parts,
  int nlocs,
  uint64_t flops_per_byte_moved);

