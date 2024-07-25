#include "twolayer.h"

struct relationwise_t {
    struct ginfo_t {
        partition_t    partition;
        vector<join_t> joins;
        vector<int>    locations;

        optional<partition_t>          refinement_partition;
        optional<vector<refinement_t>> refis;

        bool has_refinement() const
        {
            return bool(refinement_partition);
        }
    };
    // Note: all partition and refinement partitions are with respect to
    //       real dtypes

    relationwise_t(graph_t const& graph, vector<partition_t> const& parts);

    vector<placement_t> get_placements() const;
    placement_t         get_placement_at(int gid) const;

    std::function<partition_t const&(int)>    f_get_partition() const;
    std::function<partition_t const&(int)>    f_get_refinement_partition() const;
    std::function<vector<refinement_t>&(int)> f_get_mutable_refis();

    int num_agg_blocks_at(int gid) const;

    void print_info() const;

    set<int> get_refi_usage_locs(rid_t const& rid) const;
    uint64_t get_refi_bytes(rid_t const& rid) const;
    uint64_t get_join_out_bytes(jid_t const& jid) const;

    int& get_loc(jid_t const& jid)
    {
        return ginfos[jid.gid].locations[jid.bid];
    }
    int const& get_loc(jid_t const& jid) const
    {
        return ginfos[jid.gid].locations[jid.bid];
    }

    refinement_t const& get_refi(rid_t const& rid) const
    {
        return ginfos[rid.gid].refis.value()[rid.bid];
    }
    join_t const& get_join(jid_t const& jid) const
    {
        return ginfos[jid.gid].joins[jid.bid];
    }

    graph_t const&  graph;
    vector<ginfo_t> ginfos;
};
