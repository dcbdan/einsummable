#include "base.h"
// #include <fstream>

void server_base_t::insert_gid_without_data(int gid, relation_t const& relation)
{
    auto iter = gid_map.find(gid);
    if (iter != gid_map.end()) {
        throw std::runtime_error("this gid is already in the server");
    }
    gid_map.insert({gid, relation});
}

void server_base_t::execute_graph(graph_t const&               graph,
                                  vector<placement_t> const&   placements,
                                  map<string, scalar_t> const& scalar_vars)
{
    auto make_relation = [&](int gid, vtensor_t<int> const& tids) {
        return relation_t{
            .dtype = graph.out_dtype(gid), .placement = placements[gid], .tids = tids};
    };

    auto [inn_g_to_t, out_g_to_t, taskgraph] = taskgraph_t::make(graph, placements);
    if (make_parallel_partialize_groups()) {
        for (auto& node : taskgraph.nodes) {
            auto& op = node.op;
            if (op.is_partialize()) {
                auto& partialize = op.get_partialize();
                partialize.make_parallel();
            }
        }
    }

    int      num_msgs = 0;
    uint64_t num_bytes = 0;
    for (auto const& node : taskgraph.nodes) {
        if (node.op.is_move()) {
            num_msgs++;
            num_bytes += node.op.get_move().size;
        }
    }
    DOUT("executing taskgraph with " << num_msgs << " moves, " << num_bytes << " bytes moved");

    {
        std::ofstream f("tg.gv");
        taskgraph.print_graphviz(f);
        DOUT("printed tg.gv");
    }

    // inn_g_to_t is input id to taskid in taskgraph
    // TODO: this remap(remap_relations_t r) kind of function signature only appear in
    // server_dist_base_t. We need to remove this
    remap_relations_t r;
    for (auto const& [gid, dst_tids] : inn_g_to_t) {
        // map the previous gid,relation to new gid,relation after we make taskgraph
        r.insert(get_relation(gid),           // src relation
                 make_relation(gid, dst_tids) // dst relation
        );
    }

    // auto remap_start_time = std::chrono::high_resolution_clock::now();

  remap(r);
  DOUT("----------Done Remap----------");

    // auto remap_end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> elapsed = remap_end_time - remap_start_time;
    // std::cout << "Remap elapsed time is " << elapsed.count() << " milliseconds" << std::endl;

    execute(taskgraph, scalar_vars);

    gid_map.clear();
    for (auto const& [gid, tids] : out_g_to_t) {
        gid_map.insert({gid, make_relation(gid, tids)});
    }
}

void server_base_t::execute(taskgraph_t const&           taskgraph,
                            map<int, relation_t> const&  new_gid_map,
                            map<string, scalar_t> const& scalar_vars)
{
    execute(taskgraph, scalar_vars);
    gid_map = new_gid_map;
}

dbuffer_t server_base_t::get_tensor_from_gid(int gid)
{
    return get_tensor(get_relation(gid));
}

vector<int> server_base_t::get_gids() const
{
    vector<int> ret;
    ret.reserve(gid_map.size());
    for (auto const& [gid, _] : gid_map) {
        ret.push_back(gid);
    }
    return ret;
}

relation_t const& server_base_t::get_relation(int gid) const
{
    return gid_map.at(gid);
}

void server_base_t::insert_constant(int gid, relation_t const& relation, scalar_t value)
{
    insert_constant_relation(relation, value);
    gid_map.insert({gid, relation});
}

void server_base_t::insert_tensor(int gid, relation_t const& dst_relation, dbuffer_t src_tensor)
{
    insert_relation(dst_relation, src_tensor);
    gid_map.insert({gid, dst_relation});
}

void server_base_t::insert_tensor(int gid, placement_t const& pl, dbuffer_t src_tensor)
{
    // get some new tids to use in the relation
    int t = get_max_tid() + 1;

    vtensor_t<int> tids(pl.block_shape());
    vector<int>&   ts = tids.get();
    std::iota(ts.begin(), ts.end(), t);

    relation_t relation{.dtype = src_tensor.dtype, .placement = pl, .tids = tids};

    insert_tensor(gid, relation, src_tensor);
}

void server_base_t::insert_tensor(int gid, vector<uint64_t> const& shape, dbuffer_t src_tensor)
{
    insert_tensor(gid, placement_t(partition_t::singleton(shape)), src_tensor);
}

void server_base_t::insert_constant(int gid, placement_t const& dst_pl, scalar_t value)
{
    int t = get_max_tid() + 1;

    vtensor_t<int> tids(dst_pl.block_shape());
    vector<int>&   ts = tids.get();
    std::iota(ts.begin(), ts.end(), t);

    relation_t dst_relation{.dtype = value.dtype, .placement = dst_pl, .tids = tids};

    insert_constant(gid, dst_relation, value);
}

void server_base_t::remap(map<int, relation_t> const& gid_to_new_relations)
{
    // Get all tids that are not going to be used and delete them.
    {
        vector<int> erase_gids;
        for (auto const& [gid, _] : gid_map) {
            if (gid_to_new_relations.count(gid) == 0) {
                erase_gids.push_back(gid);
            }
        }
        erase(erase_gids);
    }

    remap_relations_t r;
    for (auto const& [gid, new_rel] : gid_to_new_relations) {
        r.insert(gid_map.at(gid), new_rel);
    }

    remap(r);

    gid_map = gid_to_new_relations;
}

void server_base_t::remap_gids(vector<tuple<int, int>> const& remap)
{
    map<int, relation_t> ret;

    // Get all tids that are not going to be used and delete them.
    {
        set<int> src_gids;
        {
            auto tmp = vector_mapfst(remap);
            src_gids = set<int>(tmp.begin(), tmp.end());
        }
        vector<int> erase_gids;
        {
            for (auto const& [gid, _] : gid_map) {
                if (src_gids.count(gid) == 0) {
                    erase_gids.push_back(gid);
                }
            }
        }
        erase(erase_gids);
    }

    for (auto const& [src, dst] : remap) {
        ret.insert({dst, gid_map.at(src)});
    }

    gid_map = ret;
}

void server_base_t::erase(vector<int> const& gids)
{
    vector<tuple<int, int>> loc_tid_pairs;
    for (auto const& gid : gids) {
        auto const& rel = gid_map.at(gid);
        auto const& locs = rel.placement.locations.get();
        auto const& tids = rel.tids.get();
        vector_concatenate_into(loc_tid_pairs, vector_zip(locs, tids));
    }

    erase_tids(loc_tid_pairs);

    for (auto const& gid : gids) {
        gid_map.erase(gid);
    }
}

map<string, scalar_t> scalar_vars_from_wire(string const& s)
{
    std::stringstream     inn(s);
    map<string, scalar_t> ret;
    while (inn) {
        string name = istream_consume_alphanumeric_u(inn);
        if (name.size() == 0) {
            inn.get();
            if (inn) {
                throw std::runtime_error("parse fail");
            }
            break;
        }
        istream_expect(inn, "|");
        scalar_t scalar;
        inn >> scalar;
        istream_expect(inn, "|");
        ret.insert({name, scalar});
    }
    return ret;
}

string scalar_vars_to_wire(map<string, scalar_t> const& vars)
{
    std::stringstream out;
    for (auto const& [name, var] : vars) {
        out << name << "|" << var << "|";
    }
    return out.str();
}
