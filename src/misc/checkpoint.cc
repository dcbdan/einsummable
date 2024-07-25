#include "checkpoint.h"

void graph_id_manager_t::insert(int which, int fid, int sid)
{
    auto& [fid_to_sid, sid_to_fid] = data[which];
    {
        auto [_, did_insert] = fid_to_sid.insert({fid, sid});
        if (!did_insert) {
            throw std::runtime_error("graph_id_manager did not insert: fid_to_sid");
        }
    }
    {
        auto [_, did_insert] = sid_to_fid.insert({sid, fid});
        if (!did_insert) {
            throw std::runtime_error("graph_id_manager did not insert: sid_to_fid");
        }
    }
}

optional<int> graph_id_manager_t::get_sid(int which, int fid) const
{
    auto diter = data.find(which);
    if (diter == data.end()) {
        return std::nullopt;
    }
    auto const& m = std::get<0>(diter->second);
    auto        iter = m.find(fid);
    if (iter == m.end()) {
        return std::nullopt;
    }
    return iter->second;
}

optional<int> graph_id_manager_t::get_fid(int which, int sid) const
{
    auto diter = data.find(which);
    if (diter == data.end()) {
        return std::nullopt;
    }
    auto const& m = std::get<1>(diter->second);
    auto        iter = m.find(sid);
    if (iter == m.end()) {
        return std::nullopt;
    }
    return iter->second;
}

optional<int> graph_id_manager_t::get_sid_from_sid(int which_src, int src_sid, int which_dst) const
{
    auto maybe = get_fid(which_src, src_sid);
    if (maybe) {
        return get_sid(which_dst, maybe.value());
    } else {
        return std::nullopt;
    }
}

void graph_id_manager_t::print() const
{
    for (auto const& [which, ms] : data) {
        auto const& [fid_to_sid, _] = ms;
        for (auto const& [fid, sid] : fid_to_sid) {
            DOUT("fid " << fid << " | (which " << which << ": " << sid << ")");
        }
    }
}

static int set_label(map<int, int>&     ret,
                     graph_t const&     graph,
                     vector<int> const& checkpoints,
                     set<int> const&    forward_ids,
                     int                gid)
{
    // Invariant: forward_ids.count(gid) > 0

    {
        auto iter = ret.find(gid);
        if (iter != ret.end()) {
            return iter->second;
        }
    }

    {
        auto iter = std::find(checkpoints.begin(), checkpoints.end(), gid);
        if (iter != checkpoints.end()) {
            int val = std::distance(checkpoints.begin(), iter);
            ret.insert({gid, val});
            return val;
        }
    }

    int         val = -1;
    auto const& node = graph.nodes[gid];
    for (int const& out_gid : node.outs) {
        if (forward_ids.count(out_gid) > 0) {
            val = std::max(val, set_label(ret, graph, checkpoints, forward_ids, out_gid));
        }
    }

    if (val == -1) {
        val = checkpoints.size();
    }

    ret.insert({gid, val});
    return val;
}

static map<int, int> build_forward_labels(graph_t const&     graph,
                                          vector<int> const& checkpoints,
                                          set<int> const&    forward_ids)
{
    map<int, int> ret;
    for (int const& gid : forward_ids) {
        set_label(ret, graph, checkpoints, forward_ids, gid);
    }
    return ret;
}

static int get_min_checkpoint(graph_t const&       graph,
                              map<int, int> const& forward_labels,
                              map<int, int> const& backward_labels,
                              int                  id,
                              int                  ret)
{
    {
        auto iter = forward_labels.find(id);
        if (iter != forward_labels.end()) {
            return std::min(ret, iter->second);
        }
    }
    {
        auto iter = backward_labels.find(id);
        if (iter != backward_labels.end()) {
            return std::min(ret, iter->second);
        }
    }

    auto const& node = graph.nodes[id];
    for (int const& inn_id : node.get_inns_set()) {
        ret = get_min_checkpoint(graph, forward_labels, backward_labels, inn_id, ret);
    }

    return ret;
}

struct subgraph_former_t {
    graph_t const&       full_graph;
    map<int, int> const& forward_labels;
    map<int, int> const& backward_labels;
    int                  num_forward_sections;

    bool is_forward_section_input(int fid) const
    {
        int         which = get_forward_section(fid);
        auto const& node = full_graph.nodes[fid];

        if (node.inns.size() == 0) {
            return true;
        }

        if (node.op.is_formation()) {
            int inn_fid = node.inns[0];
            if (get_forward_section(inn_fid) != which) {
                throw std::runtime_error("division across formation edge!");
            }
            return is_forward_section_input(inn_fid);
        }

        for (int const& inn_fid : node.get_inns_set()) {
            if (get_forward_section(inn_fid) != which) {
                return true;
            }
        }

        return false;
    }

    int copy_from_forward_section_inputs(graph_id_manager_t::context_t id_context,
                                         graph_t&                      ret,
                                         int                           fid) const
    {
        {
            auto maybe = id_context.get(fid);
            if (maybe) {
                return maybe.value();
            }
        }

        if (is_forward_section_input(fid)) {
            auto const& op = full_graph.nodes[fid].op;
            int         nid = ret.insert_input(op.out_shape(), op.out_dtype());
            id_context.insert(fid, nid);
            return nid;
        }

        auto const& node = full_graph.nodes[fid];
        vector<int> new_inns;
        for (int const& inn_fid : node.inns) {
            int new_inn = copy_from_forward_section_inputs(id_context, ret, inn_fid);
            new_inns.push_back(new_inn);
        }

        int nid = ret.insert(node.op, new_inns);
        id_context.insert(fid, nid);
        return nid;
    }

    int get_backward_section_nid(int                           which,
                                 graph_id_manager_t::context_t id_context,
                                 graph_t&                      ret,
                                 int                           fid) const
    {
        {
            auto maybe = id_context.get(fid);
            if (maybe) {
                return maybe.value();
            }
        }

        // Assumption: At this point, fid is not on backward section `which`

        auto [is_forward, fid_which] = get_section(fid);
        if (!is_forward && fid_which == which) {
            throw std::runtime_error("this fid should've been in the manager");
        }

        // If this node is
        // 1. from a backward section or
        // 2. from the last forward section,
        // Then add it as an input and depend on it directly

        bool add_as_input = false;
        if (is_forward && fid_which + 1 == num_forward_sections) {
            add_as_input = true;
        }
        if (!is_forward) {
            add_as_input = true;
        }

        if (add_as_input) {
            auto const& op = full_graph.nodes[fid].op;
            int         nid = ret.insert_input(op.out_shape(), op.out_dtype());
            id_context.insert(fid, nid);
            return nid;
        }

        // Otherwise, recurse until hitting an input from the
        // dependent forward-section.

        return copy_from_forward_section_inputs(id_context, ret, fid);
    }

    graph_t form(graph_id_manager_t::context_t id_context, int which) const
    {
        graph_t ret;
        for (int const& fid : get_ordered_section(false, which)) {
            auto const& node = full_graph.nodes[fid];
            vector<int> new_inns;
            for (int const& inn_fid : node.inns) {
                int inn_nid = get_backward_section_nid(which, id_context, ret, inn_fid);
                new_inns.push_back(inn_nid);
            }
            int nid = ret.insert(node.op, new_inns);
            id_context.insert(fid, nid);
        }

        return ret;
    }

    graph_t forward(graph_id_manager_t::context_t id_context) const
    {
        graph_t ret;
        for (int const& fid : full_graph.get_flop_order()) {
            if (forward_labels.count(fid) == 0) {
                continue;
            }
            auto const& node = full_graph.nodes[fid];
            vector<int> new_inns;
            for (int const& inn : node.inns) {
                new_inns.push_back(id_context.get(inn).value());
            }
            int nid = ret.insert(node.op, new_inns);
            id_context.insert(fid, nid);
        }

        return ret;
    }

    tuple<bool, int> get_section(int fid) const
    {
        {
            auto iter = forward_labels.find(fid);
            if (iter != forward_labels.end()) {
                return {true, iter->second};
            }
        }

        {
            auto iter = backward_labels.find(fid);
            if (iter != backward_labels.end()) {
                return {false, iter->second};
            }
        }

        throw std::runtime_error("should not occur");
    }

    int get_forward_section(int fid) const
    {
        return forward_labels.at(fid);
    }

    int get_backward_section(int fid) const
    {
        return backward_labels.at(fid);
    }

    vector<int> get_ordered_section(bool forward, int which) const
    {
        vector<int> ret;
        for (int const& fid : full_graph.get_flop_order()) {
            auto [is_f, section] = get_section(fid);
            if (section != which) {
                continue;
            }
            if (is_f && forward) {
                ret.push_back(fid);
            } else if (!is_f && !forward) {
                ret.push_back(fid);
            }
        }
        return ret;
    }
};

struct remap_former_t {
    graph_t const& full_graph;

    graph_id_manager_t& id_manager;
    vector<graph_t>&    graphs;

    void turn_off_saves()
    {
        for (auto& graph : graphs) {
            for (int gid = 0; gid != graph.nodes.size(); ++gid) {
                auto& node = graph.nodes[gid];
                node.op.set_save(false);
            }
        }
    }

    // Make sure that fid exists on this graph and is a save
    int set_save(int which, int full_id)
    {
        {
            auto maybe = id_manager.get_sid(which, full_id);
            if (maybe) {
                int const& sid = maybe.value();
                graphs[which].nodes[sid].op.set_save(true);
                return sid;
            }
        }

        // In this case, this graph does not have the save node so it needs
        // to get the save node from the previous graph.
        if (which == 0) {
            throw std::runtime_error("can't set save, missing node");
        }

        // get the node from the previous graph
        int prev_id = set_save(which - 1, full_id);

        // insert the new node as an input
        auto const& prev_node = graphs[which - 1].nodes[prev_id];
        int         curr_id =
            graphs[which].insert_input(prev_node.op.out_shape(), prev_node.op.out_dtype());

        // and make sure to save it
        graphs[which].nodes[curr_id].op.set_save(true);

        // and make sure to update the id manager
        id_manager.insert(which, full_id, curr_id);

        return curr_id;
    }

    map<int, int> remap_saves_to_full()
    {
        int           which = graphs.size() - 1;
        map<int, int> ret;
        for (int fid = 0; fid != full_graph.nodes.size(); ++fid) {
            auto const& fnode = full_graph.nodes[fid];
            if (fnode.op.is_save()) {
                int sid = set_save(which, fid);
                ret.insert({sid, fid});
            }
        }
        return ret;
    }

    map<int, int> remap_inputs_from_full() const
    {
        map<int, int>  ret;
        graph_t const& g = graphs[0];
        for (int sid = 0; sid != g.nodes.size(); ++sid) {
            auto const& node = g.nodes[sid];
            if (node.op.is_input()) {
                int fid = id_manager.get_fid(0, sid).value();
                ret.insert({fid, sid});
            }
        }
        return ret;
    }

    map<int, int> remap_to(int which_dst)
    {
        if (which_dst == 0) {
            throw std::runtime_error("invalid which_dst");
        }
        map<int, int>  ret;
        graph_t const& dst_g = graphs[which_dst];
        for (int dst_id = 0; dst_id != dst_g.nodes.size(); ++dst_id) {
            auto const& dst_node = dst_g.nodes[dst_id];
            if (dst_node.op.is_input()) {
                int full_id = id_manager.get_fid(which_dst, dst_id).value();
                int src_id = set_save(which_dst - 1, full_id);
                ret.insert({src_id, dst_id});
            }
        }
        return ret;
    }
};

vector<map<int, int>>
form_all_remaps(graph_t const& full_graph, graph_id_manager_t& id_manager, vector<graph_t>& graphs)
{
    remap_former_t former{.full_graph = full_graph, .id_manager = id_manager, .graphs = graphs};

    former.turn_off_saves();

    vector<map<int, int>> ret;

    // Note: to get the full remaps, it is very important to do
    //       the remaps in reverse, since then all the saves will
    //       be properly added

    ret.push_back(former.remap_saves_to_full());

    for (int i = graphs.size() - 1; i >= 1; --i) {
        ret.push_back(former.remap_to(i));
    }

    ret.push_back(former.remap_inputs_from_full());

    std::reverse(ret.begin(), ret.end());
    return ret;
}

checkpoint_graphs_t::checkpoint_graphs_t(graph_t const&     full_graph,
                                         vector<int> const& checkpoints,
                                         set<int> const&    forward_ids)
{
    map<int, int> forward_labels = build_forward_labels(full_graph, checkpoints, forward_ids);

    map<int, int> backward_labels;
    int           max_backward_label = -1;
    for (int id = 0; id != full_graph.nodes.size(); ++id) {
        auto iter = forward_labels.find(id);
        if (iter != forward_labels.end()) {
            int cp = iter->second;
        } else {
            int cpoint = get_min_checkpoint(
                full_graph, forward_labels, backward_labels, id, checkpoints.size());
            backward_labels.insert({id, cpoint});
            max_backward_label = std::max(max_backward_label, cpoint);
        }
    }

    subgraph_former_t former{.full_graph = full_graph,
                             .forward_labels = forward_labels,
                             .backward_labels = backward_labels,
                             .num_forward_sections = int(checkpoints.size())};

    graphs.push_back(former.forward(manager.make_context(0)));

    for (int i = max_backward_label; i >= 0; --i) {
        int which = graphs.size();
        graphs.push_back(former.form(manager.make_context(which), i));
    }

    vector<map<int, int>> _remaps = form_all_remaps(full_graph, manager, graphs);
    remaps.reserve(_remaps.size());
    for (auto const& _r : _remaps) {
        remaps.emplace_back(_r.begin(), _r.end());
    }
}

checkpoint_taskgraphs_t::checkpoint_taskgraphs_t(checkpoint_graphs_t const& gs,
                                                 vector<placement_t> const& full_pls)
{
    for (int which = 0; which != gs.graphs.size(); ++which) {
        graph_t const& graph = gs.graphs[which];

        vector<placement_t> pls;
        pls.reserve(graph.nodes.size());
        for (int sid = 0; sid != graph.nodes.size(); ++sid) {
            pls.push_back(full_pls.at(gs.manager.get_fid(which, sid).value()));
        }

        auto make_rels = [&](map<int, vtensor_t<int>> const& mtids) {
            map<int, relation_t> ret;
            for (auto const& [sid, tids] : mtids) {
                ret.insert({sid,
                            relation_t{.dtype = graph.out_dtype(sid),
                                       .placement = pls[sid],
                                       .tids = tids}});
            }
            return ret;
        };

        auto [inn_tids, save_tids, tg] = taskgraph_t::make(graph, pls);
        infos.push_back(info_t{.init_rel = make_rels(inn_tids),
                               .taskgraph = std::move(tg),
                               .save_rel = make_rels(save_tids)});
    }
}
