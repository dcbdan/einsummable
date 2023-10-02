#include "base.h"

void server_base_t::execute(
  graph_t const& graph,
  vector<placement_t> const& placements)
{
  auto make_relation = [&](int gid, vtensor_t<int> const& tids) {
    return relation_t {
      .dtype = graph.out_dtype(gid),
      .placement = placements[gid],
      .tids = tids
    };
  };

  auto [inn_g_to_t, out_g_to_t, taskgraph] =
    taskgraph_t::make(graph, placements);

  remap_relations_t r;
  for(auto const& [gid, dst_tids]: inn_g_to_t) {
    r.remap.emplace_back(
      get_relation(gid),             // src relation
      make_relation(gid, dst_tids)   // dst relation
    );
  }

  remap(r);

  execute(taskgraph);

  gid_map.clear();
  for(auto const& [gid, tids]: out_g_to_t) {
    gid_map.insert({gid, make_relation(gid, tids)});
  }
}

dbuffer_t server_base_t::get_tensor(int gid)
{
  return get_tensor(get_relation(gid));
}

relation_t const& server_base_t::get_relation(int gid) const
{
  return gid_map.at(gid);
}

void server_base_t::remap_gids(vector<tuple<int,int>> const& remap)
{
  map<int, relation_t> ret;

  for(auto const& [src,dst]: remap) {
    ret.insert({dst, gid_map.at(src)});
  }

  gid_map = ret;
}

