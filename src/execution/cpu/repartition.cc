#include "repartition.h"

#include "execute.h"

#include "../../einsummable/graph.h"
#include "../../einsummable/taskgraph.h"

void repartition(
  mpi_t* mpi,
  remap_relations_t const& _remap,
  map<int, buffer_t>& data)
{
  auto const& remap = _remap.remap;

  graph_constructor_t g;

  vector<tuple<int,int>> remap_gid;
  for(auto const& [src,dst]: remap) {
    int gid_src = g.insert_input(src.placement, src.dtype);
    int gid_dst = g.insert_formation(dst.placement, gid_src, true);
    remap_gid.emplace_back(gid_src, gid_dst);
  }

  auto [gid_to_inn, gid_to_out, taskgraph] = taskgraph_t::make(
    g.graph, g.get_placements());

  {
    map<int, buffer_t> tmp;
    for(int i = 0; i != remap_gid.size(); ++i) {
      auto const& gid  = std::get<0>(remap_gid[i]);
      auto const& info = std::get<0>(remap[i]);
      vector<int> const& locs = info.placement.locations.get();
      vector<int> const& inn_tids = info.tids.get();
      vector<int> const& mid_tids = gid_to_inn.at(gid).get();
      if(inn_tids.size() != mid_tids.size()) {
        throw std::runtime_error("!");
      }
      for(int j = 0; j != inn_tids.size(); ++j) {
        if(mpi && locs[j] == mpi->this_rank) {
          int const& inn_tid = inn_tids[j];
          int const& mid_tid = mid_tids[j];
          tmp.insert({mid_tid, data.at(inn_tid)});
        }
      }
    }
    data = tmp;
  }

  {
    auto settings = execute_taskgraph_settings_t::only_touch_settings();
    kernel_manager_t ks;
    execute(taskgraph, settings, ks, mpi, data);
  }

  map<int, buffer_t> tmp;
  for(int i = 0; i != remap_gid.size(); ++i) {
    auto const& gid  = std::get<1>(remap_gid[i]);
    auto const& info = std::get<1>(remap[i]);
    vector<int> const& locs = info.placement.locations.get();
    vector<int> const& out_tids = info.tids.get();
    vector<int> const& mid_tids = gid_to_out.at(gid).get();
    for(int j = 0; j != out_tids.size(); ++j) {
      if(mpi && locs[j] == mpi->this_rank) {
        int const& out_tid = out_tids[j];
        int const& mid_tid = mid_tids[j];
        tmp.insert({out_tid, data.at(mid_tid)});
      }
    }
  }
  data = tmp;
}

