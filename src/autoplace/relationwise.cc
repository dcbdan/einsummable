#include "relationwise.h"

relationwise_t::relationwise_t(
  graph_t const& g,
  vector<partition_t> const& parts)
  : graph(g)
{
  std::function<partition_t const&(int)> get_partition =
    [&parts](int gid) -> partition_t const&
  {
    return parts[gid];
  };

  auto get_refinement_partition = f_get_refinement_partition();
  auto get_refis = f_get_mutable_refis();

  ginfos.reserve(graph.nodes.size());
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    partition_t const& part = parts[gid];
    int nblocks = part.num_parts();

    bool has_refinement = graph.nodes[gid].outs.size() > 0;

    // Note: all locations are initially filled to negative 1

    ginfos.push_back(ginfo_t {
      .partition = part,
      .joins =
        twolayer_construct_joins(graph, gid, part),
      .locations = vector<int>(nblocks, -1),
      .refinement_partition =
        has_refinement                                                        ?
        optional<partition_t>(
          twolayer_construct_refinement_partition(graph, gid, get_partition)) :
        std::nullopt                                                          ,
      .refis = std::nullopt
    });

    ginfo_t& ginfo = ginfos.back();

    twolayer_insert_join_deps(
      graph, gid, ginfo.joins, ginfo.partition, get_refinement_partition);

    if(ginfo.has_refinement()) {
      ginfo.refis = twolayer_construct_refis_and_connect_joins(
        graph, gid, ginfo.joins, ginfo.partition, ginfo.refinement_partition.value());
    }

    twolayer_insert_refi_outs_from_join_deps(
      graph, gid, ginfo.joins, get_refis);
  }
}

vector<placement_t> relationwise_t::get_placements() const
{
  vector<placement_t> ret;
  ret.reserve(ginfos.size());
  for(auto const& ginfo: ginfos) {
    ret.emplace_back(
      ginfo.partition,
      vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
  }
  return ret;
}

placement_t relationwise_t::get_placement_at(int gid) const
{
  ginfo_t const& ginfo = ginfos[gid];
  return placement_t(
    ginfo.partition,
    vtensor_t<int>(ginfo.partition.block_shape(), ginfo.locations));
}

std::function<partition_t const&(int)>
relationwise_t::f_get_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).partition;
  };
}

std::function<partition_t const&(int)>
relationwise_t::f_get_refinement_partition() const {
  return [this](int gid) -> partition_t const& {
    return ginfos.at(gid).refinement_partition.value();
  };
}

std::function<vector<refinement_t>&(int)>
relationwise_t::f_get_mutable_refis() {
  return [this](int gid) -> vector<refinement_t>& {
    return ginfos.at(gid).refis.value();
  };
}

int relationwise_t::num_agg_blocks_at(int gid) const {
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];
  int out_rank = node.op.out_rank();
  partition_t agg_partition(vector<partdim_t>(
    ginfo.partition.partdims.begin() + out_rank,
    ginfo.partition.partdims.end()));
  return agg_partition.num_parts();
}

set<int> relationwise_t::get_refi_usage_locs(rid_t const& rid) const {
  auto const& [gid, bid] = rid;
  auto const& ginfo = ginfos[gid];
  refinement_t const& refi = ginfo.refis.value()[bid];

  set<int> ret;
  for(auto const& jid: refi.outs) {
    auto const& [join_gid, join_bid] = jid;
    auto const& join_ginfo = ginfos[join_gid];
    int const& loc = join_ginfo.locations[join_bid];
    if(loc != -1) {
      ret.insert(loc);
    }
  }
  return ret;
}

uint64_t relationwise_t::get_refi_bytes(rid_t const& rid) const {
  auto const& [gid, bid] = rid;
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];

  auto dtype = node.op.out_dtype();
  uint64_t dsz = dtype_size(dtype);
  if(dtype_is_complex(dtype)) {
    dsz *= 2;
  }

  uint64_t nelem = ginfo.refinement_partition.value().block_size_at_bid(bid);
  return nelem * dsz;
}

uint64_t relationwise_t::get_join_out_bytes(jid_t const& jid) const {
  auto const& [gid, bid] = jid;
  auto const& ginfo = ginfos[gid];
  auto const& node = graph.nodes[gid];

  auto dtype = node.op.out_dtype();
  uint64_t dsz = dtype_size(dtype);
  if(dtype_is_complex(dtype)) {
    dsz *= 2;
  }

  auto const& join_part = ginfo.partition;
  auto shape = join_part.tensor_shape_at(join_part.from_bid(bid));
  shape.resize(node.op.out_rank());
  uint64_t nelem = product(shape);

  return nelem * dsz;
}

void relationwise_t::print_info() const
{
  for(int gid = 0; gid != graph.nodes.size(); ++gid) {
    std::cout << "gid: " << gid << std::endl;
    auto const& ginfo = ginfos[gid];
    int nbid = ginfo.locations.size();
    for(int bid = 0; bid != nbid; ++bid) {
      auto const& join = ginfo.joins[bid];
      vector<rid_t> rids(join.deps.begin(), join.deps.end());
      std::cout << "   J " << bid << ": " << rids << std::endl;
    }
    if(ginfo.refis) {
      auto const& refis = ginfo.refis.value();
      for(int bid = 0; bid != refis.size(); ++bid) {
        auto const& refi = refis[bid];
        for(auto const& unit: refi.units) {
          std::cout << "  R " << bid << ": " << unit.deps << std::endl;
        }
      }
    }
  }
}


