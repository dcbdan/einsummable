#include "../src/matrixgraph/ff.h"
#include "../src/einsummable/coster.h"

cluster_t make_cluster(int nlocs) {
  using device_t = cluster_t::device_t;
  using connection_t = cluster_t::connection_t;

  // nvidia tesla p100 9.3 Teraflops single precision
  uint64_t giga = 1e9;
  uint64_t tera = 1e12;
  uint64_t nvidia_tesla_p100 = (tera * 93) / 10;


  uint64_t compute_on_device = nvidia_tesla_p100;
  uint64_t bandwidth_between_device = 20 * giga;

  int capacity = 1; // all kernels have a utilization of 1 for now,
                    // so  give all devices a capacity of 1



  vector<device_t> devices;
  for(int loc = 0; loc != nlocs; ++loc) {
    devices.push_back(device_t {
      .compute = compute_on_device / capacity,
      .capacity = capacity
    });
  }

  vector<connection_t> connections;
  for(int i = 0; i != nlocs; ++i) {
  for(int j = 0; j != nlocs; ++j) {
    if(i != j) {
      connections.push_back(connection_t {
        .bandwidth = bandwidth_between_device,
        .src = i,
        .dst = j
      });
    }
  }}

  return cluster_t::make(devices, connections);
}

void update_graph_partitions(
  graph_t& graph,
  vector<partition_t> parts)
{
  int n_nodes = graph.nodes.size();
  for(int i = 0; i != n_nodes; ++i) {
    graph.nodes[i].placement = placement_t(parts[i]);
  }
}

void main01_autopart() {
  uint64_t dn = 1000;
  uint64_t dp = 100;
  uint64_t dd = 100;
  vector<uint64_t> dws = {105, 110, 115, 120, 125};
  float learning_rate = 0.001;

  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,learning_rate);
  auto const& mgraph = ff_info.mgraph;

  vector<int> outs = ff_info.wsout;
  outs.push_back(ff_info.sqdiff);
  auto [graph, m_to_g] = mgraph.compile(outs);

  graph.print();

  vector<char> _line(40, '/');
  std::string line(_line.begin(), _line.end());
  std::cout << line << std::endl;

  uint64_t mmlike_sizing = 75*75*75;
  uint64_t min_sizing = 50*50;

  {
    vector<partition_t> new_parts = autopartition(
      graph, mmlike_sizing, min_sizing);
    update_graph_partitions(graph, new_parts);
  }

  graph.print();
  std::cout << line << std::endl;

  {
    set<tuple<int,int>> same_parts;
    for(int i = 0; i != ff_info.wsinn.size(); ++i) {
      int const& winn = m_to_g.at(ff_info.wsinn[i]);
      int const& wout = m_to_g.at(ff_info.wsout[i]);
      DOUT(winn << ", " << wout);
      same_parts.insert({winn, wout});
    }

    vector<partition_t> new_parts = autopartition(
      graph,
      mmlike_sizing, min_sizing,
      same_parts, {}
    );

    update_graph_partitions(graph, new_parts);
  }

  graph.print();
}

struct autoplace_t {
  struct gid_t {
    int id;
    int index;
  };

  autoplace_t(
    graph_t& g,
    cluster_t const& c,
    int nloc,
    set<tuple<int,int>> const& equal_placements);

  void set_gid(gid_t const& gid, int loc);

  void undo();

  cluster_t const& cluster;
  graph_t& graph;
  int const nloc;

  // The node ids that have placements
  // which are being set (one per equivalence)
  set<int> valid_ids;
  vector<int> to_valid_id;
  // Given one gid, get all the gids that must have
  // the same location. Each item in an equivalence
  // set has a different node id than all other
  // idems in that equivalence set.
  equal_items_t<gid_t> equal_items;

  // (each gid.id ins in valid_ids)
  map<gid_t, int> _undo_locs;

private:
  inline int&       _get_loc(gid_t const& gid);
  inline int const& _get_loc(gid_t const& gid) const;
};

bool operator<(autoplace_t::gid_t const& lhs, autoplace_t::gid_t const& rhs) {
  return two_tuple_lt(lhs, rhs);
}
bool operator==(autoplace_t::gid_t const& lhs, autoplace_t::gid_t const& rhs) {
  return two_tuple_eq(lhs, rhs);
}

int main() {
  uint64_t dn = 1000;
  uint64_t dp = 100;
  uint64_t dd = 100;
  vector<uint64_t> dws = {105, 110, 115, 120, 125};
  float learning_rate = 0.001;

  ff_sqdiff_t ff_info = ff_sqdiff_update(dn,dp,dd,dws,learning_rate);
  auto const& mgraph = ff_info.mgraph;

  vector<int> outs = ff_info.wsout;
  outs.push_back(ff_info.sqdiff);
  auto [graph, m_to_g] = mgraph.compile(outs);

  //graph.print();

  int nloc = 4;
  cluster_t cluster = make_cluster(nloc);

  autoplace_t autoplace(graph, cluster, nloc, {});

  for(auto const& id: autoplace.valid_ids) {
    std::cout << id << std::endl;
  }

}

void autoplace_t::undo() {
  for(auto const& [gid, loc]: _undo_locs) {
    if(equal_items.has(gid)) {
      for(auto const& equiv_gid: equal_items.get_at(gid)) {
        _get_loc(equiv_gid) = loc;
      }
    } else {
      _get_loc(gid) = loc;
    }
  }

  _undo_locs = {};
}

void autoplace_t::set_gid(gid_t const& gid, int loc) {
  if(valid_ids.count(gid.id) == 0) {
    throw std::runtime_error("can't set gid for this node");
  }

  if(_undo_locs.count(gid) == 0) {
    _undo_locs.insert({gid, loc});
  }

  if(equal_items.has(gid)) {
    for(auto const& equiv_gid: equal_items.get_at(gid)) {
      _get_loc(equiv_gid) = loc;
    }
  } else {
    _get_loc(gid) = loc;
  }
}

int& autoplace_t::_get_loc(gid_t const& gid) {
  auto const& [id,index] = gid;
  return graph.nodes[id].placement.locations.get()[index];
}
int const& autoplace_t::_get_loc(gid_t const& gid) const {
  auto const& [id,index] = gid;
  return graph.nodes[id].placement.locations.get()[index];
}

autoplace_t::autoplace_t(
  graph_t& g,
  cluster_t const& c,
  int n,
  set<tuple<int,int>> const& eqs_)
    : graph(g), cluster(c), nloc(n)
{
  // Note: to say that two nodes have equivalent placement is an
  //       elementwise statement.

  auto eqs = eqs_;

  equal_items_t<int> eq_nodes;

  // Any time a formation node has an input with the
  //    same partition, add to constraints
  // Any time an einsummable has one input and there is no agg,
  //   add to constraints if the partitions are equivalent
  //   through permutation.
  int n_nodes = graph.nodes.size();
  for(int id = 0; id != n_nodes; ++id) {
    auto const& node = graph.nodes[id];
    if(node.op.is_formation()) {
      int const& id_inn = node.inns[0];
      auto const& node_inn = graph.nodes[id_inn];
      if(node.placement.partition == node_inn.placement.partition) {
        // Note: this won't be reached if the input is getting aggregated

        // just do this when processing eqs
        eqs.insert({id, node.inns[0]});
      }
    } else if(
      node.op.is_einsummable() &&
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
        // tell eq_nodes these nodes belong to the same equivalency set
        eq_nodes.insert(id, id_inn);

        // now we have to reach past the permutation
        auto inn_shape  = part_inn.block_shape();
        auto join_shape = part.block_shape();
        vector<int> join_index(join_shape.size(), 0);
        do {
          vector<int> inn_index = e.get_input_from_join(join_index, 0);
          equal_items.insert(
            {id,     idxs_to_index(join_shape, join_index)},
            {id_inn, idxs_to_index(inn_shape, inn_index)}
          );
        } while(increment_idxs(join_shape, join_index));
      }
    }
  }

  for(auto const& [id_i,id_j]: eqs) {
    if(id_i == id_j) {
      continue;
    }

    auto const& part_i = g.nodes[id_i].placement.partition;
    auto const& part_j = g.nodes[id_j].placement.partition;
    if(part_i != part_j) {
      throw std::runtime_error("eqs partitions are not equal!");
    }

    eq_nodes.insert(id_i, id_j);

    // add to equal_items these corresponding items
    int num_parts = part_i.num_parts();
    for(int idx = 0; idx != num_parts; ++idx) {
      equal_items.insert(
        {id_i, idx},
        {id_j, idx}
      );
    }
  }

  // TODO set to_valid_id (or not)

  // Set valid ids, one per equivalency
  auto eq_candidates = eq_nodes.candidates();
  valid_ids.insert(eq_candidates.begin(), eq_candidates.end());
  // add all the unique'd partiitoned nodes
  for(int i = 0; i != n_nodes; ++i) {
    if(!eq_nodes.has(i)) {
      valid_ids.insert(i);
    }
  }

  // TODO initliaze the locations in a "good" way
  //      (if matmul, should be evenly distributed and where inputs are)
  //      (if formation, should be at where inputs be)
  //      (if input, should be evenly distributed)
  //      Maybe use the twophase graph to do this?
}
