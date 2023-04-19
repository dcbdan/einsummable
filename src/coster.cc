#include "coster.h"
#include "copyregion.h"

cluster_t cluster_t::make(
  vector<cluster_t::device_t> const& devices,
  vector<cluster_t::connection_t> const& cs)
{
  // Remove duplicate connections in cs
  // by summing the corresponding bandwidth

  vector<connection_t> connections;
  map<tuple<int, int>, int> to_connection;

  for(auto const& c: cs) {
    auto key = c.key();

    if(to_connection.count(key) > 0) {
      connections[to_connection[key]].bandwidth += c.bandwidth;
    } else {
      connections.push_back(c);
      to_connection.insert({key, connections.size()-1});
    }
  }

  return cluster_t {
    .devices = devices,
    .connections = connections,
    .to_connection = to_connection
  };
}

float cluster_t::move(int src, int dst, uint64_t bytes) const {
  connection_t const& c = connections[to_connection.at({src,dst})];
  return (1.0 / c.bandwidth) * bytes;
}

float cluster_t::compute(int loc, uint64_t flops) const {
  device_t const& d = devices[loc];
  return (1.0 / d.compute) * flops;
}

int twolayergraph_t::insert_join(uint64_t flops, int util, gid_t gid, vector<rid_t> deps)
{
  int ret = joins.size();

  joins.push_back(join_t {
    .flops = flops,
    .util = util,
    .gid = gid,
    .deps = deps,
    .outs = {}
  });

  order.push_back(twolayerid_t { .id = ret, .is_join = true });

  for(auto const& rid: deps) {
    refinements[rid].outs.insert(ret);
  }

  return ret;
}

int twolayergraph_t::insert_empty_refinement()
{
  int ret = refinements.size();

  refinements.push_back(refinement_t {
    .units = {},
    .outs = {}
  });

  order.push_back(twolayerid_t { .id = ret, .is_join = false });

  return ret;
}

void twolayergraph_t::add_agg_unit(int rid, uint64_t bytes, vector<jid_t> deps)
{
  auto& refi = refinements[rid];

  refi.units.push_back(agg_unit_t {
    .bytes = bytes,
    .deps = deps
  });

  for(auto const& jid: deps) {
    joins[jid].outs.insert(rid);
  }
}

// Note: Almost a copy of union_partition_holders in src/taskgraph.cc
partition_t union_partitions(vector<partition_t> const& ps)
{
  vector<partdim_t> partdims;
  int rank = ps[0].block_shape().size();
  partdims.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    vector<partdim_t> xs;
    xs.reserve(ps.size());
    for(auto const& p: ps) {
      xs.push_back(p.partdims[i]);
    }
    partdims.push_back(partdim_t::unions(xs));
  }
  return partition_t(partdims);
}

// This function is a riff on state_t::communicate
// in taskgraph.cc used in taskgraph_t::make.
twolayergraph_t twolayergraph_t::make(graph_t const& graph) {
  twolayergraph_t ret(graph);

  vector<tensor_t<rid_t>> all_refis(graph.nodes.size());
  vector<partition_t> all_refinement_partitions;

  // set up all the refinement partitions, which for a given node is
  // the refinement of the usage partitions
  all_refinement_partitions.reserve(graph.nodes.size());
  for(int join_id = 0; join_id != graph.nodes.size(); ++join_id) {
    auto const& join_node = graph.nodes[join_id];
    vector<partition_t> usage_partitions;
    usage_partitions.reserve(2*join_node.outs.size());
    for(auto const& out_id: join_node.outs) {
      auto const& out_node = graph.nodes[out_id];
      if(out_node.op.is_formation()) {
        usage_partitions.push_back(out_node.placement.partition);
      } else {
        // Note that an einsummable node can use an input multiple times
        // and therefore there may be multiple usage partitions to collect
        auto const& einsummable = join_node.op.get_einsummable();
        for(int which_input = 0; which_input != out_node.inns.size(); ++which_input) {
          if(out_node.inns[which_input] == join_id) {
            usage_partitions.emplace_back(einsummable.get_input_from_join(
              out_node.placement.partition.partdims,
              which_input));
          }
        }
      }
    }

    all_refinement_partitions.push_back(union_partitions(usage_partitions));
  }

  for(auto const& graph_id: graph.get_order()) {
    auto const& node = graph.nodes[graph_id];

    partition_t const& join_partition = node.placement.partition;
    auto const& join_locations = node.placement.locations;

    auto join_block_shape = join_partition.block_shape();

    // initialize join_ids by inserting
    // join ops into the graph for each block
    tensor_t<jid_t> join_ids = tensor_t<jid_t>(join_block_shape);
    {
      vector<int> join_index(join_block_shape.size(), 0);

      // flops
      //   input nodes: 0
      //   formation nodes: 0
      //   join nodes: the tensor block size
      std::function<uint64_t()> get_flops;
      if(node.op.is_einsummable()) {
        get_flops = [&] {
          return product(join_partition.tensor_shape_at(join_index));
        };
      } else {
        get_flops = []{ return 0; };
      }

      // TODO: Implement get_util somehow..
      //
      // The actual worker utilization depends on what the device is,
      // which this function should remain agnostic to.
      //
      // A gpu may have 1000 workers whereas a cpu may have 10.
      // A matmul will have a full utilization whereas an elementwise
      // op may have a utilization of 1 on a cpu if it is single threaded.
      //
      // Perhaps have every op either get a full utilization or
      // singleton utilization.
      //
      // In any case, a utilization of 1 and a worker capacity of 1
      // is equivalent to not having worker capacities anyway, which
      // is what should be implemented in the meantime.
      std::function<int()> get_util = []{ return 1; };

      // deps
      //   input nodes: {}
      //   formation nodes: same as a straight einsummable op
      //   einsummable nodes: reach into each input and grab figure it out
      std::function<vector<int>()> get_deps;
      if(node.op.is_input()) {
        get_deps = []{ return vector<int>(); };
      } else {
        // We have
        //   (1) an einsummable op over the graph node
        //   (2) input refinement partitions
        //   (3) an output hrect
        einsummable_t einsummable;
        if(node.op.is_formation()) {
          auto op_shape = node.op.shape();
          int rank = op_shape.size();

          vector<vector<int>> inns(1);
          inns[0] = vector<int>(rank);
          std::iota(inns[0].begin(), inns[0].end(), 0);

          einsummable = einsummable_t {
            .join_shape = op_shape,
            .inns = inns,
            .out_rank = rank,
            .join = scalar_join_t::mul,  // will not be used
            .castable = castable_t::add, // will not be used
          };
        } else {
          einsummable = node.op.get_einsummable();
        }

        get_deps = [&]{
          vector<int> ret;
          ret.reserve(4*node.inns.size()); // just a guess on the number of inputs

          auto hrect = join_partition.get_hrect(join_index);

          // for each input, get the inputs refinement ids that map
          // onto the corresponding input hrect.
          for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
            int const& inn = node.inns[which_inn];

            partition_t const& inn_partition = all_refinement_partitions[inn];

            tensor_t<rid_t> const& inn_refis = all_refis[inn];

            auto inn_hrect = einsummable.get_input_from_join(hrect, which_inn);

            auto inn_region = inn_partition.get_region(inn_hrect);
            vector<int> inn_index = vector_mapfst(inn_region);
            do {
              ret.push_back(inn_refis.at(inn_index));
            } while(increment_idxs_region(inn_region, inn_index));
          }

          return ret;
        };
      }

      do {
        gid_t gid {
          .id = graph_id,
          .index = idxs_to_index(join_block_shape, join_index)
        };
        join_ids.at(join_index) = ret.insert_join(
          get_flops(),
          get_util(),
          gid,
          get_deps());
      } while(increment_idxs(join_block_shape, join_index));
    }

    partition_t const& refi_partition = all_refinement_partitions[graph_id];

    int join_rank = node.op.rank();
    int out_rank  = node.op.out_rank();
    int agg_rank  = join_rank - out_rank;

    auto const& _join_partdims = join_partition.partdims;

    partition_t out_partition(vector<partdim_t>(
      _join_partdims.begin(),
      _join_partdims.begin() + out_rank));

    std::optional<vector<int>> maybe_agg_shape;
    if(agg_rank > 0) {
      partition_t agg_partition(vector<partdim_t>(
        _join_partdims.begin() + out_rank,
        _join_partdims.end()));

      maybe_agg_shape = agg_partition.block_shape();
    }

    // set up the refinement nodes

    auto refi_shape = refi_partition.block_shape();
    all_refis[graph_id] = tensor_t<rid_t>(refi_shape);
    tensor_t<rid_t>& refi_ids = all_refis[graph_id];

    // initialize empty refinements
    {
      vector<int> refi_index(refi_shape.size(), 0);
      do {
        refi_ids.at(refi_index) = ret.insert_empty_refinement();
      } while(increment_idxs(refi_shape, refi_index));
    }

    vector<int> out_shape = out_partition.block_shape();
    vector<int> out_index(out_shape.size(), 0);
    do {
      copyregion_t get_regions(refi_partition, out_partition, out_index);
      do {
        vector<int> refi_index = vector_from_each_member(
          get_regions.info, int, idx);
        vector<uint64_t> read_shape = vector_from_each_member(
          get_regions.info, uint64_t, size);

        vector<jid_t> deps;
        if(maybe_agg_shape) {
          vector<int> agg_index(agg_rank, 0);
          auto const& agg_shape = maybe_agg_shape.value();
          deps.reserve(product(agg_shape));
          do {
            vector<int> join_index = vector_concatenate(out_index, agg_index);
            deps.push_back(join_ids.at(join_index));
          } while(increment_idxs(agg_shape, agg_index));
        } else {
          // the join index is the out index if there is no agg
          // and there is only one input
          auto const& join_index = out_index;
          deps.push_back(join_ids.at(join_index));
        }

        ret.add_agg_unit(
          refi_ids.at(refi_index),
          product(read_shape),
          deps);
      } while(get_regions.increment());
    } while(increment_idxs(out_shape, out_index));
  }

  return ret;
}

costgraph_t costgraph_t::make(twolayergraph_t const& twolayer) {
  costgraph_t ret;

  map<tuple<int, int>, int> ridloc_to_costid;
  map<int, tuple<int, int>> jid_to_costidloc;

  for(auto const& [id, is_join]: twolayer.order) {
    if(is_join) {
      auto const& jid = id;
      auto const& join = twolayer.joins[jid];

      auto const& loc = twolayer.join_location(join);

      vector<int> deps;
      deps.reserve(join.deps.size());
      for(auto const& rid: join.deps) {
        deps.push_back(ridloc_to_costid.at({rid, loc}));
      }

      jid_to_costidloc[jid] = {
        ret.insert_compute(loc, join.flops, join.util, deps),
        loc
      };
    } else {
      auto const& rid = id;
      auto refi = twolayer.refinements[rid];

      // deduce where this refi will be used
      std::set<int> usage_locs;
      for(auto const& out_jid: refi.outs) {
        usage_locs.insert(twolayer.join_location(out_jid));
      }

      // insert moves as necc and fill out ridloc_to_costid
      for(auto const& dst: usage_locs) {
        vector<int> deps;
        for(auto const& [bytes, jids]: refi.units) {
          for(auto const& jid: jids) {
            auto const& [inn_costid, src] = jid_to_costidloc.at(jid);
            if(src == dst) {
              deps.push_back(inn_costid);
            } else {
              deps.push_back(ret.insert_move(src, dst, bytes, inn_costid));
            }
          }
        }
        int costid = ret.insert_barrier(deps);
        ridloc_to_costid.insert({ {rid, dst}, costid });
      }
    }
  }

  return ret;
}

int costgraph_t::insert_compute(
  int loc, uint64_t flops, int util,
  vector<int> const& deps)
{
  return insert(
    compute_t {
      .loc = loc,
      .flops = flops,
      .util = util
    },
    deps);
}

int costgraph_t::insert_move(int src, int dst, uint64_t bytes, int id)
{
  return insert(
    move_t {
      .src = src,
      .dst = dst,
      .bytes = bytes
    },
    {id}
  );
}

int costgraph_t::insert_barrier(vector<int> const& deps)
{
  return insert(barrier_t(), deps);
}

int costgraph_t::insert(
  op_t op,
  vector<int> const& deps)
{
  int ret = nodes.size();

  nodes.push_back(node_t {
    .inns = std::set(deps.begin(), deps.end()),
    .outs = {},
    .op = op
  });

  for(auto const& inn: nodes.back().inns) {
    nodes[inn].outs.insert(ret);
  }

  return ret;
}

template <typename T>
struct worker_map_t {
  worker_map_t(cluster_t const& c):
    cluster(c),
    info(c.devices.size() + c.connections.size())
  {}

  T& operator()(int loc) {
    return info[loc];
  }
  T& operator()(int src, int dst) {
    return info[cluster.devices.size() + cluster.to_connection.at({src,dst})];
  }
  T& operator()(costgraph_t::op_t const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)) {
      return this->operator()(std::get<costgraph_t::compute_t>(x));
    } else if(std::holds_alternative<costgraph_t::move_t>(x)) {
      return this->operator()(std::get<costgraph_t::move_t>(x));
    } else {
      throw std::runtime_error("invalid alternative in worker_map");
    }
  }
  T& operator()(costgraph_t::compute_t const& c) {
    return this->operator()(c.loc);
  }
  T& operator()(costgraph_t::move_t const& m) {
    return this->operator()(m.src, m.dst);
  }

  vector<T> const& get() const { return info; }
  vector<T>&       get()       { return info; }

private:
  cluster_t const& cluster;

  vector<T> info;
};

struct cost_state_t {
  cost_state_t(cluster_t const& c, costgraph_t const& cg)
    : cluster(c), costgraph(cg), time(0.0), pending(c), capacity(c)
  {
    // initialize capacity
    for(int loc = 0; loc != cluster.devices.size(); ++loc) {
      capacity(loc) = cluster.devices[loc].capacity;
    }
    for(auto const& [_, src, dst]: cluster.connections) {
      capacity(src, dst) = 1;
    }

    // initialize pending and num_remaining
    num_remaining.reserve(costgraph.nodes.size());
    for(int id = 0; id != costgraph.nodes.size(); ++id) {
      auto const& node = costgraph.nodes[id];
      auto const& [inns, outs, op] = node;
      num_remaining.push_back(inns.size());

      if(inns.size() == 0) {
        if(node.is_barrier()) {
          throw std::runtime_error("a barrier must have dependencies");
        } else {
          pending(op).insert(id);
        }
      }
    }
  }

  void decrement_nodes(set<int> outs) {
    // this function is recursive; if outs is empty,
    // this is the base case
    if(outs.size() == 0) {
      return;
    }

    set<int> barrier_outs;
    for(auto const& out_id: outs) {
      num_remaining[out_id] -= 1;
      if(num_remaining[out_id] == 0) {
        auto const& out_node = costgraph.nodes[out_id];
        if(out_node.is_barrier()) {
          barrier_outs.insert(out_node.outs.begin(), out_node.outs.end());
        } else {
          pending(out_node.op).insert(out_id);
        }
      }
    }

    // now tail recurse on the collected barrier outs
    decrement_nodes(std::move(barrier_outs));
  }

  // returns true if something was taken from in_progress
  bool step() {
    if(in_progress.size() == 0) {
      return false;
    }

    // take off of in_progress
    auto [_time, op_id] = in_progress.top();
    in_progress.pop();

    // increment time
    time = _time;

    // alias the node
    auto const& node = costgraph.nodes[op_id];

    // these resources are now freed
    capacity(node.op) += node.worker_utilization();

    // tell the out nodes there is one less dependency
    // and possibly add to pending
    decrement_nodes(node.outs);

    return true;
  }

  void fill_with_work(int worker_id) {
    auto& capacity_vec = capacity.get();
    auto& pending_vec = pending.get();

    int& capacity_here          = capacity_vec[worker_id];
    std::set<int>& pending_here = pending_vec[worker_id];

    vector<int> possible;
    possible.reserve(pending_here.size());
    for(int const& op_id: pending_here) {
      auto const& node = costgraph.nodes[op_id];
      if(node.worker_utilization() <= capacity_here) {
        possible.push_back(op_id);
      }
    }

    while(possible.size() > 0) {
      int op_id = vector_random_pop(possible);
      auto const& node = costgraph.nodes[op_id];

      // move from pending
      pending_here.erase(op_id);
      // utilize the workers
      capacity_here -= node.worker_utilization();
      // and start working
      float time_when_finished = time + compute_cost(node.op);
      in_progress.push({time_when_finished, op_id});

      auto it = std::copy_if(
        possible.begin(),
        possible.end(),
        possible.begin(),
        [&](int const& id) {
          return costgraph.nodes[id].worker_utilization() < capacity_here;
        });
      possible.resize(it - possible.begin());
    }
  }

  void fill_with_work() {
    int num_workers = capacity.get().size();
    for(int i = 0; i != num_workers; ++i) {
      fill_with_work(i);
    }
  }

  float compute_cost(costgraph_t::op_t const& x)
  {
    if(std::holds_alternative<costgraph_t::compute_t>(x)){
      auto const& compute = std::get<costgraph_t::compute_t>(x);
      return cluster.compute(compute.loc, compute.flops);
    } else if(std::holds_alternative<costgraph_t::move_t>(x)) {
      auto const& move = std::get<costgraph_t::move_t>(x);
      return cluster.move(move.src, move.dst, move.bytes);
    } else {
      throw std::runtime_error("invalid alternative in compute_cost");
    }
  }

  // returns true if pending and in progress is empty
  bool is_done() {
    if(in_progress.size() > 0) {
      return false;
    }
    for(auto const& pending_at: pending.get()) {
      if(pending_at.size() != 0) {
        return false;
      }
    }
    return true;
  }

  cluster_t const& cluster;
  costgraph_t const& costgraph;

  float time;

  // for each worker, store the ids of all ops that can be
  // started
  worker_map_t<set<int>> pending;

  // store when the id will be ready, and the id
  priority_queue_least<tuple<float, int>> in_progress;

  worker_map_t<int> capacity;

  vector<int> num_remaining; // for each id, the num remaining
};

float costgraph_t::operator()(cluster_t const& cluster) const {
  cost_state_t state(cluster, *this);

  do {
    state.fill_with_work();
  } while(state.step());

  if(!state.is_done()) {
    throw std::runtime_error("costgraph comput time: state should be done");
  }

  return state.time;
}

