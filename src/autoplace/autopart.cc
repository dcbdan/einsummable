#include "autopart.h"
#include "../einsummable/taskgraph.h"

optional<partition_t> make_finer_via_doubling(partition_t const& p) {
  auto const& pds = p.partdims;

  uint64_t mx_sz = 0;
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    auto sizes = pd.sizes();
    uint64_t sz = *std::min_element(sizes.begin(), sizes.end());
    if(sz >= 2 && sz > mx_sz) {
      mx_sz = sz;
      split_d = d;
    }
  }

  if(split_d == -1) {
    return std::nullopt;
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::split_each(pds[split_d], 2);

  return partition_t(new_pds);
}

optional<partition_t> make_coarser_via_halving(partition_t const& p) {
  auto const& pds = p.partdims;

  uint64_t mn_sz = std::numeric_limits<uint64_t>::max();
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    auto sizes = pd.sizes();
    uint64_t sz = *std::min_element(sizes.begin(), sizes.end());
    if(sizes.size() >= 2 && sz < mn_sz) {
      mn_sz = sz;
      split_d = d;
    }
  }

  if(split_d == -1) {
    return std::nullopt;
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::merge_each(pds[split_d], 2);

  return partition_t(new_pds);
}

optional<partition_t> make_finer_via_increment(partition_t const& p) {
  auto const& pds = p.partdims;

  int min_parts;
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    uint64_t total = pd.total();
    int n_parts = pd.spans.size();
    if(n_parts != total) {
      if(split_d == -1 || n_parts < min_parts) {
        min_parts = n_parts;
        split_d = d;
      }
    }
  }

  if(split_d == -1) {
    return std::nullopt;
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::split(pds[split_d].total(), min_parts + 1);

  return partition_t(new_pds);
}

optional<partition_t> make_coarser_via_decrement(partition_t const& p) {
  auto const& pds = p.partdims;

  int max_parts;
  int split_d = -1;

  for(int d = 0; d != pds.size(); ++d) {
    auto const& pd = pds[d];
    uint64_t total = pd.total();
    int n_parts = pd.spans.size();
    if(n_parts != 1) {
      if(split_d == -1 || n_parts > max_parts) {
        max_parts = n_parts;
        split_d = d;
      }
    }
  }

  if(split_d == -1) {
    return std::nullopt;
  }

  vector<partdim_t> new_pds = pds;
  new_pds[split_d] = partdim_t::split(pds[split_d].total(), max_parts - 1);

  return partition_t(new_pds);
}

uint64_t get_smallest_block_size(partition_t const& p) {
  uint64_t ret = 1;
  for(auto const& pd: p.partdims) {
    auto szs = pd.sizes();
    ret *= *std::min_element(szs.begin(), szs.end());
  }
  return ret;
}

partition_t make_correctly_sized_partition(
  partition_t const& init,
  uint64_t min_sizing,
  optional<int> maybe_max_blocking,
  bool via_doubling)
{
  partition_t p = init;

  auto _make_coarser = via_doubling ?
    make_coarser_via_halving        :
    make_coarser_via_decrement      ;

  auto _make_finer = via_doubling   ?
    make_finer_via_doubling         :
    make_finer_via_increment        ;

  if(get_smallest_block_size(p) < min_sizing) {
    do {
      auto _p = _make_coarser(p);
      if(_p) {
        p = _p.value();
      } else {
        break;
      }
    } while(get_smallest_block_size(p) < min_sizing);

    return p;
  }

  if(maybe_max_blocking) {
    auto const& max_blocking = maybe_max_blocking.value();

    if(p.num_parts() > max_blocking) {
      do {
        auto _p = _make_coarser(p);
        if(_p) {
          p = _p.value();
        } else {
          throw std::runtime_error("should always be able to make coarser");
        }
      } while(p.num_parts() > max_blocking);
    } else {
      while(true) {
        auto _p = _make_finer(p);
        if(_p && _p.value().num_parts() <= max_blocking &&
           get_smallest_block_size(_p.value()) >= min_sizing)
        {
          p = _p.value();
        } else {
          break;
        }
      }
    }

    if(p.num_parts() > max_blocking) {
      throw std::runtime_error("?");
    }
  }

  return p;
}

vector<partition_t> autopartition(
  graph_t const& graph,
  uint64_t min_sizing,
  int max_blocking,
  equal_items_t<int> equal_constraints,
  bool via_doubling)
{
  vector<partition_t> ret;

  for(auto const& node: graph.nodes) {
    if(node.op.is_input()) {
      partition_t init = partition_t::singleton(node.op.out_shape());
      ret.push_back(make_correctly_sized_partition(
        init, min_sizing, max_blocking, via_doubling));
    } else if(node.op.is_formation()) {
      auto const& inn = node.inns[0];
      int out_rank = node.op.out_shape().size();
      if(graph.nodes[inn].op.has_aggregation()) {
        auto const& pds = ret[inn].partdims;
        partition_t init(vector<partdim_t>(pds.begin(), pds.begin() + out_rank));
        ret.push_back(
          make_correctly_sized_partition(init, min_sizing, max_blocking, via_doubling));
      } else {
        ret.push_back(ret[inn]);
      }
    } else if(node.op.is_complexer()) {
      auto const& complexer = node.op.get_complexer();
      auto const& inn = node.inns[0];
      if(complexer.is_to_real()) {
        partition_t const& init = ret[inn];
        ret.push_back(double_last_dim(init));
      } else {
        bool can_halve = true;
        auto init = ret[inn];
        auto sizes = init.partdims.back().sizes();
        for(auto const& sz: sizes) {
          if(sz % 2 != 0) {
            can_halve = false;
            break;
          }
        }
        if(can_halve) {
          halve_last_dim_inplace(init);
          ret.push_back(init);
        } else {
          uint64_t total = init.partdims.back().total();
          if(total % 2 != 0) {
            throw std::runtime_error("should not happen: can't divide by 2");
          }
          total /= 2;
          init.partdims.back() = partdim_t::singleton(total);
          ret.push_back(
            make_correctly_sized_partition(init, min_sizing, max_blocking, via_doubling));
        }
      }
    } else if(node.op.is_concat()) {
      auto const& concat = node.op.get_concat();

      vector<uint64_t> dim_szs;
      int rank = concat.shape().size();
      vector<vector<partdim_t>> all_pds(rank);

      for(auto const& inn: node.inns) {
        auto const& inn_part = ret[inn];
        auto const& pds = inn_part.partdims;
        for(int i = 0; i != rank; ++i) {
          auto const& partdim = pds[i];
          if(i == concat.dim) {
            vector_concatenate_into(dim_szs, partdim.sizes());
          } else {
            all_pds[i].push_back(partdim);
          }
        }
      }

      vector<partdim_t> pds;
      for(int i = 0; i != rank; ++i) {
        if(i == concat.dim) {
          pds.push_back(partdim_t::from_sizes(dim_szs));
        } else {
          pds.push_back(partdim_t::unions(all_pds[i]));
        }
      }

      partition_t init(pds);

      ret.push_back(
        make_correctly_sized_partition(init, min_sizing, max_blocking, via_doubling));
    } else if(node.op.is_subset()) {
      int const& inn = node.inns[0];
      auto const& subset = node.op.get_subset();
      auto const& inn_part = ret[inn];
      auto new_part_full = inn_part.subset(subset.get_hrect());
      auto new_part = partition_t(subset.squeeze_vec(new_part_full.partdims));
      ret.push_back(
        make_correctly_sized_partition(new_part, min_sizing, max_blocking, via_doubling));
    } else if(node.op.is_einsummable()) {
      auto const& e = node.op.get_einsummable();
      vector<vector<partdim_t>> pds(e.join_shape.size());
      for(int which_inn = 0; which_inn != node.inns.size(); ++which_inn) {
        int const& inn = node.inns[which_inn];
        auto const& partdims = ret[inn].partdims;

        auto const& modes = e.inns[which_inn];

        for(int i = 0; i != modes.size(); ++i) {
          pds[modes[i]].push_back(partdims[i]);
        }
      }
      vector<partdim_t> pd;
      for(auto const& _pd: pds) {
        pd.push_back(partdim_t::unions(_pd));
      }
      partition_t init(pd);
      ret.push_back(
        make_correctly_sized_partition(init, min_sizing, max_blocking, via_doubling));
    } else {
      throw std::runtime_error("missing node type");
    }
  }

  // TODO: set equal constraints

  return ret;
}

optional<vector<partition_t>>
autopart_from_inputs(
  graph_t const& graph,
  map<int, partition_t> const& input_to_part)
{
  // ugh: partition has no default constructor, so passing in an empty
  //      partdims vector, which is an invalid partition, so the constructor
  //      could feasibly throw an error.
  vector<partition_t> ret(graph.nodes.size(), partition_t({}));

  auto get_partdim = [&](int gid, int dim) {
    return ret[gid].partdims[dim];
  };

  for(int const& gid: graph.get_order()) {
    auto const& node = graph.nodes[gid];
    if(node.op.is_input()) {
      ret[gid] = input_to_part.at(gid);
    } else if(node.op.is_formation()) {
      int out_rank = node.op.out_rank();
      auto const& inn_gid = node.inns[0];
      auto const& inn_pds = ret[inn_gid].partdims;
      ret[gid] = partition_t(vector<partdim_t>(
        inn_pds.begin(), inn_pds.begin() + out_rank));
    } else if(node.op.is_einsummable()) {
      auto const& einsummable = node.op.get_einsummable();

      vector<vector<partdim_t>> inn_pds;
      inn_pds.reserve(node.inns.size());
      for(int const& inn_gid: node.inns) {
        inn_pds.push_back(ret[inn_gid].partdims);
      }

      auto maybe_join_shape = einsummable_t::construct_join_shape_(
        einsummable.inns,
        inn_pds,
        partdim_t(),
        [](partdim_t const& lhs, partdim_t const& rhs) { return lhs == rhs; });

      if(maybe_join_shape) {
        ret[gid] = partition_t(maybe_join_shape.value());
      } else {
        // most likely the input shapes did not correctly match
        return std::nullopt;
      }
    } else {
      // not implemented!
      return std::nullopt;
    }
  }

  return ret;
}
