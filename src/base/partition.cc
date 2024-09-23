#include "partition.h"

partition_t::partition_t(vector<partdim_t> const& p):
  partdims(p)
{}

partition_t partition_t::singleton(vector<uint64_t> shape) {
  vector<partdim_t> partdims;
  partdims.reserve(shape.size());
  for(auto const& sz: shape) {
    partdims.push_back(partdim_t::singleton(sz));
  }
  return partition_t(partdims);
};

partition_t partition_t::from_wire(string const& str) {
  es_proto::Partition p;
  if(!p.ParseFromString(str)) {
    throw std::runtime_error("could not parse partition");
  }
  return from_proto(p);
}

partition_t partition_t::from_proto(es_proto::Partition const& p) {
  vector<partdim_t> partdims;
  partdims.reserve(p.partdims_size());
  for(int i = 0; i != p.partdims_size(); ++i) {
    partdims.push_back(partdim_t::from_proto(p.partdims(i)));
  }
  return partition_t(partdims);
}

string partition_t::to_wire() const {
  es_proto::Partition p;
  to_proto(p);
  string ret;
  p.SerializeToString(&ret);
  return ret;
}

void partition_t::to_proto(es_proto::Partition& p) const {
  for(auto const& partdim: partdims) {
    es_proto::Partdim* pd = p.add_partdims();
    partdim.to_proto(*pd);
  }
}

bool partitions_region(
  vector<vector<tuple<uint64_t, uint64_t>>> const& hrects,
  vector<uint64_t> const& shape)
{
  int rank = shape.size();

  // Cut the shape hrect into a refined set of blocks
  // based on the partial units
  partition_t refinement = [&] {
    vector<partdim_t> partdims;
    partdims.reserve(rank);
    {
      vector<vector<uint64_t>> spans(rank);
      for(int i = 0; i != rank; ++i) {
        spans[i].push_back(shape[i]);
      }
      for(auto const& hrect: hrects) {
        for(int i = 0; i != rank; ++i) {
          auto const& [beg,end] = hrect[i];
          auto& ss = spans[i];
          if(beg != 0) {
            ss.push_back(beg);
          }
          ss.push_back(end);
        }
      }
      for(vector<uint64_t>& ss: spans) {
        std::sort(ss.begin(), ss.end());
        vector_remove_duplicates(ss);
        partdims.push_back(partdim_t::from_spans(ss));
      }
    }
    return partition_t(partdims);
  }();

  // for each touch, increment the relevant write regions
  auto refinement_shape = refinement.block_shape();
  vtensor_t<int> counts(refinement_shape);

  for(auto const& hrect: hrects) {
    vector<tuple<int,int>> region = refinement.get_exact_region(hrect);
    vector<int> index = vector_mapfst(region);
    do {
      counts.at(index) += 1;
    } while(increment_idxs_region(region, index));
  }

  // Check that the entire write shape is partitioned.
  //
  // If the out regions are not disjoint, then
  //   some num_touch will be bigger than one.
  // If some of the shape is not written to,
  //   then some nume_touch will be zero.

  for(auto const& num_touch: counts.get()) {
    if(num_touch != 1) {
      return false;
    }
  }
  return true;
}

bool operator==(partition_t const& lhs, partition_t const& rhs) {
  return vector_equal(lhs.partdims, rhs.partdims);
}
bool operator!=(partition_t const& lhs, partition_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, partition_t const& p) {
  out << "partition" << p.partdims;
  return out;
}

vector<partition_t> from_proto_partition_list(es_proto::PartitionList const& pl) {
  vector<partition_t> ret;
  ret.reserve(pl.parts_size());
  for(int i = 0; i != pl.parts_size(); ++i) {
    ret.push_back(partition_t::from_proto(pl.parts(i)));
  }
  return ret;
}

es_proto::PartitionList to_proto_partition_list(vector<partition_t> const& pl) {
  es_proto::PartitionList ret;
  for(auto const& p: pl) {
    p.to_proto(*ret.add_parts());
  }
  return ret;
}

vector<partition_t> from_wire_partition_list(string const& str) {
  es_proto::PartitionList ret;
  if(!ret.ParseFromString(str)) {
    throw std::runtime_error("could not parse partition list");
  }
  return from_proto_partition_list(ret);
}

string to_wire_partition_list(vector<partition_t> const& pl) {
  es_proto::PartitionList ret = to_proto_partition_list(pl);
  string str;
  ret.SerializeToString(&str);
  return str;
}
