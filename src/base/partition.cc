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

