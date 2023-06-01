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

vector<uint64_t> partition_t::total_shape() const {
  return vector_from_each_method(partdims, uint64_t, total);
}

int partition_t::num_parts() const {
  return product(this->block_shape());
}

vector<int> partition_t::block_shape() const {
  return vector_from_each_method(partdims, int, num_parts);
}

vector<uint64_t> partition_t::tensor_shape_at(vector<int> const& idxs) const
{
  if(idxs.size() != partdims.size()) {
    throw std::runtime_error("partition_t::tensor_shape_at");
  }

  vector<uint64_t> ret;
  ret.reserve(partdims.size());
  for(int i = 0; i != partdims.size(); ++i) {
    ret.push_back(partdims[i].size_at(idxs[i]));
  }

  return ret;
}

// Get the hyper-rectanuglar set represnted by this index
vector<tuple<uint64_t, uint64_t>>
partition_t::get_hrect(vector<int> const& idxs) const
{
  if(idxs.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_hrect");
  }

  vector<tuple<uint64_t, uint64_t>> ret;
  ret.reserve(idxs.size());
  for(int i = 0; i != partdims.size(); ++i) {
    ret.push_back(partdims[i].which_vals(idxs[i]));
  }

  return ret;
}

// hrect:  hyper-rectangular subset of uint64s
// region: hyper-rectangular subset of blocks
vector<tuple<int,int> >
partition_t::get_exact_region(
  vector<tuple<uint64_t,uint64_t>> const& region) const
{
  if(region.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_exact_region");
  }
  vector<tuple<int,int> > ret;
  ret.reserve(region.size());
  for(int i = 0; i != partdims.size(); ++i) {
    auto const& [beg,end] = region[i];
    ret.push_back(partdims[i].exact_region(beg,end));
  }
  return ret;
}

vector<tuple<int,int> >
partition_t::get_region(
  vector<tuple<uint64_t,uint64_t>> const& region) const
{
  if(region.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_region");
  }
  vector<tuple<int,int> > ret;
  ret.reserve(region.size());
  for(int i = 0; i != partdims.size(); ++i) {
    auto const& [beg,end] = region[i];
    ret.push_back(partdims[i].region(beg,end));
  }
  return ret;
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

