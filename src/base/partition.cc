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

bool partition_t::refines(partition_t const& other) const
{
  if(other.partdims.size() != partdims.size()) {
    return false;
  }
  for(int i = 0; i != partdims.size(); ++i) {
    if(!partdims[i].refines(other.partdims[i])) {
      return false;
    }
  }
  return true;
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

partition_t partition_t::subset(
  vector<tuple<int, int>> const& region) const
{
  if(region.size() != partdims.size()) {
    throw std::runtime_error("invalid region size for subsetting");
  }

  vector<partdim_t> pds;

  for(int i = 0; i != region.size(); ++i) {
    auto const& [b,e] = region[i];
    auto full_szs = partdims[i].sizes();

    if(b < 0 || b >= e || e > full_szs.size()) {
      throw std::runtime_error("invalid region");
    }

    pds.push_back(partdim_t::from_sizes(vector<uint64_t>(
      full_szs.begin() + b,
      full_szs.begin() + e)));
  }

  return partition_t(pds);
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

vector<int> partition_t::get_index_covering(
  vector<tuple<uint64_t,uint64_t>> const& hrect) const
{
  vector<tuple<int,int>> ret = get_region(hrect);

  if(ret.size() != hrect.size()) {
    throw std::runtime_error("get index covering should not happen");
  }

  for(auto const& [b,e]: ret) {
    if(b >= e) {
      throw std::runtime_error("get index covering should not happen");
    }
    if(b + 1 != e) {
      throw std::runtime_error(
        "get_index_covering: cannot have multiple index");
    }
  }

  return vector_mapfst(ret);
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

