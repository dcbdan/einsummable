inline
vector<uint64_t> partition_t::total_shape() const {
  return vector_from_each_method(partdims, uint64_t, total);
}

inline
int partition_t::num_parts() const {
  return product(this->block_shape());
}

inline
vector<int> partition_t::block_shape() const {
  return vector_from_each_method(partdims, int, num_parts);
}

inline
vector<int> partition_t::from_bid(int const& bid) const {
  vector<int> shape = block_shape();
  return index_to_idxs(shape, bid);
}

inline
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

inline
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

inline
vtensor_t<uint64_t> partition_t::all_block_sizes() const {
  vector<int> shape = block_shape();
  vector<int> idx(shape.size(), 0);

  vector<uint64_t> ret;
  ret.reserve(product(shape));
  do {
    ret.push_back(product(tensor_shape_at(idx)));
  } while(increment_idxs(shape, idx));

  return vtensor_t(shape, ret);
}

inline
uint64_t partition_t::block_size_at_bid(int bid) const {
  return product(tensor_shape_at(from_bid(bid)));
}

inline
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

inline
partition_t partition_t::subset(
  vector<tuple<uint64_t, uint64_t>> const& hrect) const
{
  if(hrect.size() != partdims.size()) {
    throw std::runtime_error("invalid region size for subsetting");
  }

  vector<partdim_t> pds;
  for(int i = 0; i != hrect.size(); ++i) {
    auto const& [b,e] = hrect[i];
    pds.push_back(partdims[i].subset(b,e));
  }

  return partition_t(pds);
}

// Get the hyper-rectanuglar set represnted by this index
inline
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
inline
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

inline
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

inline
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
