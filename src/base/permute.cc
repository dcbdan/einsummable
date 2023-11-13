#include "permute.h"

permute_t::permute_t(uint64_t min_block_size):
  min_block_size(min_block_size)
{}

tuple<vector<uint64_t>, vector<uint64_t>>
permute_t::build_strides(
  vector<uint64_t> const& dims,
  vector<int> const& perm)
{
  using vec = vector<uint64_t>;
  tuple<vec,vec> ret(vec(dims.size()), vec(dims.size()));
  auto& [str_inn, str_out] = ret;

  // set the strides
  uint64_t m_inn = 1;
  uint64_t m_out = 1;
  for(int i = dims.size() - 1; i >= 0; --i) {
    str_inn[     i ] = m_inn;
    str_out[perm[i]] = m_out;

    m_inn *= dims[     i ];
    m_out *= dims[perm[i]];
  }

  return ret;
}

bool permute_t::has_fuse(vector<uint64_t>& dims, vector<int>& perm) const {
  for(int i = 0; i < perm.size()-1; ++i) {
    if(perm[i] + 1 == perm[i+1]) {
      int which = perm[i];
      dims[which] = dims[which] * dims[which+1];
      remove(which+1, dims, perm);
      return true;
    }
  }
  return false;
}

bool permute_t::has_singleton(vector<uint64_t>& dims, vector<int>& perm) const {
  for(int i = 0; i < dims.size()-1; ++i) {
    if(dims[i] == 1) {
      remove(i, dims, perm);
      return true;
    }
  }
  return false;
}

void permute_t::remove(int i, vector<uint64_t>& dims, vector<int>& perm) const {
  // i = 1
  // [d0,d1,d2,d3,d4]
  // [d0,d2,d3,d4]     <- copy over
  // [d0,d2,d3]        <- resize
  for(int x = i; x < dims.size()-1; ++x) {
    dims[x] = dims[x+1];
  }
  dims.resize(dims.size()-1);

  // [3,1,2,4,0]
  // [3,2,4,0,0] <- removed
  // [3,2,4,0]   <- resized
  // [2,1,3,0]   <- decremented

  // find where x lives
  int x = 0;
  for(; x != perm.size(); ++x) {
    if(perm[x] == i) {
      break;
    }
  }

  // shift to the left and resize
  for(; x < perm.size()-1; ++x) {
    perm[x] = perm[x+1];
  }
  perm.resize(perm.size()-1);

  // decrement things greater than i
  for(auto& p: perm) {
    if(p > i) {
      p--;
    }
  }
}

