#include "setup.h"

vector<uint64_t> divide_evenly(int num_parts, uint64_t n) {
  if(n < num_parts) {
    throw std::runtime_error("Cannot have size zero parts");
  }
  vector<uint64_t> ret(num_parts, n / num_parts);
  uint64_t d = n % num_parts;
  for(int i = 0; i != d; ++i) {
    ret[i]++;
  }
  return ret;
}

vector<tuple<uint64_t, uint64_t>>
center_hrect(
  vector<tuple<uint64_t, uint64_t>> const& full,
  vector<tuple<uint64_t, uint64_t>> const& small)
{
  if(full.size() != small.size()) {
    throw std::runtime_error("center_hrect: incorrect sizes");
  }

  vector<tuple<uint64_t, uint64_t>> ret;
  ret.reserve(full.size());

  for(int i = 0; i != full.size(); ++i) {
    auto const& [fb,fe] = full[i];
    auto const& [sb,se] = small[i];
    if(fb <= sb && sb < se && se <= fe) {
      ret.emplace_back(sb-fb, se-fb);
    }
  }

  return ret;
}

vector<uint64_t> shape_hrect(
  vector<tuple<uint64_t, uint64_t>> const& hrect)
{
  vector<uint64_t> ret;
  ret.reserve(hrect.size());
  for(auto const& [x,y]: hrect) {
    ret.push_back(y-x);
  }
  return ret;
}
