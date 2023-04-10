#include "setup.h"

int product(vector<int> const& xs) {
  int ret = 1;
  for(int const& x: xs) {
    ret *= x;
  }
  return ret;
}

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


