#include "setup.h"

int product(vector<int> const& xs) {
  int ret = 1;
  for(int const& x: xs) {
    ret *= x;
  }
  return ret;
}

void print_vec(vector<int> const& xs) {
  std::cout << "{";
  if(xs.size() >= 1) {
    std::cout << xs[0];
  }
  if(xs.size() > 1) {
    for(int i = 1; i != xs.size(); ++i) {
      std::cout << "," << xs[i];
    }
  }
  std::cout << "}";
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


