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

void set_seed(int seed) {
  random_gen() = std::mt19937(seed);
}

std::mt19937& random_gen() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return gen;
}

// random number in [beg,end)
int runif(int beg, int end) {
  return std::uniform_int_distribution<>(beg, end-1)(random_gen());
}
int runif(int n) {
  return runif(0, n);
}
int runif(vector<double> probs) {
  if(probs.size() == 0) {
    throw std::runtime_error("invalid runif props");
  }
  std::inclusive_scan(probs.begin(), probs.end(), probs.begin());
  double v = std::uniform_real_distribution<double>(0.0, probs.back())(random_gen());
  int ret = std::lower_bound(probs.begin(), probs.end(), v) - probs.begin();
  if(ret == probs.size()) {
    // This shouldnt happen often
    ret -= 1;
  }
  return ret;
}

bool in_range(int val, int beg, int end) {
  return val >= beg && val < end;
}

// Stolen from http://myeyesareblind.com/2017/02/06/Combine-hash-values/
// where this is the boost implementation
void hash_combine_impl(std::size_t& seed, std::size_t value)
{
    seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

