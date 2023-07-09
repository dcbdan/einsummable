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

uint64_t align_to_power_of_two(uint64_t number, uint8_t power) {
  if(number == 0) {
    return number;
  }

  number--;
  for(int i = 0; i != power; ++i) {
    number |= (1 << i);
  }
  number++;

  return number;
}

uint64_t uint64_log2(uint64_t val) {
  uint64_t ret = 0;
  while (val >>= 1) {
    ret++;
  }
  return ret;
}

vector<int> as_out_perm(
  vector<int> const& inn,
  vector<int> const& out)
{
  if(inn.size() != out.size()) {
    throw std::runtime_error("as out perm: must be same size");
  }

  // ijk->kij
  // 012->???

  map<int,int> i_to_o;
  for(int z = 0; z != inn.size(); ++z) {
    i_to_o[inn[z]] = z;
  }

  vector<int> out_perm;
  out_perm.reserve(out.size());
  for(int z = 0; z != out.size(); ++z) {
    out_perm.push_back(i_to_o[out[z]]);
  }

  return out_perm;
}

void istream_expect(std::istream& inn, string const& xs) {
  for(auto const& x: xs) {
    if(x != inn.get()) {
      throw std::runtime_error("expected " + xs);
    }
  }
}

int istream_expect_or(std::istream& inn, vector<string> const& options) {
  // One could only move the stream forward with peek and get,
  // which would probably be faster. But this instead just checks each
  // option one by one, which is good enough for the foreseeable use cases

  int ret = -1;
  int n = -1;

  auto reset_pos = inn.tellg();

  for(int which = 0; which != options.size(); ++which) {
    auto const& xs = options[which];

    inn.seekg(reset_pos);

    bool match = true;
    for(auto const& x: xs) {
      if(x == inn.peek()) {
        inn.get();
      } else {
        match = false;
        break;
      }
    }

    if(match) {
      if(int(xs.size()) > n) {
        ret = which;
        n = xs.size();
      }
    }
  }

  if(ret == -1) {
    throw std::runtime_error("could not find a match");
  }

  inn.seekg(reset_pos + decltype(reset_pos)(n));
  return ret;
}

