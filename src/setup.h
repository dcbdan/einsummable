#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

using std::vector;

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

template <typename T>
void vector_concatenate(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
}

vector<int> divide_evenly(int num_parts, uint64_t n) {
  if(n < num_parts) {
    throw std::runtime_error("Cannot have size zero parts");
  }
  vector<int> ret(num_parts, n / num_parts);
  uint64_t d = n % num_parts;
  for(int i = 0; i != d; ++i) {
    ret[i]++;
  }
  return ret;
}

template <typename T>
vector<T> _reverse_variadic_to_vec(T i) {
  vector<T> x(1, i);
  return x;
}
template <typename T, typename... Args>
vector<T> _reverse_variadic_to_vec(T i, Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  x.push_back(i);
  return x;
}

template <typename T, typename... Args>
vector<T> variadic_to_vec(Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  std::reverse(x.begin(), x.end());
  return x;
}
