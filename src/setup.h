#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <variant>
#include <tuple>

using std::vector;
using std::tuple;

int product(vector<int> const& xs);

void print_vec(vector<int> const& xs);

vector<uint64_t> divide_evenly(int num_parts, uint64_t n);

template <typename T>
void vector_concatenate(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
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
