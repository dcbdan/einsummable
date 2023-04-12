#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <variant>
#include <tuple>
#include <set>
#include <map>
#include <optional>

using std::vector;
using std::tuple;
using std::set;
using std::map;

template <typename T>
T product(vector<T> const& xs)
{
  T ret = 1;
  for(int const& x: xs) {
    ret *= x;
  }
  return ret;
}

template <typename T>
void print_vec(vector<T> const& xs)
{
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

vector<uint64_t> divide_evenly(int num_parts, uint64_t n);

template <typename T, typename U>
vector<T> vector_mapfst(vector<tuple<T, U>> const& xys) {
  vector<T> xs;
  xs.reserve(xys.size());
  for(auto const& [x,_]: xys) {
    xs.push_back(x);
  }
  return xs;
}

template <typename T>
void vector_concatenate_into(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
}

template <typename T>
vector<T> vector_concatenate(vector<T> vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
  return vs;
}

template <typename T>
bool vector_equal(vector<T> const& xs, vector<T> const& ys) {
  if(xs.size() != ys.size()) {
    return false;
  }
  for(int i = 0; i != xs.size(); ++i) {
    if(xs[i] != ys[i]) {
      return false;
    }
  }

  return true;
}

// Remove the duplicates in a sorted list
template <typename T>
void vector_remove_duplicates(vector<T>& xs) {
  std::size_t i = 0;
  std::size_t j = 0;
  while(j < xs.size()) {
    xs[i++] = xs[j++];
    while(xs[i-1] == xs[j]) {
      ++j;
    }
  }
  xs.resize(i);
}

// Take a bunch of sorted lists and merge em into a single sorted list
template <typename T>
vector<T> vector_sorted_merges(vector<vector<T>> const& xs) {
  // TODO: make this more efficient
  vector<T> ret;
  for(auto const& x: xs) {
    vector_concatenate(ret, x);
  }
  std::sort(ret.begin(), ret.end());
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

// -------------------
// |A   |A           |
// |----|------------|
// |A   |A  |A       |
// |    |   |B       |
// |    |   |        |
// -------------------
// A: full
// B: small
//
//  --------------
//  |   |Output  |
//  |   |        |
//  |   |        |
//  --------------
// It is an error if the small is not within the big
vector<tuple<uint64_t, uint64_t>>
center_hrect(
  vector<tuple<uint64_t, uint64_t>> const& full,
  vector<tuple<uint64_t, uint64_t>> const& small);


vector<uint64_t> shape_hrect(
  vector<tuple<uint64_t, uint64_t>> const& hrect);
