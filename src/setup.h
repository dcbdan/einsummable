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

#define DOUT(x) std::cout << x << std::endl;
#define DLINEOUT(x) std::cout << __LINE__ << " " << x << std::endl;
#define DLINE DLINEOUT(' ')
#define DLINEFILEOUT(x) std::cout << __FILE__ << " @ " << __LINE__ << ": " << x << std::endl;
#define DLINEFILE DLINEFILEOUT(' ')

#define vector_from_each_member(xs, member_type, member_name) [](auto const& xs) { \
    std::vector<member_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.member_name; }); \
    return ret; \
  }(xs)

#define vector_from_each_method(xs, type, method) [](auto const& xs) { \
    std::vector<type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.method(); }); \
    return ret; \
  }(xs)

#define vector_from_each_tuple(xs, which_type, which) [](auto const& xs) { \
    std::vector<which_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return std::get<which>(x); }); \
    return ret; \
  }(xs)


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
  print_vec(std::cout, xs);
}

template <typename T>
void print_vec(std::ostream& out, vector<T> const& xs)
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

template <typename T>
std::ostream& operator<<(std::ostream& out, vector<T> const& ts) {
  print_vec(out, ts);
  return out;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, tuple<T, U> const& x12) {
  auto const& [x1,x2] = x12;
  out << "tup[" << x1 << "|" << x2 << "]";
  return out;
}

vector<uint64_t> divide_evenly(int num_parts, uint64_t n);

template <typename T, typename U>
vector<T> vector_mapfst(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 0);
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


