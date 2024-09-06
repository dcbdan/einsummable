#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <complex>
#include <variant>
#include <tuple>
#include <set>
#include <map>
#include <array>
#include <optional>
#include <sstream>
#include <string>
#include <random>
#include <queue>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include "half.hpp"

#define DOUT(x) \
  std::cout << x << std::endl;
#define DLINEOUT(x) \
  std::cout << "Line " << __LINE__ << " | " << __FILE__ << " | " << x << std::endl;
#define DLINE \
  DLINEOUT(' ')
#define DLINEFILEOUT(x) \
  std::cout << __FILE__ << " @ " << __LINE__ << " | " << x << std::endl;
#define DLINEFILE \
  DLINEFILEOUT(' ')

#define vector_from_each_member(items, member_type, member_name) [](auto const& xs) { \
    std::vector<member_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.member_name; }); \
    return ret; \
  }(items)

#define vector_from_each_method(items, type, method) [](auto const& xs) { \
    std::vector<type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.method(); }); \
    return ret; \
  }(items)

#define vector_from_each_tuple(items, which_type, which) [](auto const& xs) { \
    std::vector<which_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return std::get<which>(x); }); \
    return ret; \
  }(items)

using std::vector;
using std::tuple;
using std::set;
using std::map;
using std::optional;
using std::string;

using float16_t = half_float::half;

using hrect_t = vector<tuple<uint64_t, uint64_t>>;

template <typename T>
T product(vector<T> const& xs)
{
  T ret = 1;
  for(T const& x: xs) {
    ret *= x;
  }
  return ret;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, tuple<T, U> const& x12) {
  auto const& [x1,x2] = x12;
  out << "tup[" << x1 << "|" << x2 << "]";
  return out;
}

template <typename T>
void print_vec(vector<T> const& xs)
{
  print_vec(std::cout, xs);
}


template <typename T>
void print_vec(std::ostream& out, vector<T> const& xs)
{
  out << "{";
  if(xs.size() >= 1) {
    out << xs[0];
  }
  if(xs.size() > 1) {
    for(int i = 1; i != xs.size(); ++i) {
      out << "," << xs[i];
    }
  }
  out << "}";
}


template <typename T>
std::ostream& operator<<(std::ostream& out, vector<T> const& ts) {
  print_vec(out, ts);
  return out;
}


uint64_t uint64_div(uint64_t top, uint64_t bot, string err_msg);

vector<uint64_t> divide_evenly(int num_parts, uint64_t n);

template <typename T, typename U>
vector<T> vector_mapfst(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 0);
}

template <typename T, typename U>
vector<U> vector_mapsnd(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, U, 1);
}

template <typename T>
[[nodiscard]] vector<T> vector_concatenate(vector<T> vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
  return vs;
}
template <typename T>
void vector_concatenate_into(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
}

template <typename T>
vector<T> vector_flatten(vector<vector<T>> const& vvs) {
  vector<T> ret;
  for(vector<T> const& vs: vvs) {
    vector_concatenate_into(ret, vs);
  }
  return ret;
}

template <typename T>
T vector_min_element(vector<T> const& xs) {
  if(xs.size() == 0) {
    throw std::runtime_error("vector_min_element input empty");
  }
  return *std::min_element(xs.begin(), xs.end());
}

template <typename T>
T vector_max_element(vector<T> const& xs) {
  if(xs.size() == 0) {
    throw std::runtime_error("vector_max_element input empty");
  }
  return *std::max_element(xs.begin(), xs.end());
}

template <typename T, typename F>
void vector_doeach(vector<T>& xs, F f) {
  for(T& x: xs) { f(x); }
}

template <typename T, typename F>
void vector_doeach(vector<T> const& xs, F f) {
  for(T const& x: xs) { f(x); }
}

#define vector_domethod(xs, f) \
  vector_doeach(xs, std::mem_fn(&std::remove_reference<decltype(xs[0])>::type::f));

template <typename X, typename F>
auto vector_max_transform(vector<X> const& xs, F f) -> decltype(f(xs[0])) {
  if(xs.size() == 0) {
    throw std::runtime_error("vector_max_transform: empty input");
  }

  using T = decltype(f(xs[0]));

  T vmax = f(xs[0]);
  T vtest;
  for(int i = 1; i != xs.size(); ++i) {
    vtest = f(xs[i]);
    if(vmax < vtest) {
      vmax = vtest;
    }
  }
  return vmax;
}

#define vector_max_method(xs, f) \
  vector_max_transform(xs, std::mem_fn(&std::remove_reference<decltype(xs[0])>::type::f))

template <typename T>
T vector_sum(vector<T> const& xs) {
  T ret = 0;
  for(auto const& x: xs) {
    ret += x;
  }
  return ret;
}

template <typename T>
[[nodiscard]] vector<T> vector_add(vector<T> const& lhs, vector<T> const& rhs) {
  vector<T> ret;
  ret.reserve(lhs.size());
  for(int i = 0; i != lhs.size(); ++i) {
    ret.push_back(lhs[i] + rhs[i]);
  }
  return ret;
}

template <typename T>
vector<T> vector_double(vector<T> const& inn) {
  return vector_add(inn, inn);
}

template <typename T>
vector<T> vector_halve(vector<T> const& inn) {
  vector<T> ret;
  ret.reserve(inn.size());
  for(int i = 0; i != inn.size(); ++i) {
    ret.push_back(inn[i] / 2);
  }
  return ret;
}

template <typename T>
void vector_add_into(vector<T>& out, vector<T> const& inn) {
  for(int i = 0; i != out.size(); ++i) {
    out[i] += inn[i];
  }
}

template <typename T>
[[nodiscard]] vector<T> vector_sub(vector<T> const& lhs, vector<T> const& rhs) {
  vector<T> ret;
  ret.reserve(lhs.size());
  for(int i = 0; i != lhs.size(); ++i) {
    ret.push_back(lhs[i] - rhs[i]);
  }
  return ret;
}

template <typename T>
void vector_sub_into(vector<T>& out, vector<T> const& inn) {
  for(int i = 0; i != out.size(); ++i) {
    out[i] -= inn[i];
  }
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
  while(j != xs.size()) {
    xs[i++] = xs[j++];
    while(j != xs.size() && xs[i-1] == xs[j]) {
      ++j;
    }
  }
  xs.resize(i);
}

template <typename T>
void vector_uniqueify_inplace(vector<T>& xs) {
  if(xs.size() <= 1) {
    return;
  }

  set<std::size_t> discard;
  for(std::size_t i = 0; i != xs.size()-1; ++i) {
    for(std::size_t j = i+1; j != xs.size(); ++j) {
      if(xs[i] == xs[j]) {
        discard.insert(j);
      }
    }
  }

  auto s = discard.begin();
  std::size_t j = 0;
  for(int i = 0; i != xs.size(); ++i) {
    if(s == discard.end() || *s != i) {
      xs[j++] = xs[i];
    } else {
      s++;
    }
  }
  xs.resize(j);
}

template <typename T>
[[nodiscard]] vector<T> vector_uniqueify(vector<T> const& xs)
{
  if(xs.size() <= 1) {
    return xs;
  }

  set<std::size_t> discard;
  for(std::size_t i = 0; i != xs.size()-1; ++i) {
    for(std::size_t j = i+1; j != xs.size(); ++j) {
      if(xs[i] == xs[j]) {
        discard.insert(j);
      }
    }
  }

  auto s = discard.begin();
  vector<T> ret;
  ret.reserve(xs.size() - discard.size());
  for(int i = 0; i != xs.size(); ++i) {
    if(s == discard.end() || *s != i) {
      ret.push_back(xs[i]);
    } else {
      s++;
    }
  }
  return ret;
}

// Take a bunch of sorted lists and merge em into a single sorted list
// This is nlogn, but for n < 100, it's pretty fast cuz std::sort is fast.
// For lorge n, use a nlogk algorthm where k = xs.size(). An implementation
// tested wasnt faster until n > 5000.
template <typename T>
vector<T> vector_sorted_merges(vector<vector<T>> const& xs) {
  vector<T> ret;
  for(auto const& x: xs) {
    vector_concatenate_into(ret, x);
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

template <typename T>
void vector_erase_value(vector<T>& xs, T const& value)
{
  auto which = std::find(xs.begin(), xs.end(), value);
  if(which == xs.end()) {
    throw std::runtime_error("vector_erase_value: no value found");
  }
  xs.erase(which);
}

template <typename T>
bool vector_has(vector<T> const& xs, T const& value)
{
  return std::find(xs.begin(), xs.end(), value) != xs.end();
}

template <typename T1, typename T2>
vector<tuple<T1, T2>> vector_zip(
  vector<T1> const& lhs,
  vector<T2> const& rhs)
{
  if(lhs.size() != rhs.size()) {
    throw std::runtime_error("vector_zip expects inputs to be the same size");
  }

  vector<tuple<T1, T2>> ret;
  ret.reserve(lhs.size());
  for(int i = 0; i != lhs.size(); ++i) {
    ret.emplace_back(lhs[i], rhs[i]);
  }

  return ret;
}

template <typename T1, typename T2>
tuple<vector<T1>, vector<T2>> vector_unzip(
  vector<tuple<T1, T2>> const& xs)
{
  vector<T1> lhs;
  vector<T2> rhs;

  lhs.reserve(xs.size());
  rhs.reserve(xs.size());

  for(auto const& [l,r]: xs) {
    lhs.push_back(l);
    rhs.push_back(r);
  }

  return {lhs,rhs};
}

template <typename T, typename F>
void vector_filter_inplace(vector<T>& xs, F f)
{
  auto iter = std::copy_if(xs.begin(), xs.end(), xs.begin(), f);
  xs.resize(std::distance(xs.begin(), iter));
}

template <typename T, typename F>
void vector_sort_inplace(vector<T>& xs, F f)
{
  std::sort(xs.begin(), xs.end(), f);
}

template <typename K, typename V>
vector<K> map_get_keys(map<K, V> const& xys) {
  vector<K> ret;
  ret.reserve(xys.size());
  for(auto const& [x,_]: xys) {
    ret.push_back(x);
  }
  return ret;
}

template <typename T, typename F>
void set_erase_if_inplace(
  set<T>& xs, F f)
{
  for(auto iter = xs.begin(); iter != xs.end(); ) {
    if(f(*iter)) {
      iter = xs.erase(iter);
    } else {
      ++iter;
    }
  }
}

template <typename T>
vector<T> vector_iota(int n) {
  vector<T> ret(n);
  std::iota(ret.begin(), ret.end(), T(0));
  return ret;
}

template <typename RandomIter>
vector<std::size_t> argsort(RandomIter beg, RandomIter end) {
  vector<std::size_t> ret = vector_iota<std::size_t>(end-beg);
  std::sort(ret.begin(), ret.end(), [&](int const& lhs, int const& rhs) {
    return *(beg + lhs) < *(beg + rhs);
  });
  return ret;
}

template <typename T>
set<T> set_minus(set<T> const& all_these, set<T> const& except_these)
{
  set<T> ret;
  for(auto const& v: all_these) {
    if(except_these.count(v) == 0) {
      ret.insert(v);
    }
  }
  return ret;
}

template <typename T>
bool set_has_empty_intersection(set<T> const& lhs, set<T> const& rhs)
{
  for(auto const& v: lhs) {
    if(rhs.count(v) > 0) {
      return false;
    }
  }
  return true;
}

template <typename T>
void set_union_inplace(set<T>& ret, set<T> const& these) {
  for(auto const& x: these) {
    ret.insert(x);
  }
}

template <typename T>
void set_union_inplace(set<T>& ret, vector<T> const& these) {
  for(auto const& x: these) {
    ret.insert(x);
  }
}

template <typename Iter, typename F>
Iter max_element_transform(
  Iter first,
  Iter last,
  F f)
{
  // TODO: create a transform iterator instead of
  //       copying everything into scores
  vector<decltype(f(*first))> scores;
  scores.reserve(last-first);
  for(Iter iter = first; iter != last; ++iter) {
    scores.push_back(f(*iter));
  }
  int offset = std::max_element(scores.begin(), scores.end()) - scores.begin();
  return first + offset;
}

template <typename Iter, typename F>
Iter min_element_transform(
  Iter first,
  Iter last,
  F f)
{
  // TODO: same as max_element_transform
  vector<decltype(f(*first))> scores;
  scores.reserve(last-first);
  for(Iter iter = first; iter != last; ++iter) {
    scores.push_back(f(*iter));
  }
  int offset = std::min_element(scores.begin(), scores.end()) - scores.begin();
  return first + offset;
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

template <typename T>
T parse_with_ss(string const& s)
{
  T out;
  std::istringstream ss(s);
  ss >> out;
  return out;
}

// Parse [], [T], [T,...,T] with parse_with_ss for each element
template <typename T>
vector<T> parse_vector(string const& s, char sep = ',', char open = '[', char close = ']')
{
  vector<T> ret;

  if(s.size() < 2) {
    throw std::runtime_error("failed to parse vector: len < 2");
  }
  if(*s.begin() != open or *(s.end() - 1) != close) {
    throw std::runtime_error("parse vector: needs brackets");
  }

  auto xx = s.begin() + 1;
  auto end = s.end() - 1;
  while(xx != end) {
    auto yy = std::find(xx, end, sep);
    if(xx == yy) {
      throw std::runtime_error("parse_vector: empty substring");
    }
    ret.push_back(parse_with_ss<T>(std::string(xx,yy)));
    if(yy == end) {
      xx = end;
    } else {
      xx = yy + 1;
    }
  }

  return ret;
}

template <typename T>
string write_with_ss(T const& val)
{
  std::ostringstream ss;
  ss << val;
  return ss.str();
}

void set_seed(int seed);

std::mt19937& random_gen();

// random number in [beg,end)
int runif(int beg, int end);

// random number in [0,n)
int runif(int n);

int runif(vector<double> probs);

double rnorm();

template <typename T>
T vector_random_pop(vector<T>& xs) {
  int idx = runif(xs.size());
  auto iter = xs.begin() + idx;
  T ret = *iter;
  xs.erase(iter);
  return ret;
}

template <typename T>
using priority_queue_least = std::priority_queue<T, vector<T>, std::greater<T>>;
// For priority_queue_least, the top most element is the smallest,
// which is the opposite behaviour of priority_queue which puts the
// largest element at the top.

bool in_range(int val, int beg, int end);

#define clock_now std::chrono::high_resolution_clock::now
#define steady_now std::chrono::steady_clock::now

using timestamp_t = decltype(clock_now());
using steady_timestamp_t = decltype(steady_now());

struct raii_print_time_elapsed_t {
  raii_print_time_elapsed_t(string msg, bool hide=false):
    msg(msg), start(clock_now()), out(std::cout), hide(hide)
  {}

  raii_print_time_elapsed_t():
    msg(), start(clock_now()), out(std::cout)
  {}

  ~raii_print_time_elapsed_t() {
    if(hide) {
      return;
    }
    auto end = clock_now();
    using namespace std::chrono;
    auto duration = (double) duration_cast<microseconds>(end - start).count()
                  / (double) duration_cast<microseconds>(1s         ).count();

    if(msg.size() > 0) {
      out << msg << " | ";
    }
    out << "Total Time (seconds): " << duration << std::endl;
  }

  string const msg;
  timestamp_t const start;
  std::ostream& out;
  bool hide;
};

using gremlin_t = raii_print_time_elapsed_t;

// Example:
//   struct ab_t {
//     int a;
//     int b;
//   }
//   bool operator<(ab_t const& lhs, ab_t const& rhs) {
//     return two_tuple_lt(lhs, rhs);
//   }
template <typename T>
inline bool two_tuple_lt(T const& lhs, T const& rhs) {
  auto const& [lhs_a, lhs_b] = lhs;
  auto const& [rhs_a, rhs_b] = rhs;
  if(lhs_a < rhs_a) {
    return true;
  }
  if(lhs_a == rhs_a) {
    return lhs_b < rhs_b;
  }
  return false;
}
template <typename T>
inline bool two_tuple_eq(T const& lhs, T const& rhs) {
  auto const& [lhs_a, lhs_b] = lhs;
  auto const& [rhs_a, rhs_b] = rhs;
  return lhs_a == rhs_a && lhs_b == rhs_b;
}

template <typename T>
inline bool three_tuple_lt(T const& lhs, T const& rhs) {
  auto const& [lhs_a, lhs_b, lhs_c] = lhs;
  auto const& [rhs_a, rhs_b, rhs_c] = rhs;
  if(lhs_a < rhs_a) {
    return true;
  }
  if(lhs_a == rhs_a) {
    if(lhs_b < rhs_b) {
      return true;
    }
    if(lhs_b == rhs_b) {
      return lhs_c < rhs_c;
    }
  }
  return false;
}

template <typename T>
inline bool three_tuple_eq(T const& lhs, T const& rhs) {
  auto const& [lhs_a, lhs_b, lhs_c] = lhs;
  auto const& [rhs_a, rhs_b, rhs_c] = rhs;
  return lhs_a == rhs_a && lhs_b == rhs_b && lhs_c == rhs_c;
}

template <typename T>
struct equal_items_t {
  equal_items_t() {}

  equal_items_t(set<tuple<T,T>> const& eqs) {
    for(auto const& [a,b]: eqs) {
      insert(a,b);
    }
  }
  equal_items_t(vector<tuple<T,T>> const& eqs) {
    for(auto const& [a,b]: eqs) {
      insert(a,b);
    }
  }

  bool has(T const& a) const {
    return to_sets.count(a) > 0;
  }

  // warning: a is in the return set
  set<T> const& get_at(T const& a) const {
    int const& idx = to_sets.at(a);
    return sets[idx];
  }

  set<T> pop_at(T const& a) {
    int idx = to_sets.at(a);

    set<T> ret = sets[idx];

    sets.erase(sets.begin() + idx);

    for(auto const& x: ret) {
      to_sets.erase(x);
    }

    for(auto& [_,i]: to_sets) {
      if(i > idx) {
        i -= 1;
      }
    }

    return ret;
  }

  void erase_at(T const& a) const {
    pop_at(a);
  }

  vector<T> candidates() const {
    vector<T> ret;
    ret.reserve(sets.size());
    for(auto const& s: sets) {
      ret.push_back(*s.begin());
    }
    return ret;
  }

  void insert(T const& a, T const& b) {
    if(a == b) {
      sets.push_back({a});
      to_sets.insert({a, sets.size()-1});
      return;
    }

    bool has_a = to_sets.count(a) > 0;
    bool has_b = to_sets.count(b) > 0;
    if(has_a && has_b) {
      int which_a = to_sets[a];
      int which_b = to_sets[b];
      if(which_a == which_b) {
        // nothing to do, they already belong to the same set
      } else {
        // the sets need to merged

        set<T> set_a = sets[which_a];
        set<T> set_b = sets[which_b];

        int const& which_sml = which_a < which_b ? which_a : which_b;
        int const& which_big = which_a < which_b ? which_b : which_a;

        sets.erase(sets.begin() + which_big);
        sets.erase(sets.begin() + which_sml);

        set<T>& set_ab = set_b;
        set_ab.insert(set_a.begin(), set_a.end());

        sets.push_back(std::move(set_ab));

        // fix the mapping
        for(auto& [_, idx]: to_sets) {
          if(idx == which_a || idx == which_b) {
            idx = sets.size() - 1;
          } else if(idx > which_big) {
            idx -= 2;
          } else if(idx > which_sml) {
            idx -= 1;
          }
        }
      }
    } else if(has_a || has_b) {
      // put the not-has into the yes-has
      T const& y = has_a ? a : b;
      T const& n = has_a ? b : a;

      int id = to_sets[y];

      sets[id].insert(n);
      to_sets.insert({n, id});
    } else {
      // create a new set for both a and b
      sets.push_back({a,b});
      int new_id = sets.size() - 1;
      to_sets.insert({a, new_id});
      to_sets.insert({b, new_id});
    }
  }
private:
  vector<set<T>> sets;
  map<T, int> to_sets;
};

// This is so you can do range based for loops:
//
//  // print 1,2,3,4,5
//  set<int> xs{5,4,3,2,1};
//  for(int const& item: set_in_order(xs)) {
//    std::cout << item << std::endl;
//  }
//
// Note that you can't do for(int& item: set_in_order(x)) {...}
//   since modifying elements of a std::set can't be done (that would
//   invalidate set internals)
//
// (This function doesn't do anything since std::set keeps things
//  orded for you--but it keeps one from having to remember that
//  when all one cares is that a set is a bag of elements)
template <typename T>
inline set<T> const& set_in_order(set<T> const& items) {
  return items;
}

void hash_combine_impl(std::size_t& seed, std::size_t value);

template <typename T>
optional<string> check_concat_shapes(
  int dim,
  vector<vector<T>> const& shapes)
{
  if(shapes.size() == 0) {
    return "cannot be empty list of shapes";
  }

  // they should all have the same rank
  int rank = shapes[0].size();
  for(int i = 1; i != shapes.size(); ++i) {
    if(shapes[i].size() != rank) {
      return "invalid input size";
    }
  }

  if(dim < 0 || dim >= rank) {
    return "invalid dim";
  }

  // every dim should be the same, except dim
  vector<T> dim_parts;
  for(int r = 0; r != rank; ++r) {
    if(r != dim) {
      T d = shapes[0][r];
      for(int i = 1; i != shapes.size(); ++i) {
        if(shapes[i][r] != d) {
          return "non-concat dimensions do not line up";
        }
      }
    }
  }

  return std::nullopt;
}

// return the smallest value greater than or equal to number
// that is divisible by 2^power.
uint64_t align_to_power_of_two(uint64_t number, uint8_t power);

// Find the last true element
// Assumption: evaluate returns all trues then all falses.
// If there are no trues: return end
// If there are all trues: return end-1
template <typename Iter, typename F>
Iter binary_search_find(Iter beg, Iter end, F evaluate)
{
  if(beg == end) {
    return end;
  }
  if(!evaluate(*beg)) {
    return end;
  }

  decltype(std::distance(beg,end)) df;
  while((df = std::distance(beg, end)) > 2) {
    Iter mid = beg + (df / 2);
    if(evaluate(*mid)) {
      beg = mid;
    } else {
      end = mid;
    }
  }

  if(df == 1) {
    return beg;
  }

  if(evaluate(*(end - 1))) {
    return end-1;
  } else {
    return beg;
  }
}

uint64_t uint64_log2(uint64_t val);

// out_perm = {2,0,1} implies we have ijk->kij
vector<int> as_out_perm(
  vector<int> const& inn,
  vector<int> const& out);

template <typename T>
vector<T> forward_permute(
  vector<int> const& out_perm,
  vector<T> const& inn_shape)
{
  if(inn_shape.size() != out_perm.size()) {
    throw std::runtime_error("incorrect sizing: forward permute");
  }

  // 012->201
  // vvv->???
  vector<T> ret;
  ret.reserve(inn_shape.size());
  for(int i = 0; i != inn_shape.size(); ++i) {
    ret.push_back(inn_shape[out_perm[i]]);
  }
  return ret;
}

template <typename T>
vector<T> backward_permute(
  vector<int> const& out_perm,
  vector<T> const& out_shape)
{
  vector<int> modes = vector_iota<int>(out_perm.size());

  return forward_permute(
    as_out_perm(out_perm, modes),
    out_shape);
}

template <typename Iter>
struct iter_greater_t {
  bool operator()(Iter const& lhs, Iter const& rhs) const {
    return *lhs > *rhs;
  }
};

template <typename Iter>
vector<Iter> select_topn(Iter beg, Iter end, int topn) {
  std::priority_queue<Iter, std::vector<Iter>, iter_greater_t<Iter>> q;
  auto& iter = beg;
  for(; iter != end && q.size() != topn; ++iter) {
    q.push(iter);
  }
  for(; iter != end; ++iter) {
    q.push(iter);
    q.pop();
  }
  vector<Iter> ret;
  while(q.size() > 0) {
    ret.push_back(q.top());
    q.pop();
  }

  std::reverse(ret.begin(), ret.end());

  return ret;
}

// make sure that istream has xs up next; throw an error if not
void istream_expect(std::istream& inn, string const& xs);

// find the longest parse of the options; throw an error if no parse
int istream_expect_or(std::istream& inn, vector<string> const& options);

string istream_consume_alphanumeric(std::istream& inn);
string istream_consume_alphanumeric_u(std::istream& inn);

bool is_alphanumeric(string const& s);
bool is_alphanumeric_u(string const& s);

struct unit_t {};

void* increment_void_ptr(void* ptr, uint64_t size);
void const* increment_void_ptr(void const* ptr, uint64_t size);
