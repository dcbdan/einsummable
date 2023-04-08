#pragma once
#include "setup.h"

#include "indexer.h"

template <typename T>
struct tensor_t {
  tensor_t(
    vector<int> const& shape):
      tensor_t(shape, vector<T>(product(shape)))
  {}

  tensor_t(
    vector<int> const& shape,
    vector<T> const& vec):
      shape(shape), vec(vec)
  {
    if(shape.size() == 0) {
      throw std::runtime_error("shape must not be empty");
    }
  }

  T& at(vector<int> idxs) {
    if(!is_valid_idxs(shape, idxs)) {
      throw std::runtime_error("invalid tensor access");
    }
    return vec[idxs_to_index(shape, idxs)];
  }

  T const& at(vector<int> idxs) const {
    if(!is_valid_idxs(shape, idxs)) {
      throw std::runtime_error("invalid tensor access");
    }
    return vec[idxs_to_index(shape, idxs)];
  }

  // Args must be all ints or this won't compile
  template <typename... Args>
  T& operator[](Args... args) {
    return at(variadic_to_vec<int>(args...));
  }
  template <typename... Args>
  T const& operator[](Args... args) const {
    return at(variadic_to_vec<int>(args...));
  }

  vector<T> const& get() const {
    return vec;
  }

private:
  vector<int> const shape;
  vector<T> vec;
};
