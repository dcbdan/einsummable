#pragma once
#include "setup.h"

#include "indexer.h"

template <typename T>
struct tensor_t {
  tensor_t(){}

  tensor_t(tensor_t const& other):
    tensor_t(other.shape, other.vec)
  {}
  tensor_t(tensor_t && other):
    tensor_t(other.shape, std::move(other.vec))
  {}

  tensor_t operator=(tensor_t const& other) {
    return tensor_t(other);
  }
  tensor_t operator=(tensor_t && other) {
    return tensor_t(std::move(other));
  }

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

  tensor_t(
    vector<int> const& shape,
    vector<T> && vec):
      shape(shape), vec(std::move(vec))
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

  vector<int> const& get_shape() const {
    return shape;
  }

private:
  vector<int> shape;
  vector<T> vec;
};
