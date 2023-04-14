#pragma once
#include "setup.h"

#include "indexer.h"

template <typename T>
struct tensor_t {
  tensor_t(){}

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
      if(vec.size() != 0) {
        throw std::runtime_error("vec is not empty");
      }
    } else if(product(shape) != vec.size()) {
      throw std::runtime_error("tensor input length is incorrect");
    }
  }

  tensor_t(
    vector<int> const& shape,
    vector<T> && v):
      shape(shape), vec(std::move(v))
  {
    if(shape.size() == 0) {
      if(vec.size() != 0) {
        throw std::runtime_error("vec is not empty");
      }
    } else if(product(shape) != vec.size()) {
      throw std::runtime_error("tensor input length is incorrect");
    }
  }
  T& at(vector<int> idxs) {
    if(shape.size() == 0 || !is_valid_idxs(shape, idxs)) {
      throw std::runtime_error("invalid tensor access");
    }
    return vec[idxs_to_index(shape, idxs)];
  }

  T const& at(vector<int> idxs) const {
    if(shape.size() == 0 || !is_valid_idxs(shape, idxs)) {
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

template <typename T>
std::ostream& operator<<(std::ostream& out, tensor_t<T> const& tensor)
{
  auto const& shape = tensor.get_shape();
  if(shape.size() == 0) {
    out << "tensur[null]";
    return out;
  }
  out << "tensor" << shape << "@";
  vector<int> index(shape.size(), 0);
  do {
    out << "I" << index << tensor.at(index) << " ";
  } while(increment_idxs(shape, index));

  return out;
}

