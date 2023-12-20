#pragma once
#include "setup.h"

#include "indexer.h"

template <typename T>
struct vtensor_t {
  vtensor_t(){}

  vtensor_t(
    vector<int> const& shape):
      vtensor_t(shape, vector<T>(product(shape)))
  {}

  vtensor_t(
    vector<int> const& shape,
    T const& val):
      vtensor_t(shape, vector<T>(product(shape), val))
  {}

  vtensor_t(
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

  vtensor_t(
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
  T& operator()(Args... args) {
    return at(variadic_to_vec<int>(args...));
  }
  template <typename... Args>
  T const& operator()(Args... args) const {
    return at(variadic_to_vec<int>(args...));
  }
  // I'd prefer to use operator[], but C++ only allows
  // 1 input to operator[]. AND
  //   template <typename... Args>
  //   T& operator[](Args... args) {
  // compiles without any warning..
  // There you go.

  vector<T>& get() {
    return vec;
  }
  vector<T> const& get() const {
    return vec;
  }

  vector<int> const& get_shape() const {
    return shape;
  }

  void reshape(vector<int> const& new_shape) {
    if(product(shape) != product(new_shape)) {
      throw std::runtime_error("invalid reshape arg");
    }
    shape = new_shape;
  }

  // Example: Matrix with shape {10,20}
  //          index_subtensor(i) returns the 20 vector on row i
  vtensor_t<T> index_subtensor(int idx) const {
    return index_subtensor(vector<int>{idx});
  }
  // Example: Tensor with sahpe {10,20,30,40}
  //          index_subtensor({i,j}) returns the {30,40} matrix
  //          at {i,j} position.
  vtensor_t<T> index_subtensor(vector<int> const& idxs) const {
    if(idxs.size() >= shape.size()) {
      throw std::runtime_error("cannot index into a scalar");
    }
    vector<tuple<int,int>> region;
    region.reserve(shape.size());
    for(auto const& idx: idxs) {
      region.emplace_back(idx, idx+1);
    }
    for(int i = idxs.size(); i != shape.size(); ++i) {
      region.emplace_back(0, shape[i]);
    }

    return subset(region);
  }

  vtensor_t<T> subset(vector<tuple<int,int>> const& region) const {
    if(region.size() != shape.size()) {
      throw std::runtime_error("invalid subset region tenosr");
    }

    vector<int> new_shape;
    new_shape.reserve(region.size());
    for(int i = 0; i != region.size(); ++i) {
      auto const& [b,e] = region[i];
      int const& sz = shape[i];
      if(b < 0 || b >= e || e > sz) {
        throw std::runtime_error("invalid region subset tensor");
      }
      new_shape.push_back(e-b);
    }

    vector<T> ret;
    ret.reserve(product(new_shape));

    vector<int> index = vector_mapfst(region);
    do {
      ret.push_back(this->at(index));
    } while(increment_idxs_region(region, index));

    return vtensor_t<T>(new_shape, ret);
  }

  static vtensor_t<T> concat(int dim, vector<vtensor_t<T>> const& ts) {
    vector<vector<int>> shapes = vector_from_each_member(ts, vector<int>, shape);

    vector<int> shape;
    vector<int> offsets;
    {
      optional<string> errmsg = check_concat_shapes(dim, shapes);
      if(errmsg) {
        throw std::runtime_error("vtensor_t concat: " + errmsg.value());
      }

      int total = 0;
      for(auto const& s: shapes) {
        offsets.push_back(total);
        total += s[dim];
      }

      shape = shapes[0];
      shape[dim] = total;
    }

    vtensor_t<T> ret(shape);
    for(int which_inn = 0; which_inn != ts.size(); ++which_inn) {
      vtensor_t<T> const& inn = ts[which_inn];
      vector<int> inn_index(inn.shape.size(), 0);

      int const& offset = offsets[which_inn];
      do {
        vector<int> out_index = inn_index;
        out_index[dim] += offset;

        ret.at(out_index) = inn.at(inn_index);

      } while(increment_idxs(inn.shape, inn_index));
    }

    return ret;
  }

private:
  vector<int> shape;
  vector<T> vec;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, vtensor_t<T> const& tensor)
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

