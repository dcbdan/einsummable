#pragma once

#include "setup.h"

// Note: I may be unsigned!
template <typename I>
struct indexer_utils {
  static bool is_valid_index(vector<I> const& shape, I index) {
    return index >= 0 && index < product(shape);
  }

  static bool is_valid_idxs(vector<I> const& shape, vector<I> const& idxs) {
    if(shape.size() != idxs.size()) {
      return false;
    }

    for(I i = 0; i != shape.size(); ++i) {
      if(idxs[i] < 0 || idxs[i] >= shape[i]) {
        return false;
      }
    }

    return true;
  }

  static I idxs_to_index(vector<I> const& shape, vector<I> const& idxs) {
    // ROW MAJOR.
    // The last indices "move the fastest"

    I p = 1;
    I total = 0;

    // Assuming shape and idxs are the same size and nonempty
    I i = idxs.size(); // so i > 0
    do {
      i -= 1;
      total += p*idxs[i];
      p *= shape[i];
    } while(i > 0);

    return total;
  }

  // (d0,d1) (i0,i1)
  // d1*i0 + i1
  //
  // (d0,d1,d2) (i0,i1,i2)
  // d2*d1*i0 + d2*i1 + i2
  //
  // (d0,d1,d2,d3) (i0,i1,i2,i3)
  //   d3*d2*d1*i0 + d3*d2*i1 + d3*i2 + i3
  // = d3*(d2*d1*i0 + d2*i1 + i2) + i3
  //
  // i3 = index % d3
  // index / d3
  // i2 = index % d2
  // index / d2
  // i1 = index % d1
  // index / d1
  // i0 = index % d0
  // index / d0
  static vector<I> index_to_idxs(vector<I> const& shape, I index) {
    // ROW MAJOR.
    // The last indices "move the fastest"
    vector<I> ret(shape.size());
    I i = ret.size();
    do {
      i -= 1;
      ret[i] = index % shape[i];
      index /= shape[i];
    } while (i > 0);

    return ret;
  };

  static bool increment_idxs(vector<I> const& shape, vector<I>& idxs) {
    bool could_increment = false;
    I i = idxs.size();
    do {
      i -= 1;
      if(idxs[i] + 1 == shape[i]) {
        idxs[i] = 0;
      } else {
        idxs[i] += 1;
        could_increment = true;
        break;
      }
    } while(i > 0);

    return could_increment;
  }

  // (Mostly the same as increment_idxs)
  static bool increment_idxs_region(
    vector<tuple<I,I>> const& region,
    vector<I>& idxs)
  {
    bool could_increment = false;

    I i = idxs.size();
    do {
      i -= 1;
      if(idxs[i] + 1 == std::get<1>(region[i])) {
        idxs[i] = std::get<0>(region[i]);
      } else {
        idxs[i] += 1;
        could_increment = true;
        break;
      }
    } while(i > 0);

    return could_increment;
  }
};

// TODO: come up with a consistent name for
//       vector<int> and int as index / idxs / indices / idxs / i / is
bool is_valid_index(vector<int> const& shape, int index);
bool is_valid_idxs(vector<int> const& shape, vector<int> const& idxs);
int idxs_to_index(vector<int> const& shape, vector<int> const& idxs);
vector<int> index_to_idxs(vector<int> const& shape, int index);
bool increment_idxs(vector<int> const& shape, vector<int>& idxs);
bool increment_idxs_region(
  vector<tuple<int,int>> const& region,
  vector<int>& idxs);

//int main() {
//  vector<int> shape{3,2,4,5};
//  int total = product(shape);
//  for(int i = 0; i != total; ++i) {
//    auto idxs = index_to_idxs(shape, i);
//    //std::cout << i << " | ";
//    //print_vec(idxs);
//    //std::cout << std::endl;
//    auto ii = idxs_to_index(shape, idxs);
//    if(i != ii) {
//      std::cout << "error at " << i << " with " << ii << std::endl;
//    }
//  }
//  std::cout << "done." << std::endl;
//};
//
//int main() {
//  int ni = 3;
//  int nj = 3;
//  int nk = 3;
//  vector<int> shape {ni,nj,nk};
//  for(int i = 0; i != ni; ++i) {
//  for(int j = 0; j != nj; ++j) {
//  for(int k = 0; k != nk; ++k) {
//    int index = idxs_to_index(shape, {i,j,k});
//    auto idxs = index_to_idxs(shape, index);
//    std::cout << index << " | " << idxs << std::endl;
//  }}}
//}

