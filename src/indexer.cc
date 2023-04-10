#include "indexer.h"

bool is_valid_index(vector<int> const& shape, int index) {
  return index >= 0 && index < product(shape);
}

bool is_valid_idxs(vector<int> const& shape, vector<int> const& idxs) {
  if(shape.size() != idxs.size()) {
    return false;
  }

  for(int i = 0; i != shape.size(); ++i) {
    if(idxs[i] < 0 || idxs[i] >= shape[i]) {
      return false;
    }
  }

  return true;
}

int idxs_to_index(vector<int> const& shape, vector<int> const& idxs) {
  // ROW MAJOR.
  // The last indices "move the fastest"
  int p = 1;
  int total = 0;
  for(int i = idxs.size() - 1; i >= 0; --i) {
    total += p*idxs[i];
    p *= shape[i];
  }
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
vector<int> index_to_idxs(vector<int> const& shape, int index) {
  // ROW MAJOR.
  // The last indices "move the fastest"
  vector<int> ret(shape.size());
  int d = shape.back();
  for(int i = ret.size()-1; i >= 0; --i) {
    ret[i] = index % shape[i];
    index /= shape[i];
  }
  return ret;
};

bool increment_idxs(vector<int> const& shape, vector<int>& idxs) {
  bool could_increment = false;
  for(int i = idxs.size() - 1; i >= 0; --i) {
    if(idxs[i] + 1 == shape[i]) {
      idxs[i] = 0;
    } else {
      idxs[i] += 1;
      could_increment = true;
      break;
    }
  }
  return could_increment;
}

// (Mostly the same as increment_idxs)
bool increment_idxs_region(
  vector<tuple<int,int>> const& region,
  vector<int>& idxs)
{
  bool could_increment = false;
  for(int i = idxs.size() - 1; i >= 0; --i) {
    if(idxs[i] + 1 == std::get<1>(region[i])) {
      idxs[i] = std::get<0>(region[i]);
    } else {
      idxs[i] += 1;
      could_increment = true;
      break;
    }
  }
  return could_increment;
}
