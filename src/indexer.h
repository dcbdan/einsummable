#pragma once

#include "setup.h"

bool is_valid_index(vector<int> const& shape, int index);
bool is_valid_idxs(vector<int> const& shape, vector<int> const& idxs);
int idxs_to_index(vector<int> const& shape, vector<int> const& idxs);
vector<int> index_to_idxs(vector<int> const& shape, int index);
bool increment_idxs(vector<int> const& shape, vector<int>& idxs);

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
// int main() {
//   int ni = 3;
//   int nj = 3;
//   int nk = 3;
//   vector<int> shape {ni,nj,nk};
//   for(int i = 0; i != ni; ++i) {
//   for(int j = 0; j != nj; ++j) {
//   for(int k = 0; k != nk; ++k) {
//     int index = idxs_to_index(shape, {i,j,k});
//     auto idxs = index_to_idxs(shape, index);
//     std::cout << index << " | ";
//     print_vec(idxs);
//     std::cout << std::endl;
//   }}}
// }
