#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/scalarop.cc"

int main() {
  std::cout << "trying to test creating an eimsummable_t" << std::endl;
  einsummable_t eins = einsummable_t({3, 4}, { {1,0} }, 2, scalarop_t::make_identity(), castable_t::add);

  //testing the str_to_inns_outrank
  std::string einsummable_str = "ij->sji";
  std::tuple<vector<vector<int>>, int> inns_outrank_tup = einsummable_t::str_to_inns_outrank(einsummable_str);
  vector<vector<int>> inns = std::get<0>(inns_outrank_tup);
  int outrank = std::get<1>(inns_outrank_tup);
  std::cout << "inns" << std::endl;

  //print the inns
  for (int i = 0; i < inns.size(); ++i) {
    for (int j = 0; j < inns[i].size(); ++j) {
      std::cout << inns[i][j] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "outrank :" << outrank << std::endl;
}
