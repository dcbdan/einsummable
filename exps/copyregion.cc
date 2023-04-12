#include "../src/copyregion.h"

void print(copyregion_t& c) {
  std::cout << "-----------------" << std::endl << std::endl;

  do {
    for(auto const& [idx, offset_inn, offset_out, size]: c.info) {
      std::cout << "idx:        " << idx        << std::endl;
      std::cout << "offset_inn: " << offset_inn << std::endl;
      std::cout << "offset_out: " << offset_out << std::endl;
      std::cout << "size:       " << size       << std::endl;

      std::cout << std::endl;
    }

    std::cout << "-----------------" << std::endl << std::endl;
  } while(c.increment());

}

int main() {
  uint64_t ni = 1000;
  uint64_t nj = 1000;

  partition_t singleton = partition_t::singleton({ni,nj});

  partition_t split2 = partition_t({
    partdim_t::split(ni, 2),
    partdim_t::split(nj, 2),
  });

  partition_t split3 = partition_t({
    partdim_t::split(ni, 3),
    partdim_t::split(nj, 3),
  });

  copyregion_t c1(singleton, split2, {1,1});
  copyregion_t c2(split2, singleton, {0,0});
  copyregion_t c3(split2, split3, {1,1});

  std::cout << "### " << "split2[1,1] to singleton" << std::endl;
  print(c1);

  std::cout << "### " << "singleton[0,0] to split2" << std::endl;
  print(c2);

  std::cout << "### " << "split3[1,1] to split2" << std::endl;
  print(c3);

  //std::cout << "### " << "" << std::endl;
}
