#include "../src/base/setup.h"

#include "../src/base/placement.h"

int main() {
  DOUT(partdim_t::from_sizes({10,10,10}).subset(5,25));
  DOUT(partdim_t::from_sizes({10,10,10}).subset(0,30));
  DOUT(partdim_t::from_sizes({10,10,10}).subset(0,10));
  DOUT(partdim_t::from_sizes({10,10,10}).subset(0,11));
  DOUT(partdim_t::from_sizes({10,10,10}).subset(15,18));
}

