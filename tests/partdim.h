#include "../src/base/partdim.h"

optional<string> test_partdim_spans() {
  if(partdim_t::from_spans({3,5,10}) != partdim_t::from_sizes({3,2,5})) {
    return "from_spans {3,5,10} != from_sizes({3,2,5})";
  }
  return std::nullopt;
}

optional<string> test_partdim_split() {
  if(partdim_t::split(100, 2) != partdim_t::from_sizes({50, 50})) {
    return "split(100,2) != {50,50}";
  }
  if(partdim_t::split(100, 3) != partdim_t::from_sizes({34,33,33})) {
    return "split(100,3) != {34,33,33}";
  }
  return std::nullopt;
}
