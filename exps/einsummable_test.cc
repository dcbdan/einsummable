#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/scalarop.cc"

int main() {
  {
    einsummable_t e(
      {3,4,5,6},
      { {0,1,2,3}, {0,1} },
      4,
      scalarop_t::make_mul(),
      castable_t::add);
    DOUT(e);
    DOUT(e.merge_adjacent_dims());
  }
  DOUT("--------------");
  {
    einsummable_t e(
      {3,4,5,6},
      { {0,1,2,3} },
      2,
      scalarop_t::make_identity(),
      castable_t::add);
    DOUT(e);
    DOUT(e.merge_adjacent_dims());
  }
}
