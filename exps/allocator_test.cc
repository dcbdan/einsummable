#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/scalarop.cc"

int main() {
std::cout << "trying to test new allocator_t" << std::endl;

  /*Take an example that we have a device with 100 bytes memory*/
  allocator_t allocator = allocator_t(100);

  allocator.print();
  DOUT("");

  auto [o0, _0] = allocator.allocate(6);
  auto [o1, _1] = allocator.allocate(4);
  auto [o2, _2] = allocator.allocate(2);
  auto [o3, _3] = allocator.allocate(7);
  DOUT("_0" << _0);
  DOUT("_1" << _1);
  DOUT("_2" << _2);
  DOUT("_3" << _3);
  allocator.free(o0,0);
  allocator.free(o1,0);
  allocator.free(o2,0);
  allocator.free(o3,0);
  allocator.print();
  auto [o5, _4] = allocator.allocate(10);
  DOUT("_4 " << _4);
  allocator.print();

}
