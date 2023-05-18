#include "../src/einsummable/memgraph.h"

#include "../src/einsummable/einsummable.h"
#include "../src/einsummable/scalarop.h"
#include "../src/einsummable/scalarop.cc"

int main() {
  std::cout << "trying to test new allocator_t" << std::endl;

  /*Take an example that we have a device with 100 bytes memory*/
  allocator_t allocator = allocator_t(100);

  allocator.print();


  allocator.try_to_allocate(6);
  allocator.try_to_allocate(4);
  allocator.try_to_allocate(2);
  allocator.try_to_allocate(7);
  
  allocator.try_to_allocate(81);

  std::cout << "After initial allocation: " << std::endl << std::endl;
  allocator.print();


  allocator.free(6, 0);
  allocator.free(10, 1);
  allocator.free(0, 2);
  std::cout << "After free: " << std::endl << std::endl;

  allocator.print();

  std::cout<< "after reallocation:" << std::endl << std::endl;
  allocator.try_to_allocate(6);
  allocator.print();






}
