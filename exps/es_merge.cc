#include "../src/einsummable/einsummable.h"

int main() {
//  {
//    einsummable_t matmul = einsummable_t::from_matmul(100, 100, 100);
//    einsummable_t exp(
//      vector<uint64_t>{100,100},
//      { vector<int>{0,1} },
//      2,
//      scalarop_t::make_exp());
//
//    einsummable_t x = einsummable_t::merge(0, exp, matmul);
//    DOUT(x);
//
//    einsummable_t y = einsummable_t::merge(1, exp, x);
//    DOUT(y);
//  }

//  {
//    einsummable_t matmul = einsummable_t::from_matmul(100, 100, 100);
//    einsummable_t add(
//      vector<uint64_t>{100,100},
//      { vector<int>{0,1}, vector<int>{1} },
//      2,
//      scalarop_t::make_add());
//
//    DOUT("add                         " << add);
//    DOUT("matmul                      " << matmul);
//    einsummable_t x = einsummable_t::merge(0, add, matmul);
//    DOUT("matmul(add(x,y),z)          " << x);
//
//    einsummable_t y = einsummable_t::merge(2, add, x);
//    DOUT("matmul(add(x,y),add(z,w))   " << y);
//  }

//  {
//    // ij,jk->ik
//    einsummable_t matmul = einsummable_t::from_matmul(100, 101, 102);
//    einsummable_t add_for_rhs(
//      vector<uint64_t>{101,102},
//      { vector<int>{0,1}, vector<int>{1} },
//      2,
//      scalarop_t::make_add());
//    einsummable_t add_for_lhs(
//      vector<uint64_t>{100,101},
//      { vector<int>{0,1}, vector<int>{1} },
//      2,
//      scalarop_t::make_add());
//
//    DOUT("add                         " << add_for_rhs);
//    DOUT("matmul                      " << matmul);
//    einsummable_t x = einsummable_t::merge(1, add_for_rhs, matmul);
//    DOUT("matmul(x,add(y,z))          " << x);
//    einsummable_t y = einsummable_t::merge(0, add_for_lhs, x);
//    DOUT("matmul(add(x,y),add(z,w))   " << y);
//  }

  {
    einsummable_t mul(
      vector<uint64_t>{ 100, 101, 102 },
      { {0,2}, {1} },
      2,
      scalarop_t::make_mul(),
      castable_t::add);
    einsummable_t broadcast(
      vector<uint64_t>{ 100, 102 },
      { {0} },
      2,
      scalarop_t::make_identity());

    DOUT(broadcast);
    DOUT(mul);
    einsummable_t x = einsummable_t::merge(0, broadcast, mul);
    DOUT(x);
  }
}
