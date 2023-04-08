#include "einsummable.h"

einsummable_t einsummable_t::from_matmul(uint64_t di, uint64_t dj, uint64_t dk) {
    // ij,jk->ik
    // 02 21  01
    return einsummable_t {
      .join_shape = {di, dk, dj},
      .inns = { {0, 2}, {2, 1} },
      .out_rank = 2,
      .join = scalar_join_t::mul,
      .castable = castable_t::add
    };
  }
