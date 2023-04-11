#include "partdim.h"

bool operator==(partdim_t const& lhs, partdim_t const& rhs) {
  return vector_equal(lhs.spans, rhs.spans);
}
bool operator!=(partdim_t const& lhs, partdim_t const& rhs) {
  return !(lhs == rhs);
}


