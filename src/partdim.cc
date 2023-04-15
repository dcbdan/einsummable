#include "partdim.h"

bool operator==(partdim_t const& lhs, partdim_t const& rhs) {
  return vector_equal(lhs.spans, rhs.spans);
}
bool operator!=(partdim_t const& lhs, partdim_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, partdim_t const& partdim) {
  out << "partdim" << partdim.spans;
  return out;
}
