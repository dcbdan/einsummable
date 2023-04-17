#include "partdim.h"

bool operator==(partdim_t const& lhs, partdim_t const& rhs) {
  return vector_equal(lhs.spans, rhs.spans);
}
bool operator!=(partdim_t const& lhs, partdim_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, partdim_t const& partdim) {
  if(partdim.num_parts() == 1) {
    out << "pd::singleton";
    return out;
  }

  if(partdim == partdim_t::split(partdim.total(), partdim.num_parts())) {
    out << "pd::split(" << partdim.total() << "," << partdim.num_parts() << ")";
    return out;
  }

  out << "pd::sizes" << partdim.sizes();
  return out;
}
