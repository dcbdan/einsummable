#include "partition.h"

bool operator==(partition_t const& lhs, partition_t const& rhs) {
  return vector_equal(lhs.partdims, rhs.partdims);
}
bool operator!=(partition_t const& lhs, partition_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, partition_t const& p) {
  out << "partition" << p.partdims;
  return out;
}

