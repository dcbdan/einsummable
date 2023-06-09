#include "buffer.h"

buffer_t make_buffer(uint64_t size) {
  return std::make_shared<buffer_holder_t>(size);
}

buffer_t make_buffer_reference(uint8_t* data, uint64_t size) {
  return std::make_shared<buffer_holder_t>(data, size);
}

bool operator==(buffer_t const& lhs, buffer_t const& rhs) {
  return *lhs == *rhs;
}
bool operator!=(buffer_t const& lhs, buffer_t const& rhs) {
  return !(lhs == rhs);
}
bool operator==(buffer_holder_t const& lhs, buffer_holder_t const& rhs) {
  if(lhs.size != rhs.size) {
    return false;
  }
  for(int i = 0; i != lhs.size; ++i) {
    if(lhs.data[i] != rhs.data[i]) {
      return false;
    }
  }
  return true;
}
bool operator!=(buffer_holder_t const& lhs, buffer_holder_t const& rhs) {
  return !(lhs == rhs);
}


