#include "buffer.h"

buffer_t make_buffer(uint64_t size) {
  return std::make_shared<buffer_holder_t>(size);
}

buffer_t make_buffer_reference(uint8_t* data, uint64_t size) {
  return std::make_shared<buffer_holder_t>(data, size);
}

buffer_t make_buffer_reference(void* data, uint64_t size) {
  return std::make_shared<buffer_holder_t>(
    reinterpret_cast<uint8_t*>(data),
    size);
}

buffer_t make_buffer_copy(buffer_t other) {
  if(other) {
    buffer_t ret = make_buffer(other->size);
    std::memcpy(ret->raw(), other->raw(), other->size);
    return ret;
  } else {
    return nullptr;
  }
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

std::ostream& operator<<(std::ostream& out, buffer_holder_t const& b) {
  out << "buffer[" << b.size << "]{";
  if(b.size > 0) {
    out << uint32_t(b.data[0]);
    for(int i = 1; i != b.size; ++i) {
      out << "," << uint32_t(b.data[i]);
    }
  }
  out << "}";
  return out;
}
std::ostream& operator<<(std::ostream& out, buffer_t const& b) {
  out << (*b);
  return out;
}
