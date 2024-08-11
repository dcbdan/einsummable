#pragma once
#include "setup.h"

#include <memory>

struct buffer_holder_t {
    buffer_holder_t(uint64_t size) : size(size), own(true)
    {
        data = new uint8_t[size];
    }
    buffer_holder_t(uint8_t* data, uint64_t size) : size(size), own(false), data(data) {}
    ~buffer_holder_t()
    {
        if (own) {
            delete[] data;
        }
    }

    void* raw()
    {
        return reinterpret_cast<void*>(data);
    }
    void const* raw() const
    {
        return reinterpret_cast<void*>(data);
    }

    uint64_t size;
    bool     own;
    uint8_t* data;
};

using buffer_t = std::shared_ptr<buffer_holder_t>;

buffer_t make_buffer(uint64_t size);
buffer_t make_buffer_reference(uint8_t* data, uint64_t size);
buffer_t make_buffer_reference(void* data, uint64_t size);
buffer_t make_buffer_copy(buffer_t other);

bool operator==(buffer_holder_t const& lhs, buffer_holder_t const& rhs);
bool operator!=(buffer_holder_t const& lhs, buffer_holder_t const& rhs);

bool operator==(buffer_t const& lhs, buffer_t const& rhs);
bool operator!=(buffer_t const& lhs, buffer_t const& rhs);

std::ostream& operator<<(std::ostream& out, buffer_holder_t const& b);
std::ostream& operator<<(std::ostream& out, buffer_t const& b);
