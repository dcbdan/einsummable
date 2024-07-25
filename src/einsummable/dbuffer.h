#pragma once
#include "../base/setup.h"

#include "../base/buffer.h"
#include "scalarop.h"

struct dbuffer_t {
    dbuffer_t();
    dbuffer_t(dtype_t, buffer_t);

    void zeros();
    void ones();
    void fill(scalar_t val);
    void iota(int start = 0);
    void random(string lower = "0.0", string upper = "1.0");
    void random(scalar_t lower, scalar_t upper);
    void rnorm();
    void scale(scalar_t val);

    dbuffer_t copy(optional<dtype_t> to_dtype = std::nullopt);

    dbuffer_t view_c64_as_f32();
    dbuffer_t view_f32_as_c64();

    scalar_t sum() const;
    double   sum_to_f64() const;

    scalar_t min() const;
    scalar_t max() const;

    uint64_t        nelem() const;
    uint64_t const& size() const;

    void     set(uint64_t which_elem, scalar_t const& val);
    void     agg_into(uint64_t which_elem, castable_t, scalar_t const& val);
    scalar_t get(uint64_t which_elem) const;

    void*       ptr();
    void const* ptr() const;

    void* raw()
    {
        return data->raw();
    }
    float16_t*           f16();
    float*               f32();
    double*              f64();
    std::complex<float>* c64();

    void const* raw() const
    {
        return data->raw();
    }
    float16_t const*           f16() const;
    float const*               f32() const;
    double const*              f64() const;
    std::complex<float> const* c64() const;

    dtype_t  dtype;
    buffer_t data;
};

double dbuffer_squared_distance(dbuffer_t lhs, dbuffer_t rhs);

dbuffer_t make_dbuffer(dtype_t, uint64_t num_elems);

template <typename T>
bool is_close(T const& lhs, T const& rhs, float eps = 1e-3)
{
    return (lhs <= rhs + T(eps)) && (lhs >= rhs - T(eps));
}

bool is_close(dbuffer_t const& lhs, dbuffer_t const& rhs, float eps = 1e-3);

std::ostream& operator<<(std::ostream& out, dbuffer_t const& dbuffer);

bool operator==(dbuffer_t const& lhs, dbuffer_t const& rhs);
bool operator!=(dbuffer_t const& lhs, dbuffer_t const& rhs);

void print_nelems(dbuffer_t const& dbuffer);
