#include "scalarop.h"

scalar_t agg_power(castable_t castable, uint64_t n, scalar_t val)
{
    if (n == 0) {
        throw std::runtime_error("agg power: invalid n value");
    }
    if (n == 1) {
        // no castable is applied when n = 1
        return val;
    }
    if (castable == castable_t::min || castable == castable_t::max) {
        if (dtype_is_complex(val.dtype)) {
            throw std::runtime_error("can't max min complex");
        } else {
            return val;
        }
    }
    if (castable == castable_t::add) {
        // TODO: what if n is really big?
        return scalarop_t::make_scale(scalar_t(val.dtype, write_with_ss(double(n)))).eval(val);
    }
    if (castable == castable_t::mul) {
        // TODO: what if n is really big?
        return scalarop_t::make_power(int(n), val.dtype).eval(val);
    }
    throw std::runtime_error("should not reach: agg power");
}

compare_t compare_flip(compare_t c)
{
    switch (c) {
        case compare_t::lt:
            return compare_t::ge;
        case compare_t::gt:
            return compare_t::le;
        case compare_t::eq:
            return compare_t::eq;
        case compare_t::le:
            return compare_t::gt;
        case compare_t::ge:
            return compare_t::lt;
    }
    throw std::runtime_error("compare_flip: should not reach");
}

uint64_t dtype_size(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return 2;
        case dtype_t::f32:
            return 4;
        case dtype_t::f64:
            return 8;
        case dtype_t::c64:
            return 8;
    }
    throw std::runtime_error("should not reach");
}

bool dtype_is_real(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return true;
        case dtype_t::f32:
            return true;
        case dtype_t::f64:
            return true;
        case dtype_t::c64:
            return false;
    }
    throw std::runtime_error("should not reach");
}

bool dtype_is_complex(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return false;
        case dtype_t::f32:
            return false;
        case dtype_t::f64:
            return false;
        case dtype_t::c64:
            return true;
    }
    throw std::runtime_error("should not reach");
}

dtype_t dtype_real_component(dtype_t dtype)
{
    if (dtype == dtype_t::c64) {
        return dtype_t::f32;
    }
    throw std::runtime_error("can't get real component dtype");
}

dtype_t dtype_random(bool include_complex)
{
    int dd = runif(include_complex ? 4 : 3);
    if (dd == 0) {
        return dtype_t::f16;
    } else if (dd == 1) {
        return dtype_t::f32;
    } else if (dd == 2) {
        return dtype_t::f64;
    } else if (dd == 3) {
        return dtype_t::c64;
    } else {
        throw std::runtime_error("dtype random should not reach");
    }
}

scalar_t::scalar_t() : scalar_t(float(0.0)) {}

scalar_t::scalar_t(dtype_t d, string const& s) : dtype(d)
{
    if (dtype == dtype_t::c64) {
        // not allowing this only because it's unclear what s would have to contain
        throw std::runtime_error("cannot create scalar_t from a complex string");
    } else if (dtype == dtype_t::f16) {
        f16() = parse_with_ss<float16_t>(s);
    } else if (dtype == dtype_t::f32) {
        f32() = parse_with_ss<float>(s);
    } else if (dtype == dtype_t::f64) {
        f64() = parse_with_ss<double>(s);
    } else {
        throw std::runtime_error("should not reach scalar");
    }
}

scalar_t::scalar_t(float16_t v) : dtype(dtype_t::f16)
{
    _copy_to_data(reinterpret_cast<uint8_t*>(&v), 2);
}

scalar_t::scalar_t(float v) : dtype(dtype_t::f32)
{
    _copy_to_data(reinterpret_cast<uint8_t*>(&v), 4);
}

scalar_t::scalar_t(double v) : dtype(dtype_t::f64)
{
    _copy_to_data(reinterpret_cast<uint8_t*>(&v), 8);
}

scalar_t::scalar_t(std::complex<float> v) : dtype(dtype_t::c64)
{
    _copy_to_data(reinterpret_cast<uint8_t*>(&v), 8);
}

scalar_t::scalar_t(scalar_t const& other) : dtype(other.dtype)
{
    _copy_to_data(other.data, 8);
}

void scalar_t::_copy_to_data(uint8_t const* other, int n)
{
    std::copy(other, other + n, data);
}

scalar_t scalar_t::zero(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return scalar_t(float16_t(0.0));
        case dtype_t::f32:
            return scalar_t(float(0.0));
        case dtype_t::f64:
            return scalar_t(double(0.0));
        case dtype_t::c64:
            return scalar_t(std::complex<float>(0.0, 0.0));
    }
    throw std::runtime_error("should not reach");
}

double _f64_inf()
{
    static_assert(std::numeric_limits<double>::is_iec559, "for inf");
    return std::numeric_limits<double>::infinity();
}
double _f64_ninf()
{
    static_assert(std::numeric_limits<double>::is_iec559, "for ninf");
    double ret = -std::numeric_limits<double>::infinity();
    return ret;
}
double _f64_nan()
{
    static_assert(std::numeric_limits<double>::is_iec559, "for nan");
    double ret = std::numeric_limits<double>::quiet_NaN();
    return ret;
}

static float16_t const& f16_inf()
{
    static float16_t ret(_f64_inf());
    return ret;
}
static float const& f32_inf()
{
    static float ret(_f64_inf());
    return ret;
}
static double const& f64_inf()
{
    static double ret(_f64_inf());
    return ret;
}

static float16_t const& f16_ninf()
{
    static float16_t ret(_f64_ninf());
    return ret;
}
static float const& f32_ninf()
{
    static float ret(_f64_ninf());
    return ret;
}
static double const& f64_ninf()
{
    static double ret(_f64_ninf());
    return ret;
}

static float16_t const& f16_nan()
{
    static float16_t ret(_f64_nan());
    return ret;
}
static float const& f32_nan()
{
    static float ret(_f64_nan());
    return ret;
}
static double const& f64_nan()
{
    static double ret(_f64_nan());
    return ret;
}

scalar_t scalar_t::negative_inf(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return scalar_t(f16_ninf());
        case dtype_t::f32:
            return scalar_t(f32_ninf());
        case dtype_t::f64:
            return scalar_t(f64_ninf());
        case dtype_t::c64:
            throw std::runtime_error("no -inf for complex");
    }
    throw std::runtime_error("should not reach");
}

scalar_t scalar_t::inf(dtype_t dtype)
{
    switch (dtype) {
        case dtype_t::f16:
            return scalar_t(f16_inf());
        case dtype_t::f32:
            return scalar_t(f32_inf());
        case dtype_t::f64:
            return scalar_t(f64_inf());
        case dtype_t::c64:
            throw std::runtime_error("no inf for complex");
    }
    throw std::runtime_error("should not reach");
}

scalar_t scalar_t::one(dtype_t dtype)
{
    // TODO: check if we should modify so it returns for c64
    // if(dtype == dtype_t::c64) {
    //   throw std::runtime_error("no one provided for c64");
    // }

    switch (dtype) {
        case dtype_t::f16:
            return scalar_t(float16_t(1.0));
        case dtype_t::f32:
            return scalar_t(float(1.0));
        case dtype_t::f64:
            return scalar_t(double(1.0));
        case dtype_t::c64:
            return scalar_t(std::complex<float>(1.0, 0.0));
    }
    throw std::runtime_error("should not reach");
}

scalar_t scalar_t::negative_one(dtype_t dtype)
{
    // if(dtype == dtype_t::c64) {
    //   throw std::runtime_error("no negative one provided for c64");
    // }

    switch (dtype) {
        case dtype_t::f16:
            return scalar_t(float16_t(-1.0));
        case dtype_t::f32:
            return scalar_t(float(-1.0));
        case dtype_t::f64:
            return scalar_t(double(-1.0));
        case dtype_t::c64:
            return scalar_t(std::complex<float>(-1.0, 0.0));
    }
    throw std::runtime_error("should not reach");
}

scalar_t scalar_t::convert(dtype_t d) const
{
    return scalar_t::convert(*this, d);
}

scalar_t scalar_t::convert(scalar_t const& other, dtype_t new_dtype)
{
    if (other.dtype == new_dtype) {
        return other;
    }
    if (other.dtype == dtype_t::c64) {
        throw std::runtime_error("cannot convert from complex");
    }
    if (other.dtype == dtype_t::f16) {
        float16_t const& v = other.f16();
        if (new_dtype == dtype_t::f32) {
            return scalar_t(float(v));
        } else if (new_dtype == dtype_t::f64) {
            return scalar_t(double(v));
        } else {
            throw std::runtime_error("scalar_t::convert: should not reach");
        }
    } else if (other.dtype == dtype_t::f32) {
        float const& v = other.f32();
        if (new_dtype == dtype_t::f16) {
            return scalar_t(float16_t(v));
        } else if (new_dtype == dtype_t::f64) {
            return scalar_t(double(v));
        } else {
            throw std::runtime_error("scalar_t::convert: should not reach");
        }
    } else if (other.dtype == dtype_t::f64) {
        double const& v = other.f64();
        if (new_dtype == dtype_t::f16) {
            return scalar_t(float16_t(v));
        } else if (new_dtype == dtype_t::f32) {
            return scalar_t(float(v));
        } else {
            throw std::runtime_error("scalar_t::convert: should not reach");
        }
    }
    throw std::runtime_error("scalar_t::convert: should not reach");
}

string scalar_t::str() const
{
    switch (dtype) {
        case dtype_t::f16:
            return write_with_ss(f16());
        case dtype_t::f32:
            return write_with_ss(f32());
        case dtype_t::f64:
            return write_with_ss(f64());
        case dtype_t::c64:
            return write_with_ss(c64());
    }
    throw std::runtime_error("should not reach");
}

float16_t& scalar_t::f16()
{
    return *reinterpret_cast<float16_t*>(data);
}
float& scalar_t::f32()
{
    return *reinterpret_cast<float*>(data);
}
double& scalar_t::f64()
{
    return *reinterpret_cast<double*>(data);
}
std::complex<float>& scalar_t::c64()
{
    return *reinterpret_cast<std::complex<float>*>(data);
}

float16_t const& scalar_t::f16() const
{
    return *reinterpret_cast<float16_t const*>(data);
}
float const& scalar_t::f32() const
{
    return *reinterpret_cast<float const*>(data);
}
double const& scalar_t::f64() const
{
    return *reinterpret_cast<double const*>(data);
}
std::complex<float> const& scalar_t::c64() const
{
    return *reinterpret_cast<std::complex<float> const*>(data);
}

void const* scalar_t::raw() const
{
    return reinterpret_cast<void const*>(data);
}

namespace scalar_ns
{

op_t op_t::make_constant(scalar_t value)
{
    return op_t{.op = constant{value}};
}

op_t op_t::make_hole(int arg, dtype_t dtype)
{
    return op_t{.op = hole{arg, dtype}};
}

op_t op_t::make_variable(string name, dtype_t dtype)
{
    return op_t{.op = variable{name, dtype}};
}

op_t op_t::make_ite(compare_t c)
{
    return op_t{.op = ite{c}};
}

string op_t::h_str(int arg, dtype_t dtype)
{
    return write_with_ss(make_hole(arg, dtype));
}

bool op_t::is_constant() const
{
    return std::holds_alternative<constant>(op);
}
bool op_t::is_hole() const
{
    return std::holds_alternative<hole>(op);
}
bool op_t::is_variable() const
{
    return std::holds_alternative<variable>(op);
}
bool op_t::is_add() const
{
    return std::holds_alternative<add>(op);
}
bool op_t::is_mul() const
{
    return std::holds_alternative<mul>(op);
}
bool op_t::is_exp() const
{
    return std::holds_alternative<exp>(op);
}
bool op_t::is_log() const
{
    return std::holds_alternative<log>(op);
}
bool op_t::is_power() const
{
    return std::holds_alternative<power>(op);
}
bool op_t::is_ite() const
{
    return std::holds_alternative<ite>(op);
}
bool op_t::is_convert() const
{
    return std::holds_alternative<convert>(op);
}
bool op_t::is_conj() const
{
    return std::holds_alternative<conj>(op);
}
bool op_t::is_real() const
{
    return std::holds_alternative<real>(op);
}
bool op_t::is_imag() const
{
    return std::holds_alternative<imag>(op);
}
bool op_t::is_cplex() const
{
    return std::holds_alternative<cplex>(op);
}

scalar_t op_t::get_constant() const
{
    return std::get<constant>(op).value;
}

op_t::variable op_t::get_variable() const
{
    return std::get<variable>(op);
}

int op_t::get_which_input() const
{
    return std::get<hole>(op).arg;
}

dtype_t op_t::get_hole_dtype() const
{
    return std::get<hole>(op).dtype;
}

op_t::hole op_t::get_hole() const
{
    return std::get<hole>(op);
}

double op_t::get_power() const
{
    return std::get<power>(op).to_the;
}

compare_t op_t::get_ite_compare() const
{
    return std::get<ite>(op).compare;
}

dtype_t op_t::get_convert() const
{
    return std::get<convert>(op).dtype;
}

int op_t::num_inputs() const
{
    if (is_constant() || is_hole() || is_variable()) {
        return 0;
    }
    if (is_power() || is_exp() || is_log() || is_convert() || is_conj() || is_real() || is_imag()) {
        return 1;
    }
    if (is_add() || is_mul() || is_cplex()) {
        return 2;
    }
    if (is_ite()) {
        return 4; // if (compare v0 v1) then v2 else v3
    }
    throw std::runtime_error("should not reach: num_inputs");
}

scalar_t op_t::eval(vector<scalar_t> const& xs) const
{
    if (xs.size() != num_inputs()) {
        throw std::runtime_error("invalid op_t::eval");
    }
    if (is_constant()) {
        return get_constant();
    }
    if (is_hole() || is_variable()) {
        throw std::runtime_error("cannot eval inputs or variables; no state here");
    }
    if (is_add()) {
        return _eval_add(xs[0], xs[1]);
    }
    if (is_mul()) {
        return _eval_mul(xs[0], xs[1]);
    }
    if (is_exp()) {
        return _eval_exp(xs[0]);
    }
    if (is_log()) {
        return _eval_log(xs[0]);
    }
    if (is_power()) {
        return _eval_power(get_power(), xs[0]);
    }
    if (is_ite()) {
        return _eval_ite(get_ite_compare(), xs[0], xs[1], xs[2], xs[3]);
    }
    if (is_convert()) {
        return _eval_convert(get_convert(), xs[0]);
    }
    if (is_conj()) {
        return _eval_conj(xs[0]);
    }
    if (is_real()) {
        return _eval_real(xs[0]);
    }
    if (is_imag()) {
        return _eval_imag(xs[0]);
    }
    if (is_cplex()) {
        return _eval_cplex(xs[0], xs[1]);
    }
    throw std::runtime_error("should not reach");
}

node_t node_t::make_constant(scalar_t value)
{
    return node_t{.op = op_t::make_constant(value), .dtype = value.dtype, .children = {}};
}

scalar_t node_t::eval(vector<scalar_t> const& inputs, map<string, scalar_t> const& variables) const
{
    if (op.is_hole()) {
        int which = op.get_which_input();
        if (which < 0 || which >= inputs.size()) {
            throw std::runtime_error("scalar node_t eval does not have the input");
        }
        scalar_t const& ret = inputs[which];
        if (ret.dtype != op.get_hole().dtype) {
            throw std::runtime_error("invalid dtype in inputs");
        }
        return ret;
    }
    if (op.is_variable()) {
        auto const& v = op.get_variable();
        auto        iter = variables.find(v.name);
        if (iter == variables.end()) {
            throw std::runtime_error("variable not in variable map");
        }
        scalar_t const& ret = iter->second;
        if (ret.dtype != v.dtype) {
            throw std::runtime_error("variable value provided has wrong dtype");
        }
        return ret;
    }

    vector<scalar_t> cs;
    cs.reserve(children.size());
    for (auto const& child : children) {
        cs.push_back(child.eval(inputs, variables));
    }

    return op.eval(cs);
}

scalar_t op_t::_eval_add(scalar_t lhs, scalar_t rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("_eval_add");
    }
    switch (lhs.dtype) {
        case dtype_t::f16:
            return scalar_t(lhs.f16() + rhs.f16());
        case dtype_t::f32:
            return scalar_t(lhs.f32() + rhs.f32());
        case dtype_t::f64:
            return scalar_t(lhs.f64() + rhs.f64());
        case dtype_t::c64:
            return scalar_t(lhs.c64() + rhs.c64());
    }
    throw std::runtime_error("should not reach");
}

scalar_t op_t::_eval_mul(scalar_t lhs, scalar_t rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("_eval_mul: dtypes do not match");
    }
    switch (lhs.dtype) {
        case dtype_t::f16:
            return scalar_t(lhs.f16() * rhs.f16());
        case dtype_t::f32:
            return scalar_t(lhs.f32() * rhs.f32());
        case dtype_t::f64:
            return scalar_t(lhs.f64() * rhs.f64());
        case dtype_t::c64:
            return scalar_t(lhs.c64() * rhs.c64());
    }
    throw std::runtime_error("should not reach");
}

scalar_t op_t::_eval_exp(scalar_t inn)
{
    if (inn.dtype == dtype_t::c64) {
        throw std::runtime_error("not implemented: exp complex"); // TODO
    }
    if (inn.dtype == dtype_t::f16) {
        return scalar_t(half_float::exp(inn.f16()));
    }
    if (inn.dtype == dtype_t::f32) {
        return scalar_t(std::exp(inn.f32()));
    }
    if (inn.dtype == dtype_t::f64) {
        return scalar_t(std::exp(inn.f64()));
    }
    throw std::runtime_error("_eval_exp: should not reach");
}

scalar_t op_t::_eval_log(scalar_t inn)
{
    if (inn.dtype == dtype_t::c64) {
        throw std::runtime_error("not implemented: log complex"); // TODO
    }
    if (inn.dtype == dtype_t::f16) {
        return scalar_t(half_float::log(inn.f16()));
    }
    if (inn.dtype == dtype_t::f32) {
        return scalar_t(std::log(inn.f32()));
    }
    if (inn.dtype == dtype_t::f64) {
        return scalar_t(std::log(inn.f64()));
    }
    throw std::runtime_error("_eval_log: should not reach");
}

scalar_t op_t::_eval_power(double to_the, scalar_t inn)
{
    if (inn.dtype == dtype_t::c64) {
        throw std::runtime_error("cannot power complex");
    }
    if (inn.dtype == dtype_t::f16) {
        return scalar_t(half_float::pow(inn.f16(), float16_t(to_the)));
    }
    if (inn.dtype == dtype_t::f32) {
        return scalar_t(std::pow(inn.f32(), float(to_the)));
    }
    if (inn.dtype == dtype_t::f64) {
        return scalar_t(std::pow(inn.f64(), to_the));
    }
    throw std::runtime_error("_eval_power: should not reach");
}

scalar_t op_t::_eval_ite(compare_t c, scalar_t lhs, scalar_t rhs, scalar_t if_t, scalar_t if_f)
{
    if (if_t.dtype != if_f.dtype) {
        throw std::runtime_error("eval_ite");
    }
    return _compare(c, lhs, rhs) ? if_t : if_f;
}

scalar_t op_t::_eval_convert(dtype_t dtype, scalar_t inn)
{
    if (inn.dtype == dtype_t::c64 || dtype == dtype_t::c64) {
        throw std::runtime_error("not converting between complex");
    }
    if (inn.dtype == dtype) {
        throw std::runtime_error("not allowing no ops");
    }
    return scalar_t::convert(inn, dtype);
}

scalar_t op_t::_eval_conj(scalar_t inn)
{
    if (inn.dtype != dtype_t::c64) {
        throw std::runtime_error("must have c64");
    }
    return scalar_t(std::conj(inn.c64()));
}

scalar_t op_t::_eval_real(scalar_t inn)
{
    if (inn.dtype != dtype_t::c64) {
        throw std::runtime_error("must have c64");
    }
    return scalar_t(inn.c64().real());
}

scalar_t op_t::_eval_imag(scalar_t inn)
{
    if (inn.dtype != dtype_t::c64) {
        throw std::runtime_error("must have c64");
    }
    return scalar_t(inn.c64().imag());
}

scalar_t op_t::_eval_cplex(scalar_t lhs, scalar_t rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("must have same input dtypes");
    }
    if (lhs.dtype != dtype_t::f32) {
        throw std::runtime_error("must have f32");
    }
    return scalar_t(std::complex<float>(lhs.f32(), rhs.f32()));
}

optional<dtype_t> op_t::_type_add(dtype_t lhs, dtype_t rhs)
{
    if (lhs == rhs) {
        return lhs;
    }
    return std::nullopt;
}

optional<dtype_t> op_t::_type_mul(dtype_t lhs, dtype_t rhs)
{
    if (lhs == rhs) {
        return lhs;
    }
    return std::nullopt;
}

optional<dtype_t> op_t::_type_exp(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return std::nullopt;
    }
    return inn;
}

optional<dtype_t> op_t::_type_log(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return std::nullopt;
    }
    return inn;
}

optional<dtype_t> op_t::_type_power(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return std::nullopt;
    }
    return inn;
}

optional<dtype_t> op_t::_type_ite(dtype_t lhs, dtype_t rhs, dtype_t if_true, dtype_t if_false)
{
    if (lhs != rhs) {
        return std::nullopt;
    }
    if (lhs == dtype_t::c64) {
        return std::nullopt;
    }
    if (if_true != if_false) {
        return std::nullopt;
    }
    return if_true;
}

optional<dtype_t> op_t::_type_convert(dtype_t inn, dtype_t out)
{
    if (inn == dtype_t::c64 || out == dtype_t::c64) {
        return std::nullopt;
    }
    if (inn == out) {
        return std::nullopt;
    }
    return out;
}

optional<dtype_t> op_t::_type_conj(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return inn;
    } else {
        return std::nullopt;
    }
}

optional<dtype_t> op_t::_type_real(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return dtype_t::f32;
    } else {
        return std::nullopt;
    }
}

optional<dtype_t> op_t::_type_imag(dtype_t inn)
{
    if (inn == dtype_t::c64) {
        return dtype_t::f32;
    } else {
        return std::nullopt;
    }
}

optional<dtype_t> op_t::_type_cplex(dtype_t lhs, dtype_t rhs)
{
    if (lhs != rhs) {
        return std::nullopt;
    }

    if (lhs == dtype_t::f32) {
        return dtype_t::c64;
    } else {
        return std::nullopt;
    }
}

optional<dtype_t> op_t::type_of(vector<dtype_t> inns) const
{
    if (is_constant()) {
        if (inns.size() != 0) {
            return std::nullopt;
        }
        return get_constant().dtype;
    } else if (is_hole()) {
        if (inns.size() != 0) {
            return std::nullopt;
        }
        return get_hole_dtype();
    } else if (is_variable()) {
        return get_variable().dtype;
    } else if (is_add()) {
        if (inns.size() != 2) {
            return std::nullopt;
        }
        return _type_add(inns[0], inns[1]);
    } else if (is_mul()) {
        if (inns.size() != 2) {
            return std::nullopt;
        }
        return _type_mul(inns[0], inns[1]);
    } else if (is_exp()) {
        if (inns.size() != 1) {
            return std::nullopt;
        }
        return _type_exp(inns[0]);
    } else if (is_log()) {
        if (inns.size() != 1) {
            return std::nullopt;
        }
        return _type_log(inns[0]);
    } else if (is_power()) {
        if (inns.size() != 1) {
            return std::nullopt;
        }
        return _type_power(inns[0]);
    } else if (is_ite()) {
        if (inns.size() != 4) {
            return std::nullopt;
        }
        return _type_ite(inns[0], inns[1], inns[2], inns[3]);
    } else if (is_convert()) {
        return _type_convert(inns[0], get_convert());
    } else if (is_conj()) {
        return _type_conj(inns[0]);
    } else if (is_real()) {
        return _type_real(inns[0]);
    } else if (is_imag()) {
        return _type_imag(inns[0]);
    } else if (is_cplex()) {
        return _type_cplex(inns[0], inns[1]);
    } else {
        throw std::runtime_error("type_of should not reach");
    }
}

template <typename T>
bool _compare_helper(compare_t c, T const& lhs, T const& rhs)
{
    if (c == compare_t::lt) {
        return lhs < rhs;
    } else if (c == compare_t::gt) {
        return lhs > rhs;
    } else if (c == compare_t::eq) {
        return lhs == rhs;
    } else if (c == compare_t::le) {
        return lhs <= rhs;
    } else if (c == compare_t::ge) {
        return lhs >= rhs;
    } else {
        throw std::runtime_error("should not reach");
    }
}

bool op_t::_compare(compare_t c, scalar_t lhs, scalar_t rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("_compare");
    }
    if (lhs.dtype == dtype_t::c64) {
        throw std::runtime_error("cannot compare complex");
    }
    switch (lhs.dtype) {
        case dtype_t::f16:
            return _compare_helper(c, lhs.f16(), rhs.f16());
        case dtype_t::f32:
            return _compare_helper(c, lhs.f32(), rhs.f32());
        case dtype_t::f64:
            return _compare_helper(c, lhs.f64(), rhs.f64());
    }
    throw std::runtime_error("should not reach");
}

node_t node_t::derivative(int arg) const
{
    if (op.is_constant() || op.is_variable()) {
        // TODO: are we making assumptions on variable?
        return make_constant(scalar_t::zero(dtype));
    } else if (op.is_hole()) {
        if (arg == op.get_which_input()) {
            if (dtype_is_complex(dtype) && dtype != dtype_t::c64) {
                throw std::runtime_error("impl more complex dtypes");
            }
            scalar_t one = dtype_is_complex(dtype) ? scalar_t(std::complex<float>(1.0, 0.0))
                                                   : scalar_t::one(dtype);
            return make_constant(one);
        } else {
            return make_constant(scalar_t::zero(dtype));
        }
    } else if (op.is_add()) {
        node_t const& lhs = children[0];
        node_t const& rhs = children[1];

        node_t deri_lhs = lhs.derivative(arg);
        node_t deri_rhs = rhs.derivative(arg);

        return node_t{
            .op = parse_with_ss<op_t>("+"), .dtype = dtype, .children = {deri_lhs, deri_rhs}};
    } else if (op.is_mul()) {
        node_t const& lhs = children[0];
        node_t const& rhs = children[1];

        node_t deri_lhs = lhs.derivative(arg);
        node_t deri_rhs = rhs.derivative(arg);

        string s_lhs = write_with_ss(lhs);
        string s_rhs = write_with_ss(rhs);

        string s_deri_lhs = write_with_ss(deri_lhs);
        string s_deri_rhs = write_with_ss(deri_rhs);

        // Gradient of d(L(x)R(x))/dx = L' R  + L R'

        string term_lhs = "*[" + s_deri_lhs + "," + s_rhs + "]";
        string term_rhs = "*[" + s_lhs + "," + s_deri_rhs + "]";

        return parse_with_ss<node_t>("+[" + term_lhs + "," + term_rhs + "]");
    } else if (op.is_exp()) {
        // e^{f(x)} => e^{f(x)} * f'(x)
        //             --------   -----
        //             this       deri_inn
        node_t const& inn = children[0];

        node_t deri_inn = inn.derivative(arg);

        string s_deri_inn = write_with_ss(deri_inn);

        string s_this = write_with_ss(*this);

        return parse_with_ss<node_t>("*[" + s_this + "," + s_deri_inn + "]");
    } else if (op.is_log()) {
        // ln(f(x)) => 1/f(x) * f'(x)
        node_t const& inn = children[0];

        node_t deri_inn = inn.derivative(arg);

        string s_deri_inn = write_with_ss(deri_inn);

        string s_this = write_with_ss(*this);
        string s_inv_this = "power{-1.0}[" + s_this + "]";

        return parse_with_ss<node_t>("*[" + s_inv_this + "," + s_deri_inn + "]");
    } else if (op.is_power()) {
        // I(x)^i => i * { (I(x) ^{i-1}) * I'(x) }
        //           A     B               C

        double i = op.get_power();

        if (i == 0.0) {
            return make_constant(scalar_t::zero(dtype));
        }

        node_t const& inn = children[0];
        node_t        deri_inn = inn.derivative(arg);

        if (i == 1.0) {
            return deri_inn;
        }

        scalar_t scalar_i = scalar_t::convert(scalar_t(i), deri_inn.dtype);

        string s_inn = write_with_ss(inn);

        string A = "constant{" + write_with_ss(scalar_i) + "}";
        string B = "power{" + write_with_ss(i - 1) + "}[" + s_inn + "]";
        string C = write_with_ss(deri_inn);

        string BC = "*[" + B + "," + C + "]";
        string ABC = "*[" + A + "," + BC + "]";

        return parse_with_ss<node_t>(ABC);
    } else if (op.is_ite()) {
        // compare(x0, x1) ? x2  : x3  has a derivative of
        // compare(x0, x1) ? x2' : x3'
        string s0 = write_with_ss(children[0]);
        string s1 = write_with_ss(children[1]);
        string deri_s2 = write_with_ss(children[2].derivative(arg));
        string deri_s3 = write_with_ss(children[3].derivative(arg));

        string compare = write_with_ss(op.get_ite_compare());

        string ret = "ite_" + compare + "[" + s0 + "," + s1 + "," + deri_s2 + "," + deri_s3 + "]";

        return parse_with_ss<node_t>(ret);
    } else if (op.is_convert()) {
        // convert(f(x)) => convert(f'(x))
        node_t const& inn = children[0];
        node_t        deri_inn = inn.derivative(arg);
        return node_t{.op = op, .dtype = dtype, .children = vector<node_t>{deri_inn}};
    } else if (op.is_conj()) {
        throw std::runtime_error("conjugation is not holomorphic");
    } else if (op.is_real()) {
        throw std::runtime_error("real projection is not holomorphic");
    } else if (op.is_imag()) {
        throw std::runtime_error("complex projection is not holomorphic");
    } else if (op.is_cplex()) {
        throw std::runtime_error("complex(x,y) is not holomorphic");
    } else {
        throw std::runtime_error("missing derivative case");
    }
}

node_t node_t::wirtinger_derivative(int arg, bool conjugate) const
{
    // Consider s(x) = h(g(x)). The chain rule gives
    //   ds/d(z*) = dh/dg * dg/d(z*) + dh/d(g*) * (dg/dz)*
    //   ds/dz    = dh/dg * dg/dz    + dh/d(g*) * (dg/d(z*))*
    // When h is holomorphic, dh/d(g*) will be zero, so
    //   ds/d(z*) = dh/dg * dg/d(z*)
    //   ds/dz    = dh/dg * dg/dz
    // When h maps to real, dh/d(g*) = (dh/dg)*
    // When h maps real to real, (dh/dg) = (dh/dg)*
    //   ds/d(z*) = dh/dg * (dg/d(z*) + (dg/dz)*)
    //   ds/dz    = dh/dg * (dg/dz    + (dg/d(z*))*

    // dh/dg    = 1/2
    // dh/d(g*) = 1/2
    // ds/d(z*) = 1/2 * dg/d(z*) + 1/2 * (dg/dz)*
    // ds/dz    = 1/2 * dg/dz    + 1/2 * (dg/d(z*))*
    auto compute_real = [&](int which_child) {
        if (dtype != dtype_t::c64) {
            throw std::runtime_error("can't get constant: wirt deriv");
        }
        string dgdz = write_with_ss(children[which_child].wirtinger_derivative(arg, false));
        string dgdzs = write_with_ss(children[which_child].wirtinger_derivative(arg, true));
        string half = write_with_ss(make_constant(scalar_t(std::complex<float>(0.5, 0.0))));
        string ret = "+[" + (conjugate ? dgdzs : dgdz) + "," + "conj[" +
                     (conjugate ? dgdz : dgdzs) + "]" + "]";
        return parse_with_ss<node_t>("*[" + half + "," + ret + "]");
    };

    auto compute_imag = [&](int which_child) {
        // dh/d(g*) =  i/2
        // dh/dg    = -i/2
        // ds/d(z*) = -i/2 * dg/d(z*) + i/2 * (dg/dz)*
        // ds/dz    = -i/2 * dg/dz    + i/2 * (dg/d(z*))*
        string nhalfi = write_with_ss(make_constant(scalar_t(std::complex<float>(0.0, -0.5))));
        string halfi = write_with_ss(make_constant(scalar_t(std::complex<float>(0.0, 0.5))));
        string dgdz = write_with_ss(children[which_child].wirtinger_derivative(arg, false));
        string dgdzs = write_with_ss(children[which_child].wirtinger_derivative(arg, true));
        string lhs = "*[" + nhalfi + "," + (conjugate ? dgdzs : dgdz) + "]";
        string rhs = "*[" + halfi + ",conj[" + (conjugate ? dgdz : dgdzs) + "]]";
        return parse_with_ss<node_t>("+[" + lhs + "," + rhs + "]");
    };

    if (op.is_constant() || op.is_variable()) {
        // TODO: are we making assumptionso on variable?
        return make_constant(scalar_t::zero(dtype));
    } else if (op.is_hole()) {
        if (conjugate) {
            return make_constant(scalar_t::zero(dtype));
        } else {
            scalar_t one = dtype_is_complex(dtype) ? scalar_t(std::complex<float>(1.0, 0.0))
                                                   : scalar_t::one(dtype);
            return make_constant(one);
        }
    } else if (op.is_add()) {
        if (arg != 0 && arg != 1) {
            return make_constant(scalar_t::zero(dtype));
        }
        // is holomorphic and dh/dg = 1
        return children[arg].wirtinger_derivative(arg, conjugate);
    } else if (op.is_mul()) {
        if (arg != 0 && arg != 1) {
            return make_constant(scalar_t::zero(dtype));
        }
        // is holomorphic and dh/dg = the other arg
        string dh_dg = arg == 0 ? write_with_ss(children[1]) : write_with_ss(children[0]);
        string dg_dzz = write_with_ss(children[arg].wirtinger_derivative(arg, conjugate));
        return parse_with_ss<node_t>("*[" + dh_dg + "," + dg_dzz + "]");
    } else if (op.is_exp()) {
        // is holomorphic and dh/dg = e^g(x)
        if (arg != 0) {
            return make_constant(scalar_t::zero(dtype));
        }
        string dh_dg = "exp[" + write_with_ss(children[0]) + "]";
        string dg_dzz = write_with_ss(children[0].wirtinger_derivative(arg, conjugate));
        return parse_with_ss<node_t>("*[" + dh_dg + "," + dg_dzz + "]");
    } else if (op.is_log()) {
        throw std::runtime_error("not implemented");
    } else if (op.is_power()) {
        if (arg != 0) {
            return make_constant(scalar_t::zero(dtype));
        }
        // TODO
        // This may be holomorphic; it may require logarithm
        throw std::runtime_error("wirtinger deriv for power not implemented");
    } else if (op.is_ite()) {
        // TODO
        throw std::runtime_error("not implemented");
    } else if (op.is_convert()) {
        // TODO
        throw std::runtime_error("not implemented");
    } else if (op.is_conj()) {
        // dh/dg    = 0
        // dh/d(g*) = 1
        // ds/d(z*) = (dg/dz)*
        // ds/dz    = (dg/d(z*))*
        if (conjugate) {
            string dgdz = write_with_ss(children[0].wirtinger_derivative(arg, false));
            return parse_with_ss<node_t>("conj[" + dgdz + "]");
        } else {
            string dgdzs = write_with_ss(children[0].wirtinger_derivative(arg, true));
            return parse_with_ss<node_t>("conj[" + dgdzs + "]");
        }
    } else if (op.is_real()) {
        return compute_real(0);
    } else if (op.is_imag()) {
        return compute_imag(0);
    } else if (op.is_cplex()) {
        if (arg != 0 && arg != 1) {
            return make_constant(scalar_t::zero(dtype));
        }
        if (arg == 0) {
            return compute_real(0);
        } else {
            return compute_imag(1);
        }
    } else {
        throw std::runtime_error("missing wirtinger derivative case");
    }
}

node_t node_t::simplify() const
{
    node_t r0 = simplify_once();
    if (r0 == *this) {
        return r0;
    }

    node_t r1 = r0.simplify_once();
    while (true) {
        if (r0 == r1) {
            return r1;
        }
        r0 = r1.simplify_once();
        std::swap(r0, r1);
    }
}

node_t node_t::simplify_once() const
{
    if (op.is_hole() || op.is_constant() || op.is_variable()) {
        return *this;
    }

    // Case: Has no inputs and no variables (and therefore should be a constant)
    {
        set<int> holes;
        which_inputs(holes);
        set<string> variables;
        which_variables(variables);
        if (holes.size() == 0 && variables.size() == 0) {
            scalar_t val = eval({}, {});
            string   constant = ("constant{" + write_with_ss(val) + "}");
            return parse_with_ss<node_t>(constant);
        }
    }

    vector<node_t> new_children = vector_from_each_method(children, node_t, simplify);

    {
        node_t self{.op = op, .dtype = dtype, .children = new_children};

        auto maybe_normalize_order = self.normalize_order();
        if (maybe_normalize_order) {
            return maybe_normalize_order.value();
        }
    }

    // Case: Add
    if (op.is_add()) {
        // Check for 0 + x or x + 0
        node_t& lhs = new_children[0];
        node_t& rhs = new_children[1];
        if (lhs.op.is_constant() && lhs.op.get_constant() == scalar_t::zero(dtype)) {
            return rhs;
        }
        if (rhs.op.is_constant() && rhs.op.get_constant() == scalar_t::zero(dtype)) {
            return lhs;
        }

        // Check for inf + x or x + inf
        //       or -inf + x or x + -inf
        if (!dtype_is_complex(dtype)) {
            if (lhs.op.is_constant() && (lhs.op.get_constant() == scalar_t::inf(dtype) ||
                                         lhs.op.get_constant() == scalar_t::negative_inf(dtype))) {
                return lhs;
            }
            if (rhs.op.is_constant() && (rhs.op.get_constant() == scalar_t::inf(dtype) ||
                                         rhs.op.get_constant() == scalar_t::negative_inf(dtype))) {
                return rhs;
            }
        }
    }

    // TODO: also cover x^i * x^j
    //       and        x   * x
    //       and        x   * x^i
    //       and        x^i * x

    // Case: Mul
    if (op.is_mul()) {
        // Check for 0*x or x*0
        node_t& lhs = new_children[0];
        node_t& rhs = new_children[1];
        if ((lhs.op.is_constant() && lhs.op.get_constant() == scalar_t::zero(dtype)) ||
            (rhs.op.is_constant() && rhs.op.get_constant() == scalar_t::zero(dtype))) {
            return make_constant(scalar_t::zero(dtype));
        }

        // Check for (constant * (constant * hole))
        //           (c0 * (c1 * h))
        if (lhs.op.is_constant() && rhs.op.is_mul()) {
            node_t& c0 = lhs;
            node_t& c1 = rhs.children[0];
            node_t& h = rhs.children[1];
            if (c1.op.is_constant() && h.op.is_hole()) {
                scalar_t val = scalarop_t::make_mul(c0.dtype).eval(
                    {c0.op.get_constant(), c1.op.get_constant()});

                return node_t{.op = op, // op is just a mul
                              .dtype = c0.dtype,
                              .children = {node_t::make_constant(val), h}};
            }
        }

        // Check for 1*x or x*1
        scalar_t one = dtype_is_complex(dtype) ? scalar_t(std::complex<float>(1.0, 0.0))
                                               : scalar_t::one(dtype);
        if (lhs.op.is_constant() && lhs.op.get_constant() == one) {
            return rhs;
        }
        if (rhs.op.is_constant() && rhs.op.get_constant() == one) {
            return lhs;
        }

        // Check for (constant * (constant * hole))
        //           (c0 * (c1 * h))
        if (lhs.op.is_constant() && rhs.op.is_mul()) {
            node_t& c0 = lhs;
            node_t& c1 = rhs.children[0];
            node_t& h = rhs.children[1];
            if (c1.op.is_constant() && h.op.is_hole()) {
                scalar_t val = scalarop_t::make_mul(c0.dtype).eval(
                    {c0.op.get_constant(), c1.op.get_constant()});

                return node_t{.op = op, // op is just a mul
                              .dtype = c0.dtype,
                              .children = {node_t::make_constant(val), h}};
            }
        }

        // TODO: Check for 0.5*(x+x)

        // TODO: convert for x*x -> x^2 ?
    }

    // Case: Exp
    // e^0 is already covered since there'd be no inputs

    // Case: Power
    if (op.is_power()) {
        // check for x^0 or x^1
        double i = op.get_power();
        if (i == 0.0) {
            return make_constant(scalar_t::one(dtype));
        }
        if (i == 1.0) {
            node_t& inn = new_children[0];
            return inn;
        }
    }

    // Case: Ite
    if (op.is_ite()) {
        // check for ite z z x y
        // and       ite x y z z
        node_t& s0 = new_children[0];
        node_t& s1 = new_children[1];
        node_t& s2 = new_children[2];
        node_t& s3 = new_children[3];
        if (s0 == s1) {
            compare_t c = op.get_ite_compare();
            if (c == compare_t::eq) {
                return s2;
            }
        }
        if (s2 == s3) {
            return s2;
        }
    }

    // Case: Conjugate
    if (op.is_conj()) {
        node_t& inn = new_children[0];
        if (inn.op.is_conj()) {
            // Conjugate of conjugate is a no op
            return inn.children[0];
        }
    }

    // TODO?: conj(x) + x = 2real(x)
    //        conj(x) - x = 2imag(x)

    return node_t{.op = op, .dtype = dtype, .children = new_children};
}

node_t node_t::replace_variables(map<string, scalar_t> const& vars) const
{
    if (op.is_variable()) {
        auto const& v = op.get_variable();
        auto        iter = vars.find(v.name);
        if (iter == vars.end()) {
            throw std::runtime_error("missing variable");
        }
        scalar_t const& val = iter->second;
        if (val.dtype != v.dtype) {
            throw std::runtime_error("variable dtype provided is incorrect");
        }
        return make_constant(val);
    }

    vector<node_t> new_children;
    new_children.reserve(children.size());
    for (auto const& child : children) {
        new_children.push_back(child.replace_variables(vars));
    }

    return node_t{.op = op, .dtype = dtype, .children = new_children};
}

optional<node_t> node_t::normalize_order() const
{
    auto is_ordered = [](node_t const& lhs, node_t const& rhs) {
        int l = lhs.max_hole();
        int r = rhs.max_hole();
        if (r < l) {
            return false;
        }
        if (l == r && lhs != rhs) {
            string sl = write_with_ss(l);
            string sr = write_with_ss(r);
            if (sr < sl) {
                return false;
            }
        }
        return true;
    };

    if (op.is_add() || op.is_mul()) {
        node_t const& lhs = children[0];
        node_t const& rhs = children[1];
        if (!is_ordered(lhs, rhs)) {
            return node_t{.op = op, .dtype = dtype, .children = {rhs, lhs}};
        }

        return std::nullopt;
    }

    if (op.is_ite()) {
        bool fix_co = !is_ordered(children[0], children[1]);
        bool fix_tt = !is_ordered(children[2], children[3]);
        if (fix_co || fix_tt) {
            vector<node_t> new_children = children;

            op_t new_op = op;
            if (fix_co) {
                new_op = op_t::make_ite(compare_flip(op.get_ite_compare()));
                std::swap(new_children[0], new_children[1]);
            }

            if (fix_tt) {
                std::swap(new_children[2], new_children[3]);
            }

            return node_t{.op = new_op, .dtype = dtype, .children = new_children};
        }

        return std::nullopt;
    }

    return std::nullopt;
}

string node_t::to_cppstr(std::function<string(int)> w) const
{
    if (op.is_constant()) {
        return write_with_ss(op.get_constant());
    } else if (op.is_hole()) {
        return w(op.get_which_input());
    } else if (op.is_variable()) {
        // not really cpp str, but oh well
        return op.get_variable().name;
    } else if (op.is_add()) {
        auto lhs = children[0].to_cppstr(w);
        auto rhs = children[1].to_cppstr(w);
        return "(" + lhs + "+" + rhs + ")";
    } else if (op.is_mul()) {
        auto lhs = children[0].to_cppstr(w);
        auto rhs = children[1].to_cppstr(w);
        return "(" + lhs + "*" + rhs + ")";
    } else if (op.is_exp()) {
        auto inn = children[0].to_cppstr(w);
        return "_exp(" + inn + ")";
    } else if (op.is_log()) {
        auto inn = children[0].to_cppstr(w);
        return "_log(" + inn + ")";
    } else if (op.is_power()) {
        auto inn = children[0].to_cppstr(w);
        return "_pow(" + inn + "," + write_with_ss(op.get_power()) + ")";
    } else if (op.is_ite()) {
        auto i0 = children[0].to_cppstr(w);
        auto i1 = children[1].to_cppstr(w);
        auto i2 = children[2].to_cppstr(w);
        auto i3 = children[3].to_cppstr(w);
        auto c = write_with_ss(op.get_ite_compare());
        return "(" + i0 + c + i1 + "?" + i2 + ":" + i3 + ")";
    } else if (op.is_convert()) {
        auto inn = children[0].to_cppstr(w);
        switch (op.get_convert()) {
            case dtype_t::f16:
                return "float16_t(" + inn + ")";
            case dtype_t::f32:
                return "float(" + inn + ")";
            case dtype_t::f64:
                return "double(" + inn + ")";
        }
        throw std::runtime_error("should not reach");
    } else if (op.is_conj()) {
        auto inn = children[0].to_cppstr(w);
        return "_conj(" + inn + ")";
    } else if (op.is_real()) {
        auto inn = children[0].to_cppstr(w);
        return "_real(" + inn + ")";
    } else if (op.is_imag()) {
        auto inn = children[0].to_cppstr(w);
        return "_imag(" + inn + ")";
    } else if (op.is_cplex()) {
        auto lhs = children[0].to_cppstr(w);
        auto rhs = children[1].to_cppstr(w);
        return "_complex(" + lhs + "," + rhs + ")";
    } else {
        throw std::runtime_error("to_cppstr: should not reach");
    }
}

template <typename T>
uint64_t push_into_bytes(vector<uint8_t>& bytes, T const& val)
{
    auto offset = bytes.size();
    bytes.resize(offset + sizeof(T));
    T& value = *((T*)(bytes.data() + offset));
    value = val;
    return offset;
}

string access_data_str(string type, string data, uint64_t offset)
{
    // (*((float*)(data + offset)))
    return "(*((" + type + "*)(" + data + "+" + std::to_string(offset) + ")))";
}

static vector<tuple<std::function<string(string const&)>, scalarop_t>> _unary_matches(dtype_t dtype)
{
    using make_t = std::function<string(string const&)>;
    vector<tuple<make_t, scalarop_t>> ret;

    make_t f_sqrt = [](string const& inn) { return "_sqrt(" + inn + ")"; };
    make_t f_square = [](string const& inn) { return "_square(" + inn + ")"; };
    make_t f_invsqrt = [](string const& inn) { return "_invsqrt(" + inn + ")"; };

    ret.emplace_back(f_sqrt, scalarop_t::make_sqrt(dtype));
    ret.emplace_back(f_square, scalarop_t::make_square(dtype));
    ret.emplace_back(f_invsqrt, scalarop_t::make_inverse_sqrt(dtype));

    return ret;
}

static vector<tuple<std::function<string(string const&, string const&)>, scalarop_t>>
_binary_matches(dtype_t dtype)
{
    using make_t = std::function<string(string const&, string const&)>;
    vector<tuple<make_t, scalarop_t>> ret;

    make_t f_sub = [](string const& lhs, string const& rhs) { return "(" + lhs + "-" + rhs + ")"; };
    make_t f_div = [](string const& lhs, string const& rhs) { return "(" + lhs + "/" + rhs + ")"; };
    make_t f_max = [](string const& lhs, string const& rhs) {
        return "_max(" + lhs + "," + rhs + ")";
    };
    make_t f_min = [](string const& lhs, string const& rhs) {
        return "_min(" + lhs + "," + rhs + ")";
    };

    ret.emplace_back(f_sub, scalarop_t::make_sub(dtype));
    ret.emplace_back(f_div, scalarop_t::make_div(dtype));
    ret.emplace_back(f_max, scalarop_t::make_max(dtype));
    ret.emplace_back(f_min, scalarop_t::make_min(dtype));

    return ret;
}

string node_t::to_cpp_bytes(vector<uint8_t>& bytes) const
{
    // Some special cases we should match:
    //   sqrt
    //   square
    //   subtraction
    //   division
    //   max
    //   min

    if (dtype_is_real(dtype)) {
        for (auto const& [make_str, skeleton] : _unary_matches(dtype)) {
            auto maybe = _pop_match(skeleton.get_node(), this);
            if (maybe) {
                map<int, node_t const*> const& cs = maybe.value();
                string                         inn = cs.at(0)->to_cpp_bytes(bytes);
                return make_str(inn);
            }
        }

        for (auto const& [make_str, skeleton] : _binary_matches(dtype)) {
            auto maybe = _pop_match(skeleton.get_node(), this);
            if (maybe) {
                map<int, node_t const*> const& cs = maybe.value();
                string                         lhs = cs.at(0)->to_cpp_bytes(bytes);
                string                         rhs = cs.at(1)->to_cpp_bytes(bytes);
                return make_str(lhs, rhs);
            }
        }
    } else {
        // TODO: implement complex cases
    }

    if (op.is_constant()) {
        auto const& v = op.get_constant();
        if (v.dtype == dtype_t::f32) {
            auto offset = push_into_bytes(bytes, v.f32());
            return access_data_str("float", "d", offset);
        }
        if (v.dtype == dtype_t::f64) {
            auto offset = push_into_bytes(bytes, v.f64());
            return access_data_str("double", "d", offset);
        }
        if (v.dtype == dtype_t::f16) {
            auto offset = push_into_bytes(bytes, v.f16());
            return access_data_str("float16_t", "d", offset);
        }
        if (v.dtype == dtype_t::c64) {
            auto offset = push_into_bytes(bytes, v.c64());
            return access_data_str("std::complex<float,float>", "d", offset);
        }
        throw std::runtime_error("should not reach: to_cpp_bytes");
    } else if (op.is_hole()) {
        string which = std::to_string(op.get_which_input());
        return "x" + which + "[i" + which + "]";
    } else if (op.is_variable()) {
        throw std::runtime_error("no cpp bytes for variable");
    } else if (op.is_add()) {
        auto lhs = children[0].to_cpp_bytes(bytes);
        auto rhs = children[1].to_cpp_bytes(bytes);
        return "(" + lhs + "+" + rhs + ")";
    } else if (op.is_mul()) {
        auto lhs = children[0].to_cpp_bytes(bytes);
        auto rhs = children[1].to_cpp_bytes(bytes);
        return "(" + lhs + "*" + rhs + ")";
    } else if (op.is_exp()) {
        auto inn = children[0].to_cpp_bytes(bytes);
        return "_exp(" + inn + ")";
    } else if (op.is_log()) {
        auto inn = children[0].to_cpp_bytes(bytes);
        return "_log(" + inn + ")";
    } else if (op.is_power()) {
        auto inn = children[0].to_cpp_bytes(bytes);
        auto offset = push_into_bytes(bytes, op.get_power());
        auto power = access_data_str("double", "d", offset);
        return "_pow(" + inn + "," + power + ")";
    } else if (op.is_ite()) {
        auto i0 = children[0].to_cpp_bytes(bytes);
        auto i1 = children[1].to_cpp_bytes(bytes);
        auto i2 = children[2].to_cpp_bytes(bytes);
        auto i3 = children[3].to_cpp_bytes(bytes);
        auto c = write_with_ss(op.get_ite_compare());
        return "(" + i0 + c + i1 + "?" + i2 + ":" + i3 + ")";
    } else if (op.is_convert()) {
        auto inn = children[0].to_cpp_bytes(bytes);
        switch (op.get_convert()) {
            case dtype_t::f16:
                return "float16_t(" + inn + ")";
            case dtype_t::f32:
                return "float(" + inn + ")";
            case dtype_t::f64:
                return "double(" + inn + ")";
        }
        throw std::runtime_error("should not reach");
    } else if (op.is_conj()) {
        auto i0 = children[0].to_cpp_bytes(bytes);
        return "_conj(" + i0 + ")";
    } else if (op.is_real()) {
        auto i0 = children[0].to_cpp_bytes(bytes);
        return "_real(" + i0 + ")";
    } else if (op.is_imag()) {
        auto i0 = children[0].to_cpp_bytes(bytes);
        return "_imag(" + i0 + ")";
    } else if (op.is_cplex()) {
        auto i0 = children[0].to_cpp_bytes(bytes);
        auto i1 = children[1].to_cpp_bytes(bytes);
        return "_make_complex(" + i0 + "," + i1 + ")";
    } else {
        throw std::runtime_error("to_cpp_bytes: should not reach");
    }
}

void node_t::which_inputs(set<int>& items) const
{
    if (op.is_hole()) {
        items.insert(op.get_which_input());
    }
    for (auto const& child : children) {
        child.which_inputs(items);
    }
}

void node_t::which_variables(set<string>& items) const
{
    if (op.is_variable()) {
        items.insert(op.get_variable().name);
    }
    for (auto const& child : children) {
        child.which_variables(items);
    }
}

int node_t::max_hole() const
{
    if (op.is_hole()) {
        return op.get_which_input();
    }

    int ret = -1;
    for (auto const& child : children) {
        ret = std::max(ret, child.max_hole());
    }
    return ret;
}

int node_t::num_inputs() const
{
    return max_hole() + 1;
}

void node_t::increment_holes(int incr)
{
    if (op.is_hole()) {
        int& arg = std::get<op_t::hole>(op.op).arg;
        arg += incr;
    } else {
        for (auto& child : children) {
            child.increment_holes(incr);
        }
    }
}

void node_t::remap_holes(map<int, int> const& fmap)
{
    if (op.is_hole()) {
        int& arg = std::get<op_t::hole>(op.op).arg;
        arg = fmap.at(arg);
    } else {
        for (auto& child : children) {
            child.remap_holes(fmap);
        }
    }
}

void node_t::replace_at_holes(vector<node_t> const& replace_nodes)
{
    if (op.is_hole()) {
        int arg = std::get<op_t::hole>(op.op).arg;
        *this = replace_nodes[arg];
    } else {
        for (auto& child : children) {
            child.replace_at_holes(replace_nodes);
        }
    }
}

map<int, dtype_t> node_t::hole_types() const
{
    map<int, dtype_t> ret;
    _hole_types(ret);
    return ret;
}

bool node_t::type_check() const
{
    for (node_t const& child : children) {
        if (!child.type_check()) {
            return false;
        }
    }
    optional<dtype_t> maybe_dtype = op.type_of(vector_from_each_member(children, dtype_t, dtype));
    if (maybe_dtype) {
        return maybe_dtype.value() == dtype;
    } else {
        return false;
    }
}

void node_t::_hole_types(map<int, dtype_t>& ret) const
{
    if (op.is_hole()) {
        auto hole = op.get_hole();
        auto iter = ret.find(hole.arg);
        if (iter == ret.end()) {
            ret.insert({hole.arg, hole.dtype});
        } else {
            if (hole.dtype != iter->second) {
                throw std::runtime_error(
                    "This node_t has same arg holes with different "
                    "dtypes");
            }
        }
    } else {
        for (auto const& child : children) {
            child._hole_types(ret);
        }
    }
}

static bool merge_into_nodeconst_map(map<int, node_t const*>&       ret,
                                     map<int, node_t const*> const& values)
{
    for (auto const& [key, value] : values) {
        auto [iter, did_insert] = ret.insert({key, value});
        if (did_insert) {
            // Case 1: we have inserted a new key, value pair into ret
        } else {
            // Case 2: we did not insert a new key, value pair into ret.
            // It better be the case that ret[key] == value !
            node_t const& from_ret = *(iter->second);
            node_t const& from_value = *value;
            if (from_ret != from_value) {
                return false;
            }
        }
    }
    return true;
}

// Consider relu == ite compare(zero, hole0) zero hole0
// Then the recursion will hit hole0 multiple times and the
// maps will have to be checked.
//
// Consider lambda x: x + x  (hole0 + hole0)
// Here, we will have to merge {arg0, node_from_x}
optional<map<int, node_t const*>> _pop_match(node_t const* skeleton, node_t const* node)
{
    if (skeleton->dtype != node->dtype) {
        return std::nullopt;
    }

    // This is the base case of the recursion
    if (skeleton->op.is_hole()) {
        int which_arg = skeleton->op.get_which_input();
        return map<int, node_t const*>{{which_arg, node}};
    }

    if (skeleton->op != node->op) {
        return std::nullopt;
    }

    // now recurse!

    int num_children = skeleton->children.size();
    if (num_children != node->children.size()) {
        throw std::runtime_error("should not happen: same op, different number of children");
    }
    map<int, node_t const*> ret;
    for (int which = 0; which != num_children; ++which) {
        auto maybe = _pop_match(&skeleton->children[which], &node->children[which]);
        if (!maybe) {
            return std::nullopt;
        }
        bool success = merge_into_nodeconst_map(ret, maybe.value());
        if (!success) {
            return std::nullopt;
        }
    }
    return ret;
}

} // namespace scalar_ns

static dtype_t& _default_dtype()
{
    static dtype_t d = dtype_t::f32;
    return d;
}

dtype_t const& default_dtype()
{
    return _default_dtype();
}

void set_default_dtype(dtype_t new_dtype)
{
    _default_dtype() = new_dtype;
}

static dtype_t& _default_complex_dtype()
{
    static dtype_t d = dtype_t::c64;
    return d;
}

dtype_t const& default_complex_dtype()
{
    return _default_complex_dtype();
}

void set_default_complex_dtype(dtype_t new_dtype)
{
    if (!dtype_is_complex(new_dtype)) {
        throw std::runtime_error("cannot set default complex dtype to real dtype");
    }
    _default_complex_dtype() = new_dtype;
}

scalarop_t::scalarop_t() {}

scalarop_t::scalarop_t(scalar_ns::node_t const& n)
    : node(n.simplify()), arg_types(node.hole_types())
{
    if (!node.type_check()) {
        throw std::runtime_error("scalarop did not typecheck: " + write_with_ss(*this));
    }
}

scalar_t scalarop_t::eval() const
{
    return eval(vector<scalar_t>{}, {});
}

scalar_t scalarop_t::eval(scalar_t const& x0) const
{
    return eval({x0}, {});
}

scalar_t scalarop_t::eval(vector<scalar_t> const&      inputs,
                          map<string, scalar_t> const& variables) const
{
    return node.eval(inputs, variables);
}

scalarop_t scalarop_t::derivative(int arg) const
{
    return scalarop_t(node.derivative(arg));
}

scalarop_t scalarop_t::wirtinger_derivative(int arg, bool conjugate) const
{
    return scalarop_t(node.wirtinger_derivative(arg, conjugate));
}

scalarop_t scalarop_t::simplify() const
{
    return scalarop_t(node.simplify());
}

scalarop_t scalarop_t::replace_variables(map<string, scalar_t> const& vars) const
{
    return scalarop_t(node.replace_variables(vars));
}

optional<dtype_t> scalarop_t::inn_dtype(int arg) const
{
    auto iter = arg_types.find(arg);
    if (iter == arg_types.end()) {
        return std::nullopt;
    } else {
        return iter->second;
    }
}

void scalarop_t::remap_inputs(map<int, int> const& remap)
{
    node.remap_holes(remap);
    arg_types = node.hole_types();
}

set<int> scalarop_t::which_inputs() const
{
    set<int> ret;
    node.which_inputs(ret);
    return ret;
}

set<string> scalarop_t::which_variables() const
{
    set<string> ret;
    node.which_variables(ret);
    return ret;
}

bool scalarop_t::has_variables() const
{
    // Could be faster but just dispatching to which_variables..
    return which_variables().size() > 0;
}

int scalarop_t::num_inputs() const
{
    return node.num_inputs();
}

int scalarop_t::num_variables() const
{
    // Could be faster but just dispatching to which_variables..
    return which_variables().size();
}

bool scalarop_t::is_constant() const
{
    return num_inputs() == 0;
}

bool scalarop_t::is_unary() const
{
    return num_inputs() == 1;
}

bool scalarop_t::is_binary() const
{
    return num_inputs() == 2;
}

bool scalarop_t::is_castable() const
{
    return is_add() || is_mul() || is_min() || is_max();
}

bool scalarop_t::is_identity() const
{
    return *this == make_identity(node.dtype);
}

bool scalarop_t::is_add() const
{
    return *this == make_add(node.dtype);
}

bool scalarop_t::is_mul() const
{
    return *this == make_mul(node.dtype);
}

bool scalarop_t::is_min() const
{
    return *this == make_min(node.dtype);
}

bool scalarop_t::is_max() const
{
    return *this == make_max(node.dtype);
}

optional<scalar_t> scalarop_t::get_scale_from_scale() const
{
    if (!node.op.is_mul()) {
        return std::nullopt;
    }

    auto const& lhs = node.children[0];
    if (!lhs.op.is_constant()) {
        return std::nullopt;
    }

    auto const& rhs = node.children[1];
    if (!rhs.op.is_hole()) {
        return std::nullopt;
    }

    bool is_arg0 = rhs.op.get_hole().arg == 0;
    if (!is_arg0) {
        return std::nullopt;
    }

    return lhs.op.get_constant();
}

bool scalarop_t::is_constant_of(scalar_t val) const
{
    return node.op.is_constant() && node.op.get_constant() == val;
}

string scalarop_t::to_cppstr() const
{
    return to_cppstr([](int i) { return "x" + std::to_string(i); });
}
string scalarop_t::to_cppstr(std::function<string(int)> w) const
{
    return node.to_cppstr(w);
}

tuple<string, vector<uint8_t>> scalarop_t::to_cpp_bytes() const
{
    vector<uint8_t> bytes;
    string          str = node.to_cpp_bytes(bytes);
    return {str, bytes};
}

string scalarop_t::type_signature() const
{
    string ret = "";
    auto   add_to_ret = [&](int i) {
        auto maybe = inn_dtype(i);
        if (maybe) {
            ret += write_with_ss(maybe.value());
        } else {
            ret += "_";
        }
    };

    int nholes = 1 + node.max_hole();
    if (nholes == 0) {
        //
    } else {
        add_to_ret(0);
        for (int i = 1; i != nholes; ++i) {
            ret += ",";
            add_to_ret(i);
        }
    }
    ret += "->" + write_with_ss(out_dtype());

    return ret;
}

// Example: combining_op = (y0 * y1) + y2, ops = (x0 + x1, x0 + x1, 7*x0), this replaces
// y0 with x0 + x1 and
// y1 with x2 + x3 and
// y2 with 7*x4
// to get
//   ((x0 + x1) * (x2 + x3)) + (7*x4)
// Note that each input op ends up having distinct inputs
scalarop_t scalarop_t::combine(scalarop_t combining_op, vector<scalarop_t> const& inn_ops)
{
    if (combining_op.num_inputs() != inn_ops.size()) {
        throw std::runtime_error("cannot combine");
    }

    if (combining_op.num_inputs() == 0) {
        return combining_op;
    }

    vector<node_t> inn_nodes = vector_from_each_member(inn_ops, node_t, node);

    int n = inn_nodes.size();
    if (n > 1) {
        int offset = inn_nodes[0].num_inputs();
        for (int i = 1; i != n; ++i) {
            auto& inn_node = inn_nodes[i];
            int   num_here = inn_node.num_inputs();
            inn_node.increment_holes(offset);
            offset += num_here;
        }
    }

    combining_op.node.replace_at_holes(inn_nodes);
    return combining_op.simplify();
}

scalarop_t scalarop_t::replace_arguments(scalarop_t top, vector<scalarop_t> const& bottom_ops)
{
    if (top.num_inputs() != bottom_ops.size()) {
        throw std::runtime_error("cannot replace arguments");
    }

    if (top.num_inputs() == 0) {
        return top;
    }

    vector<node_t> bottom_nodes = vector_from_each_member(bottom_ops, node_t, node);
    top.node.replace_at_holes(bottom_nodes);
    return top.simplify();
}

scalarop_t scalarop_t::from_string(string const& str)
{
    return parse_with_ss<scalarop_t>(str);
}

scalarop_t scalarop_t::make_identity(dtype_t dtype)
{
    static scalarop_t s_f16 = make_arg(0, dtype_t::f16);
    static scalarop_t s_f32 = make_arg(0, dtype_t::f32);
    static scalarop_t s_f64 = make_arg(0, dtype_t::f64);
    static scalarop_t s_c64 = make_arg(0, dtype_t::c64);

    if (dtype == dtype_t::f16) {
        return s_f16;
    } else if (dtype == dtype_t::f32) {
        return s_f32;
    } else if (dtype == dtype_t::f64) {
        return s_f64;
    } else if (dtype == dtype_t::c64) {
        return s_c64;
    } else {
        throw std::runtime_error("should not reach: make_identity");
    }
}

scalarop_t scalarop_t::make_arg(int arg, dtype_t dtype)
{
    string h = op_t::h_str(arg, dtype);
    return parse_with_ss<scalarop_t>(h);
}

scalarop_t scalarop_t::make_constant(scalar_t val)
{
    return parse_with_ss<scalarop_t>("constant{" + write_with_ss(val) + "}");
}

scalarop_t scalarop_t::make_variable(string name, dtype_t dtype)
{
    if (!is_alphanumeric_u(name)) {
        throw std::runtime_error("variable name must be alphanumeric or underscore");
    }
    return parse_with_ss<scalarop_t>("variable{" + name + "|" + write_with_ss(dtype) + "}");
}

// x0 + x1
scalarop_t scalarop_t::make_add(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    return parse_with_ss<scalarop_t>("+[" + h0 + "," + h1 + "]");
}
// x0 * x1
scalarop_t scalarop_t::make_mul(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    return parse_with_ss<scalarop_t>("*[" + h0 + "," + h1 + "]");
}
// x0 / x1
scalarop_t scalarop_t::make_div(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    return parse_with_ss<scalarop_t>("*[" + h0 + ",power{-1}[" + h1 + "]]");
}

scalarop_t scalarop_t::make_neg(dtype_t dtype)
{
    scalarop_t constant = scalarop_t::make_constant(scalar_t::negative_one(dtype));
    scalarop_t variable = scalarop_t::make_arg(0, dtype);
    scalarop_t mul = scalarop_t::make_mul(dtype);
    return scalarop_t::combine(mul, {constant, variable});
}

// min(x0, x1)
scalarop_t scalarop_t::make_min(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    return parse_with_ss<scalarop_t>("ite_<[" + h0 + "," + h1 + "," + h0 + "," + h1 + "]");
}
// max(x0, x1)
scalarop_t scalarop_t::make_max(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    return parse_with_ss<scalarop_t>("ite_>[" + h0 + "," + h1 + "," + h0 + "," + h1 + "]");
}

// x0 <= x1 ? 1.0 : 0.0
scalarop_t scalarop_t::make_is_min(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    string one = "constant{" + write_with_ss(scalar_t::one(dtype)) + "}";
    string zero = "constant{" + write_with_ss(scalar_t::zero(dtype)) + "}";
    return parse_with_ss<scalarop_t>("ite_<=[" + h0 + "," + h1 + "," + one + "," + zero + "]");
}
// x0 >= x1 ? 1.0 : 0.0
scalarop_t scalarop_t::make_is_max(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    string one = "constant{" + write_with_ss(scalar_t::one(dtype)) + "}";
    string zero = "constant{" + write_with_ss(scalar_t::zero(dtype)) + "}";
    return parse_with_ss<scalarop_t>("ite_>=[" + h0 + "," + h1 + "," + one + "," + zero + "]");
}
// x0 == x1 ? 1.0 : 0.0
scalarop_t scalarop_t::make_is_equal(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    string h1 = op_t::h_str(1, dtype);
    string one = "constant{" + write_with_ss(scalar_t::one(dtype)) + "}";
    string zero = "constant{" + write_with_ss(scalar_t::zero(dtype)) + "}";
    return parse_with_ss<scalarop_t>("ite_==[" + h0 + "," + h1 + "," + one + "," + zero + "]");
}

// xn * val
scalarop_t scalarop_t::make_scale_which(scalar_t val, int arg)
{
    string hole = op_t::h_str(arg, val.dtype);
    string constant = "constant{" + write_with_ss(val) + "}";
    return parse_with_ss<scalarop_t>("*[" + hole + "," + constant + "]");
}

// xn * var
scalarop_t scalarop_t::make_scale_which(string var, int arg, dtype_t dtype)
{
    if (!is_alphanumeric_u(var)) {
        throw std::runtime_error("invalid variable name");
    }

    string hole = op_t::h_str(arg, dtype);
    string variable = "variable{" + var + "|" + write_with_ss(dtype) + "}";
    return parse_with_ss<scalarop_t>("*[" + hole + "," + variable + "]");
}

// x0 * val
scalarop_t scalarop_t::make_scale(scalar_t val)
{
    return make_scale_which(val, 0);
}
scalarop_t scalarop_t::make_scale(string var, dtype_t dtype)
{
    if (!is_alphanumeric_u(var)) {
        throw std::runtime_error("invalid variable name");
    }

    return make_scale_which(var, 0, dtype);
}

// x0 - x1
scalarop_t scalarop_t::make_sub(dtype_t dtype)
{
    string negate = write_with_ss(make_scale_which(scalar_t::negative_one(dtype), 1));
    string h0 = op_t::h_str(0, dtype);
    string op = "+[" + h0 + "," + negate + "]";
    return parse_with_ss<scalarop_t>(op);
}
// x0 + val
scalarop_t scalarop_t::make_increment(scalar_t val)
{
    string constant = "constant{" + write_with_ss(val) + "}";
    string h0 = op_t::h_str(0, val.dtype);
    return parse_with_ss<scalarop_t>("+[" + h0 + "," + constant + "]");
}

// e^x0
scalarop_t scalarop_t::make_exp(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("exp[" + h0 + "]");
}

// 1 / (1 + e^x0)
scalarop_t scalarop_t::make_sigmoid(dtype_t dtype)
{
    scalar_t n_one = scalar_t::negative_one(dtype);
    scalar_t one = scalar_t::one(dtype);

    scalarop_t ret = make_scale(n_one);        // -1*x0
    ret = combine(make_exp(dtype), {ret});     // e^(-1*x0)
    ret = combine(make_increment(one), {ret}); // 1 + e^(-1*x0)
    ret = combine(make_rcp(dtype), {ret});     // 1 / (1 + e^(-1*x0))

    return ret;
}

scalarop_t scalarop_t::make_log(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("log[" + h0 + "]");
}

scalarop_t scalarop_t::make_sqrt(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("power{0.5}[" + h0 + "]");
}

scalarop_t scalarop_t::make_inverse(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("power{-1.0}[" + h0 + "]");
}

scalarop_t scalarop_t::make_inverse_sqrt(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("power{-0.5}[" + h0 + "]");
}

scalarop_t scalarop_t::make_square(dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("power{2.0}[" + h0 + "]");
}

scalarop_t scalarop_t::make_power(int n, dtype_t dtype)
{
    string h0 = op_t::h_str(0, dtype);
    return parse_with_ss<scalarop_t>("power{" + write_with_ss(n) + "}[" + h0 + "]");
}

scalarop_t scalarop_t::make_relu(dtype_t dtype)
{
    string arg0 = op_t::h_str(0, dtype);
    string zero = "constant{" + write_with_ss(scalar_t::zero(dtype)) + "}";
    string ite = "ite_<[" + arg0 + "," + zero + "," + zero + "," + arg0 + "]";
    return parse_with_ss<scalarop_t>(ite);
}

scalarop_t scalarop_t::make_mask(compare_t compare, dtype_t dtype)
{
    string arg0 = op_t::h_str(0, dtype);
    string arg1 = op_t::h_str(1, dtype);
    string zero = "constant{" + write_with_ss(scalar_t::zero(dtype)) + "}";
    string one = "constant{" + write_with_ss(scalar_t::one(dtype)) + "}";
    string compare_str = write_with_ss(compare);
    string ite = "ite_" + compare_str + "[" + arg0 + "," + arg1 + "," + one + "," + zero + "]";
    return parse_with_ss<scalarop_t>(ite);
}

scalarop_t scalarop_t::make_silu(dtype_t dtype)
{
    string one = write_with_ss(make_constant(scalar_t::one(dtype)));
    string negative_one = write_with_ss(make_constant(scalar_t::negative_one(dtype)));

    // x
    string x = op_t::h_str(0, dtype);

    // -1*x
    string ret = "*[" + negative_one + "," + x + "]";
    // exp(-1*x)
    ret = "exp[" + ret + "]";
    // exp(-1*x) + 1
    ret = "+[" + ret + "," + one + "]";
    // 1 / (exp(-1*x) + 1)
    ret = "power{-1.0}[" + ret + "]";
    // x * (1 / (exp(-1*x) + 1))
    ret = "*[" + x + "," + ret + "]";

    return parse_with_ss<scalarop_t>(ret);
}

scalarop_t scalarop_t::make_rcp(dtype_t dtype)
{
    node_t     node;
    scalarop_t constant = scalarop_t::make_constant(scalar_t::one(dtype));
    scalarop_t variable = scalarop_t::make_arg(0, dtype);
    scalarop_t div = scalarop_t::make_div(dtype);
    return scalarop_t::combine(div, {constant, variable});
}

scalarop_t scalarop_t::make_relu_deriv(dtype_t dtype)
{
    return make_relu(dtype).derivative(0);
}

scalarop_t scalarop_t::make_from_castable(castable_t c, dtype_t dtype)
{
    if (c == castable_t::add) {
        return make_add(dtype);
    } else if (c == castable_t::mul) {
        return make_mul(dtype);
    } else if (c == castable_t::min) {
        return make_min(dtype);
    } else if (c == castable_t::max) {
        return make_max(dtype);
    } else {
        throw std::runtime_error("should not reach");
    }
}

scalarop_t scalarop_t::make_convert_dtype(dtype_t src, dtype_t dst)
{
    if (src == dst) {
        return make_identity(src);
    }
    string h0 = op_t::h_str(0, src);
    string dst_s = write_with_ss(dst);
    return parse_with_ss<scalarop_t>("to_" + dst_s + "[" + h0 + "]");
}

scalarop_t scalarop_t::make_conjugate(dtype_t inn)
{
    string h0 = op_t::h_str(0, inn);
    return parse_with_ss<scalarop_t>("conj[" + h0 + "]");
}

scalarop_t scalarop_t::make_project_real(dtype_t inn)
{
    string h0 = op_t::h_str(0, inn);
    return parse_with_ss<scalarop_t>("real[" + h0 + "]");
}

scalarop_t scalarop_t::make_project_imag(dtype_t inn)
{
    string h0 = op_t::h_str(0, inn);
    return parse_with_ss<scalarop_t>("imag[" + h0 + "]");
}

scalarop_t scalarop_t::make_complex(dtype_t out)
{
    dtype_t inn = dtype_real_component(out);
    string  h0 = op_t::h_str(0, inn);
    string  h1 = op_t::h_str(1, inn);
    return parse_with_ss<scalarop_t>("complex[" + h0 + "," + h1 + "]");
}

// These + ops could be implemented with scalarop
scalar_t& scalar_t::operator+=(scalar_t const& rhs)
{
    if (dtype != rhs.dtype) {
        throw std::runtime_error("can only add into with the same dtype");
    }

    if (dtype == dtype_t::f16) {
        f16() += rhs.f16();
    } else if (dtype == dtype_t::f32) {
        f32() += rhs.f32();
    } else if (dtype == dtype_t::f64) {
        f64() += rhs.f64();
    } else if (dtype == dtype_t::c64) {
        c64() += rhs.c64();
    } else {
        throw std::runtime_error("missing dtype for adding");
    }

    return *this;
}

scalar_t& scalar_t::operator*=(double mult)
{
    if (dtype == dtype_t::f16) {
        f16() = float16_t(double(f16()) * mult);
    } else if (dtype == dtype_t::f32) {
        f32() = float(double(f32()) * mult);
    } else if (dtype == dtype_t::f64) {
        f64() = f64() * mult;
    } else if (dtype == dtype_t::c64) {
        float real = c64().real();
        float imag = c64().imag();
        c64() = std::complex<float>(float(double(real) * mult), float(double(imag) * mult));
    } else {
        throw std::runtime_error("missing dtype for adding");
    }

    return *this;
}

scalar_t operator+(scalar_t const& lhs, scalar_t const& rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("can only add with the same dtype");
    }
    if (lhs.dtype == dtype_t::f16) {
        return scalar_t(lhs.f16() + rhs.f16());
    } else if (lhs.dtype == dtype_t::f32) {
        return scalar_t(lhs.f32() + rhs.f32());
    } else if (lhs.dtype == dtype_t::f64) {
        return scalar_t(lhs.f64() + rhs.f64());
    } else if (lhs.dtype == dtype_t::c64) {
        return scalar_t(lhs.c64() + rhs.c64());
    } else {
        throw std::runtime_error("missing dtype for adding");
    }
}

scalar_t operator-(scalar_t const& lhs, scalar_t const& rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("can only subtract with the same dtype");
    }
    if (lhs.dtype == dtype_t::f16) {
        return scalar_t(lhs.f16() - rhs.f16());
    } else if (lhs.dtype == dtype_t::f32) {
        return scalar_t(lhs.f32() - rhs.f32());
    } else if (lhs.dtype == dtype_t::f64) {
        return scalar_t(lhs.f64() - rhs.f64());
    } else if (lhs.dtype == dtype_t::c64) {
        return scalar_t(lhs.c64() - rhs.c64());
    } else {
        throw std::runtime_error("missing dtype for adding");
    }
}

scalar_t operator/(scalar_t const& lhs, scalar_t const& rhs)
{
    if (lhs.dtype != rhs.dtype) {
        throw std::runtime_error("can only divide with the same dtype");
    }
    if (lhs.dtype == dtype_t::f16) {
        return scalar_t(lhs.f16() / rhs.f16());
    } else if (lhs.dtype == dtype_t::f32) {
        return scalar_t(lhs.f32() / rhs.f32());
    } else if (lhs.dtype == dtype_t::f64) {
        return scalar_t(lhs.f64() / rhs.f64());
    } else if (lhs.dtype == dtype_t::c64) {
        return scalar_t(lhs.c64() / rhs.c64());
    } else {
        throw std::runtime_error("missing dtype for adding");
    }
}

bool operator==(scalar_t const& lhs, scalar_t const& rhs)
{
    if (lhs.dtype != rhs.dtype) {
        return false;
    }

    switch (lhs.dtype) {
        case dtype_t::f16:
            return lhs.f16() == rhs.f16();
        case dtype_t::f32:
            return lhs.f32() == rhs.f32();
        case dtype_t::f64:
            return lhs.f64() == rhs.f64();
        case dtype_t::c64:
            return lhs.c64() == rhs.c64();
    }
    throw std::runtime_error("should not reach");
}

bool operator!=(scalar_t const& lhs, scalar_t const& rhs)
{
    return !(lhs == rhs);
}

bool operator==(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs)
{
    return write_with_ss(lhs) == write_with_ss(rhs);
}

bool operator!=(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs)
{
    return !(lhs == rhs);
}

bool operator==(scalarop_t const& lhs, scalarop_t const& rhs)
{
    return write_with_ss(lhs) == write_with_ss(rhs);
}

bool operator!=(scalarop_t const& lhs, scalarop_t const& rhs)
{
    return !(lhs == rhs);
}

bool operator==(scalar_ns::op_t const& lhs, scalar_ns::op_t const& rhs)
{
    return write_with_ss(lhs) == write_with_ss(rhs);
}

bool operator!=(scalar_ns::op_t const& lhs, scalar_ns::op_t const& rhs)
{
    return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, dtype_t const& dtype)
{
    if (dtype == dtype_t::f16) {
        out << "f16";
    } else if (dtype == dtype_t::f32) {
        out << "f32";
    } else if (dtype == dtype_t::f64) {
        out << "f64";
    } else if (dtype == dtype_t::c64) {
        out << "c64";
    } else {
        throw std::runtime_error("should not reach: no dtype");
    }

    return out;
}

std::istream& operator>>(std::istream& inn, dtype_t& ret)
{
    char c = inn.get();
    if (c == 'f') {
        c = inn.get();
        if (c == '1') {
            istream_expect(inn, "6");
            ret = dtype_t::f16;
        } else if (c == '3') {
            istream_expect(inn, "2");
            ret = dtype_t::f32;
        } else if (c == '6') {
            istream_expect(inn, "4");
            ret = dtype_t::f64;
        } else {
            throw std::runtime_error("should not reach out dtype");
        }
    } else if (c == 'c') {
        istream_expect(inn, "64");
        ret = dtype_t::c64;
    }

    return inn;
}

std::ostream& operator<<(std::ostream& out, scalar_t const& c)
{
    out << c.dtype << "|";

    if (c.dtype == dtype_t::c64) {
        out << c.c64();
    } else if (c.dtype == dtype_t::f16) {
        if (c.f16() == f16_inf()) {
            out << "INF";
        } else if (c.f16() == f16_ninf()) {
            out << "NINF";
        } else if (half_float::isnan(c.f16())) {
            out << "NAN";
        } else {
            out << c.f16();
        }
    } else if (c.dtype == dtype_t::f32) {
        if (c.f32() == f32_inf()) {
            out << "INF";
        } else if (c.f32() == f32_ninf()) {
            out << "NINF";
        } else if (std::isnan(c.f32())) {
            out << "NAN";
        } else {
            out << c.f32();
        }
    } else if (c.dtype == dtype_t::f64) {
        if (c.f64() == f64_inf()) {
            out << "INF";
        } else if (c.f64() == f64_ninf()) {
            out << "NINF";
        } else if (std::isnan(c.f64())) {
            out << "NAN";
        } else {
            out << c.f64();
        }
    } else {
        throw std::runtime_error("should not reach << scalar_t");
    }

    return out;
}

std::istream& operator>>(std::istream& inn, scalar_t& c)
{
    inn >> c.dtype;

    if (inn.get() != '|') {
        throw std::runtime_error("expected bar in scalar_t parse");
    }

    char peek = inn.peek();

    if (peek == 'I') {
        istream_expect(inn, "INF");
        if (c.dtype == dtype_t::f16) {
            c.f16() = f16_inf();
        } else if (c.dtype == dtype_t::f32) {
            c.f32() = f32_inf();
        } else if (c.dtype == dtype_t::f64) {
            c.f64() = f64_inf();
        } else {
            throw std::runtime_error("unexpected dtype");
        }
    } else if (peek == 'N') {
        int which = istream_expect_or(inn, {"NINF", "NAN"});
        if (which == 0) {
            if (c.dtype == dtype_t::f16) {
                c.f16() = f16_ninf();
            } else if (c.dtype == dtype_t::f32) {
                c.f32() = f32_ninf();
            } else if (c.dtype == dtype_t::f64) {
                c.f64() = f64_ninf();
            } else {
                throw std::runtime_error("unexpected dtype");
            }
        } else if (which == 1) {
            if (c.dtype == dtype_t::f16) {
                c.f16() = f16_nan();
            } else if (c.dtype == dtype_t::f32) {
                c.f32() = f32_nan();
            } else if (c.dtype == dtype_t::f64) {
                c.f64() = f64_nan();
            } else {
                throw std::runtime_error("unexpected dtype");
            }
        } else {
            throw std::runtime_error("should not reach");
        }
    } else {
        if (c.dtype == dtype_t::c64) {
            inn >> c.c64();
        } else if (c.dtype == dtype_t::f16) {
            inn >> c.f16();
        } else if (c.dtype == dtype_t::f32) {
            inn >> c.f32();
        } else if (c.dtype == dtype_t::f64) {
            inn >> c.f64();
        } else {
            throw std::runtime_error("should not reach >> scalar_t");
        }
    }

    return inn;
}

std::ostream& operator<<(std::ostream& out, compare_t const& c)
{
    if (c == compare_t::lt) {
        out << "<";
    } else if (c == compare_t::gt) {
        out << ">";
    } else if (c == compare_t::eq) {
        out << "==";
    } else if (c == compare_t::le) {
        out << "<=";
    } else if (c == compare_t::ge) {
        out << ">=";
    } else {
        throw std::runtime_error("should not reach");
    }

    return out;
}

std::istream& operator>>(std::istream& inn, compare_t& compare)
{
    char c0 = inn.get();
    char c1 = inn.peek();

    if (c0 == '<') {
        if (c1 == '=') {
            inn.get();
            compare = compare_t::le;
        } else {
            compare = compare_t::lt;
        }
    } else if (c0 == '>') {
        if (c1 == '=') {
            inn.get();
            compare = compare_t::ge;
        } else {
            compare = compare_t::gt;
        }
    } else if (c0 == '=') {
        if (c1 == '=') {
            inn.get();
            compare = compare_t::eq;
        } else {
            throw std::runtime_error("invalid parse: compare_t");
        }
    } else {
        throw std::runtime_error("should not reach: parsing compare_t");
    }

    return inn;
}

namespace scalar_ns
{

std::ostream& operator<<(std::ostream& out, op_t const& op)
{
    if (op.is_constant()) {
        out << "constant{" << op.get_constant() << "}";
    } else if (op.is_hole()) {
        out << "hole|" << op.get_hole_dtype() << "@" << op.get_which_input();
    } else if (op.is_variable()) {
        auto const& v = op.get_variable();
        out << "variable{" << v.name << "|" << v.dtype << "}";
    } else if (op.is_add()) {
        out << "+";
    } else if (op.is_mul()) {
        out << "*";
    } else if (op.is_exp()) {
        out << "exp";
    } else if (op.is_log()) {
        out << "log";
    } else if (op.is_power()) {
        out << "power{" << op.get_power() << "}";
    } else if (op.is_ite()) {
        out << "ite_" << op.get_ite_compare();
    } else if (op.is_convert()) {
        out << "to_" << op.get_convert();
    } else if (op.is_conj()) {
        out << "conj";
    } else if (op.is_real()) {
        out << "real";
    } else if (op.is_imag()) {
        out << "imag";
    } else if (op.is_cplex()) {
        out << "complex";
    } else {
        throw std::runtime_error("should not reach");
    }

    return out;
}

std::istream& operator>>(std::istream& inn, op_t& op)
{
    char c = inn.peek();
    if (c == 'c') {
        int which = istream_expect_or(inn, {"constant{", "conj", "complex"});
        if (which == 0) {
            scalar_t v;
            inn >> v;
            istream_expect(inn, "}");
            op.op = scalar_ns::op_t::constant{.value = v};
        } else if (which == 1) {
            op.op = scalar_ns::op_t::conj{};
        } else if (which == 2) {
            op.op = scalar_ns::op_t::cplex{};
        } else {
            throw std::runtime_error("should not reach: ite or imag");
        }
    } else if (c == 'v') {
        istream_expect(inn, "variable{");
        string name = istream_consume_alphanumeric_u(inn);
        istream_expect(inn, "|");
        dtype_t dtype;
        inn >> dtype;
        istream_expect(inn, "}");
        op.op = scalar_ns::op_t::variable{.name = name, .dtype = dtype};
    } else if (c == 'h') {
        istream_expect(inn, "hole|");
        dtype_t dtype;
        inn >> dtype;
        istream_expect(inn, "@");
        int i;
        inn >> i;
        op.op = scalar_ns::op_t::hole{.arg = i, .dtype = dtype};
    } else if (c == '+') {
        inn.get();
        op.op = scalar_ns::op_t::add{};
    } else if (c == '*') {
        inn.get();
        op.op = scalar_ns::op_t::mul{};
    } else if (c == 'e') {
        istream_expect(inn, "exp");
        op.op = scalar_ns::op_t::exp{};
    } else if (c == 'l') {
        istream_expect(inn, "log");
        op.op = scalar_ns::op_t::log{};
    } else if (c == 'p') {
        istream_expect(inn, "power{");
        double i;
        inn >> i;
        op.op = scalar_ns::op_t::power{.to_the = i};
        istream_expect(inn, "}");
    } else if (c == 'i') {
        int which = istream_expect_or(inn, {"ite_", "imag"});
        if (which == 0) {
            compare_t c;
            inn >> c;
            op.op = scalar_ns::op_t::ite{.compare = c};
        } else if (which == 1) {
            op.op = scalar_ns::op_t::imag{};
        } else {
            throw std::runtime_error("should not reach: ite or imag");
        }
    } else if (c == 't') {
        istream_expect(inn, "to_");
        dtype_t d;
        inn >> d;
        op.op = scalar_ns::op_t::convert{.dtype = d};
    } else if (c == 'r') {
        istream_expect(inn, "real");
        op.op = scalar_ns::op_t::real{};
    } else {
        throw std::runtime_error("should not happen");
    }
    return inn;
}

std::ostream& operator<<(std::ostream& out, node_t const& node)
{
    out << node.op;
    if (node.children.size() == 0) {
        return out;
    }
    out << "[";
    out << node.children[0];
    if (node.children.size() > 1) {
        for (int i = 1; i != node.children.size(); ++i) {
            out << "," << node.children[i];
        }
    }
    out << "]";

    return out;
}

std::istream& operator>>(std::istream& inn, node_t& node)
{
    node.children.resize(0);

    inn >> node.op;

    int n = node.op.num_inputs();
    if (n == 0) {
        if (node.op.is_constant()) {
            node.dtype = node.op.get_constant().dtype;
        } else if (node.op.is_hole()) {
            node.dtype = node.op.get_hole_dtype();
        } else if (node.op.is_variable()) {
            node.dtype = node.op.get_variable().dtype;
        } else {
            throw std::runtime_error("parse node: invalid op with no inputs");
        }
        return inn;
    }

    istream_expect(inn, "[");
    {
        node.children.emplace_back();
        inn >> node.children.back();
    }
    if (n > 1) {
        for (int i = 1; i != n; ++i) {
            istream_expect(inn, ",");
            node.children.emplace_back();
            inn >> node.children.back();
        }
    }
    istream_expect(inn, "]");

    vector<dtype_t> inn_dtypes;
    inn_dtypes.reserve(node.children.size());
    for (node_t const& child : node.children) {
        inn_dtypes.push_back(child.dtype);
    }

    optional<dtype_t> maybe_dtype = node.op.type_of(inn_dtypes);
    if (!maybe_dtype) {
        throw std::runtime_error("type failure in node parse");
    }
    node.dtype = maybe_dtype.value();

    return inn;
}

} // namespace scalar_ns

std::ostream& operator<<(std::ostream& out, scalarop_t const& op)
{
    out << op.node;
    return out;
}

std::istream& operator>>(std::istream& inn, scalarop_t& op)
{
    scalar_ns::node_t node;
    inn >> node;
    op = scalarop_t(node);
    return inn;
}

std::ostream& operator<<(std::ostream& out, castable_t const& c)
{
    if (c == castable_t::add) {
        out << "+";
    } else if (c == castable_t::mul) {
        out << "x";
    } else if (c == castable_t::min) {
        out << "v";
    } else if (c == castable_t::max) {
        out << "^";
    } else {
        throw std::runtime_error("should not reach");
    }

    return out;
}

std::istream& operator>>(std::istream& inn, castable_t& castable)
{
    char c;
    inn.read(&c, 1);

    if (c == '+') {
        castable = castable_t::add;
    } else if (c == 'x') {
        castable = castable_t::mul;
    } else if (c == 'v') {
        castable = castable_t::min;
    } else if (c == '^') {
        castable = castable_t::max;
    } else {
        throw std::runtime_error("should not reach");
    }

    return inn;
}

std::ostream& operator<<(std::ostream& out, optional<castable_t> const& maybe_c)
{
    if (maybe_c) {
        out << maybe_c.value();
    } else {
        out << ":";
    }
    return out;
}

std::istream& operator>>(std::istream& inn, optional<castable_t>& castable)
{
    if (inn.peek() == ':') {
        castable = std::nullopt;
    } else {
        castable = castable_t::add; // just a default value
        inn >> castable.value();
    }

    return inn;
}

scalarop_t::node_t const* scalarop_t::get_node() const
{
    return &node;
}
