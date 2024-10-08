#pragma once
#include "../base/setup.h"

enum class castable_t { add, mul, min, max };

enum class compare_t { lt, gt, eq, le, ge };

compare_t compare_flip(compare_t);

enum class dtype_t { f16, f32, f64, c64 };

uint64_t dtype_size(dtype_t);

bool dtype_is_real(dtype_t);
bool dtype_is_complex(dtype_t);

dtype_t dtype_real_component(dtype_t);

// Assumption: could be the case that there is something
//             other than real and complex (in the future)

dtype_t dtype_random(bool include_complex = true);

struct scalar_t {
    scalar_t();

    scalar_t(dtype_t, string const&);

    explicit scalar_t(float16_t);
    explicit scalar_t(float);
    explicit scalar_t(double);
    explicit scalar_t(std::complex<float>);

    scalar_t(scalar_t const&);

    float16_t&           f16();
    float&               f32();
    double&              f64();
    std::complex<float>& c64();

    float16_t const&           f16() const;
    float const&               f32() const;
    double const&              f64() const;
    std::complex<float> const& c64() const;

    void const* raw() const;
    // return a string of the number;
    // this removes dtype information
    string str() const;

    scalar_t        convert(dtype_t) const;
    static scalar_t convert(scalar_t const&, dtype_t);

    static scalar_t zero(dtype_t);
    static scalar_t negative_inf(dtype_t);
    static scalar_t inf(dtype_t);

    // not valid for complex
    static scalar_t one(dtype_t);
    static scalar_t negative_one(dtype_t);
    // (Only a very narrow subset of
    //  complex ops are supported, and preventing
    //  these from returning complex makes that easier.
    //  Also, sometimes you want 1 + 1i and sometimes
    //  you want 1 + 0i...

    scalar_t& operator+=(scalar_t const& rhs);
    scalar_t& operator*=(double value);

    dtype_t dtype;
    uint8_t data[8];

private:
    void _copy_to_data(uint8_t const* other, int n);
};

// return castable(val,val,...) with n val inputs
scalar_t agg_power(castable_t castable, uint64_t n, scalar_t val);

namespace scalar_ns
{

struct op_t {
    static op_t   make_constant(scalar_t value);
    static op_t   make_hole(int arg, dtype_t dtype);
    static op_t   make_variable(string name, dtype_t dtype);
    static op_t   make_ite(compare_t);
    static string h_str(int arg, dtype_t dtype);

    struct constant {
        scalar_t value;
    };

    struct hole {
        int     arg;
        dtype_t dtype;
    };

    struct variable {
        string  name;
        dtype_t dtype;
    };

    struct add {
    };

    struct mul {
    };

    struct exp {
    };

    struct log {
    };

    struct power {
        double to_the;
    };

    struct ite {
        compare_t compare;
    };

    struct convert {
        dtype_t dtype;
    };

    // conjugate
    struct conj {
    };

    // real component
    struct real {
    };

    // complex component
    struct imag {
    };

    // create a complex from two real inputs
    struct cplex {
    };

    bool is_constant() const;
    bool is_hole() const;
    bool is_variable() const;
    bool is_add() const;
    bool is_mul() const;
    bool is_exp() const;
    bool is_log() const;
    bool is_power() const;
    bool is_ite() const;
    bool is_convert() const;
    bool is_conj() const;
    bool is_real() const;
    bool is_imag() const;
    bool is_cplex() const;

    scalar_t get_constant() const;

    variable get_variable() const;

    int get_which_input() const;

    dtype_t get_hole_dtype() const;

    hole get_hole() const;

    double get_power() const;

    compare_t get_ite_compare() const;

    dtype_t get_convert() const;

    int num_inputs() const;

    // Note: xs are the inputs of this op.
    //       holes and variables cannot be evaluated.
    scalar_t eval(vector<scalar_t> const& xs) const;

    std::variant<constant,
                 hole,
                 variable,
                 add,
                 mul,
                 exp,
                 log,
                 power,
                 ite,
                 convert,
                 conj,
                 real,
                 imag,
                 cplex>
        op;

    static scalar_t _eval_add(scalar_t lhs, scalar_t rhs);
    static scalar_t _eval_mul(scalar_t lhs, scalar_t rhs);
    static scalar_t _eval_exp(scalar_t inn);
    static scalar_t _eval_log(scalar_t inn);
    static scalar_t _eval_power(double to_the, scalar_t inn);
    static scalar_t
    _eval_ite(compare_t compare, scalar_t lhs, scalar_t rhs, scalar_t if_true, scalar_t if_false);
    static scalar_t _eval_convert(dtype_t new_dtype, scalar_t inn);
    static scalar_t _eval_conj(scalar_t inn);
    static scalar_t _eval_real(scalar_t inn);
    static scalar_t _eval_imag(scalar_t inn);
    static scalar_t _eval_cplex(scalar_t lhs, scalar_t rhs);

    static bool _compare(compare_t c, scalar_t lhs, scalar_t rhs);

    static optional<dtype_t> _type_add(dtype_t lhs, dtype_t rhs);
    static optional<dtype_t> _type_mul(dtype_t lhs, dtype_t rhs);
    static optional<dtype_t> _type_exp(dtype_t inn);
    static optional<dtype_t> _type_log(dtype_t inn);
    static optional<dtype_t> _type_power(dtype_t inn);
    static optional<dtype_t> _type_ite(dtype_t, dtype_t, dtype_t, dtype_t);
    static optional<dtype_t> _type_convert(dtype_t inn, dtype_t out);
    static optional<dtype_t> _type_conj(dtype_t inn);
    static optional<dtype_t> _type_real(dtype_t inn);
    static optional<dtype_t> _type_imag(dtype_t inn);
    static optional<dtype_t> _type_cplex(dtype_t lhs, dtype_t rhs);

    optional<dtype_t> type_of(vector<dtype_t> inns) const;
};

struct node_t {
    op_t           op;
    dtype_t        dtype;
    vector<node_t> children;

    static node_t make_constant(scalar_t value);

    scalar_t eval(vector<scalar_t> const& inputs, map<string, scalar_t> const& variables) const;

    node_t derivative(int arg) const;
    node_t wirtinger_derivative(int arg, bool conjugate) const;

    node_t simplify() const;

    node_t replace_variables(map<string, scalar_t> const& vars) const;

    string to_cppstr(std::function<string(int)> write_hole) const;

    string to_cpp_bytes(vector<uint8_t>& bytes) const;

    void which_inputs(set<int>& items) const;

    void which_variables(set<string>& variables) const;

    // if there are no holes, return -1
    int max_hole() const;

    int num_inputs() const;

    void increment_holes(int incr);

    void remap_holes(map<int, int> const& fmap);

    void replace_at_holes(vector<node_t> const& replace_ops);

    // get the arg types of all holes; if the same arg hole
    // appears with different dtype, throw an error
    map<int, dtype_t> hole_types() const;

    bool type_check() const;

private:
    node_t simplify_once() const;

    void _hole_types(map<int, dtype_t>&) const;

    // make transformations like (hole1 + hole0) -> (hole0 + hole1)
    // at the top level node
    optional<node_t> normalize_order() const;
};

optional<map<int, node_t const*>> _pop_match(node_t const* skeleton, node_t const* node);

} // namespace scalar_ns

dtype_t const& default_dtype();
void           set_default_dtype(dtype_t);

dtype_t const& default_complex_dtype();
void           set_default_complex_dtype(dtype_t);

class simple_scalarop_t;
class list_simple_scalarop_t;

struct scalarop_t {
    using op_t = scalar_ns::op_t;
    using node_t = scalar_ns::node_t;

    scalarop_t();

    scalarop_t(node_t const& node);

    scalar_t eval() const;

    scalar_t eval(scalar_t const& x0) const;

    scalar_t eval(vector<scalar_t> const&      inputs,
                  map<string, scalar_t> const& variables = {}) const;

    // not valid if dtype is complex
    scalarop_t derivative(int arg) const;

    scalarop_t wirtinger_derivative(int arg, bool conjugate) const;

    scalarop_t simplify() const;

    // Note: it is an error if not all variables are provided
    scalarop_t replace_variables(map<string, scalar_t> const& vars) const;

    dtype_t out_dtype() const
    {
        return node.dtype;
    }

    // TODO: make it so that arg must be [0,1,...,num_args-1] and return dtype_t
    optional<dtype_t> inn_dtype(int arg) const;

    bool is_used(int arg) const
    {
        return bool(inn_dtype(arg));
    }

    void remap_inputs(map<int, int> const& remap);

    set<int> which_inputs() const;

    set<string> which_variables() const;

    bool has_variables() const;

    int num_inputs() const;

    int num_variables() const;

    bool is_constant() const;

    bool is_unary() const;

    bool is_binary() const;

    bool is_castable() const;

    bool is_identity() const;

    // Check if this is exactly equal to op[hole0, hole1]
    bool is_mul() const;
    bool is_max() const;
    bool is_min() const;
    bool is_add() const;

    // If this is *[constant,hole0], return the constant
    optional<scalar_t> get_scale_from_scale() const;

    bool is_constant_of(scalar_t val) const;

    // TODO: to_cppstr do not find square, sqrt, (-) or (/) like to_cpp_bytes does !

    string to_cppstr() const;
    string to_cppstr(std::function<string(int)> write_hole) const;

    tuple<string, vector<uint8_t>> to_cpp_bytes() const;

    string type_signature() const;

    // Example: op = *, ops = (x0 + x1, x0 + x1), this returns
    //   (x0 + x1) * (x2 + x3)
    static scalarop_t combine(scalarop_t op, vector<scalarop_t> const& ops);

    static scalarop_t replace_arguments(scalarop_t top, vector<scalarop_t> const& bottom_ops);

    static scalarop_t from_string(string const& str);

    static scalarop_t make_identity(dtype_t d = default_dtype());

    static scalarop_t make_arg(int arg, dtype_t d = default_dtype());

    static scalarop_t make_constant(scalar_t val);

    static scalarop_t make_variable(string name, dtype_t d = default_dtype());

    // x0 + x1
    static scalarop_t make_add(dtype_t d = default_dtype());

    // x0 * x1
    static scalarop_t make_mul(dtype_t d = default_dtype());

    // x0 / x1
    static scalarop_t make_div(dtype_t d = default_dtype());

    static scalarop_t make_neg(dtype_t d = default_dtype());

    // min(x0, x1);
    static scalarop_t make_min(dtype_t d = default_dtype());

    // max(x0, x1);
    static scalarop_t make_max(dtype_t d = default_dtype());

    // x0 <= x1 ? 1.0 : 0.0
    static scalarop_t make_is_min(dtype_t d = default_dtype());

    // x0 >= x1 ? 1.0 : 0.0
    static scalarop_t make_is_max(dtype_t d = default_dtype());

    // x0 == x1 ? 1.0 : 0.0
    static scalarop_t make_is_equal(dtype_t d = default_dtype());

    // xn * (val or variable)
    static scalarop_t make_scale_which(scalar_t val, int arg);
    static scalarop_t make_scale_which(string name, int arg, dtype_t d = default_dtype());

    // x0 * (val or variable)
    static scalarop_t make_scale(scalar_t val);
    static scalarop_t make_scale(string name, dtype_t d = default_dtype());

    // x0 - x1
    static scalarop_t make_sub(dtype_t d = default_dtype());

    // x0 + val
    static scalarop_t make_increment(scalar_t val);

    // e^x0
    static scalarop_t make_exp(dtype_t d = default_dtype());

    // 1 / (1 + e^(-1*x0))
    static scalarop_t make_sigmoid(dtype_t d = default_dtype());

    static scalarop_t make_log(dtype_t d = default_dtype());

    static scalarop_t make_sqrt(dtype_t d = default_dtype());

    static scalarop_t make_inverse(dtype_t d = default_dtype());

    static scalarop_t make_inverse_sqrt(dtype_t d = default_dtype());

    static scalarop_t make_square(dtype_t d = default_dtype());

    static scalarop_t make_power(int n, dtype_t d = default_dtype());

    static scalarop_t make_relu(dtype_t d = default_dtype());

    static scalarop_t make_mask(compare_t compare, dtype_t d = default_dtype());

    static scalarop_t make_silu(dtype_t d = default_dtype());

    static scalarop_t make_rcp(dtype_t d = default_dtype());

    static scalarop_t make_relu_deriv(dtype_t d = default_dtype());

    static scalarop_t make_from_castable(castable_t castable, dtype_t d = default_dtype());

    static scalarop_t make_convert_dtype(dtype_t src, dtype_t dst);

    static scalarop_t make_conjugate(dtype_t d = default_complex_dtype());

    static scalarop_t make_project_real(dtype_t d = default_complex_dtype());

    static scalarop_t make_project_imag(dtype_t d = default_complex_dtype());

    static scalarop_t make_complex(dtype_t d = default_complex_dtype());

    friend std::ostream& operator<<(std::ostream& out, scalarop_t const& op);

    node_t const* get_node() const;

private:
    friend class list_simple_scalarop_t;

    node_t            node;
    map<int, dtype_t> arg_types;
};

std::ostream& operator<<(std::ostream& out, dtype_t const& c);
std::istream& operator>>(std::istream& inn, dtype_t& c);

std::ostream& operator<<(std::ostream& out, scalar_t const& c);
std::istream& operator>>(std::istream& inn, scalar_t& c);

scalar_t operator+(scalar_t const& lhs, scalar_t const& rhs);
scalar_t operator-(scalar_t const& lhs, scalar_t const& rhs);
scalar_t operator/(scalar_t const& lhs, scalar_t const& rhs);

bool operator==(scalar_t const& lhs, scalar_t const& rhs);
bool operator!=(scalar_t const& lhs, scalar_t const& rhs);

bool operator==(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);
bool operator!=(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);

bool operator==(scalarop_t const& lhs, scalarop_t const& rhs);
bool operator!=(scalarop_t const& lhs, scalarop_t const& rhs);

bool operator==(scalar_ns::op_t const& lhs, scalar_ns::op_t const& rhs);
bool operator!=(scalar_ns::op_t const& lhs, scalar_ns::op_t const& rhs);

std::ostream& operator<<(std::ostream& out, compare_t const& c);
std::istream& operator>>(std::istream& inn, compare_t& c);

namespace scalar_ns
{
// TODO: constants and args need datatype annotation
std::ostream& operator<<(std::ostream& out, op_t const& op);
std::istream& operator>>(std::istream& inn, op_t& op);

// TODO: when constructing node, deduce the dtype
std::ostream& operator<<(std::ostream& out, node_t const& node);
std::istream& operator>>(std::istream& inn, node_t& node);
} // namespace scalar_ns

std::ostream& operator<<(std::ostream& out, scalarop_t const& op);
std::istream& operator>>(std::istream& inn, scalarop_t& op);

std::ostream& operator<<(std::ostream& out, castable_t const& c);
std::istream& operator>>(std::istream& inn, castable_t& c);

std::istream& operator>>(std::istream& inn, optional<castable_t> const& maybe_c);
std::ostream& operator<<(std::ostream& out, optional<castable_t> const& maybe_c);
