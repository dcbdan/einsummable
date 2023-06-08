#pragma once
#include "../base/setup.h"

enum class castable_t { add, mul, min, max };

enum class compare_t { lt, gt, eq, le, ge };

enum class dtype_t { f16, f32, f64, c64 };

struct scalar_t {
  scalar_t();
  scalar_t(float16_t);
  scalar_t(float);
  scalar_t(double);
  scalar_t(std::complex<float>);
  scalar_t(scalar_t const&);

  float16_t          & f16();
  float              & f32();
  double             & f64();
  std::complex<float>& c64();

  float16_t            const& f16() const;
  float                const& f32() const;
  double               const& f64() const;
  std::complex<float>  const& c64() const;

  static scalar_t convert(scalar_t const&, dtype_t);

  static scalar_t zero(dtype_t);
  static scalar_t one(dtype_t);

  dtype_t dtype;
  uint8_t data[8];

private:
  void _copy_to_data(uint8_t const* other, int n);
};

namespace scalar_ns {

struct op_t {
  static op_t make_constant(scalar_t value);

  struct constant {
    scalar_t value;
  };

  struct hole {
    int arg;
    dtype_t dtype;
  };

  struct add {};

  struct mul {};

  struct exp {};

  struct power {
    double to_the;
  };

  struct ite {
    compare_t compare;
  };

  bool is_constant() const;
  bool is_hole()     const;
  bool is_add()      const;
  bool is_mul()      const;
  bool is_exp()      const;
  bool is_power()    const;
  bool is_ite()      const;

  scalar_t get_constant() const;

  int get_which_input() const;

  double get_power() const;

  compare_t get_ite_compare() const;

  hole get_hole() const;

  int num_inputs() const;

  scalar_t eval(vector<scalar_t> const& xs) const;

  std::variant<
    constant, hole,
    add, mul, exp,
    power, ite> op;

  static scalar_t _eval_add(scalar_t lhs, scalar_t rhs);
  static scalar_t _eval_mul(scalar_t lhs, scalar_t rhs);
  static scalar_t _eval_exp(scalar_t inn);
  static scalar_t _eval_power(double to_the, scalar_t inn);
  static scalar_t _eval_ite(compare_t compare,
    scalar_t lhs, scalar_t rhs, scalar_t if_true, scalar_t if_false);
  static bool _compare(compare_t c, scalar_t lhs, scalar_t rhs);

  static optional<dtype_t> _type_add(dtype_t lhs, dtype_t rhs);
  static optional<dtype_t> _type_mul(dtype_t lhs, dtype_t rhs);
  static optional<dtype_t> _type_exp(dtype_t inn);
  static optional<dtype_t> _type_power(dtype_t inn);
  static optional<dtype_t> _type_ite(dtype_t, dtype_t, dtype_t, dtype_t);
};

struct node_t {
  op_t op;
  dtype_t dtype; // TODO
  vector<node_t> children;

  static node_t make_constant(scalar_t value);

  scalar_t eval(vector<scalar_t> const& inputs) const;

  node_t derivative(int arg) const;

  node_t simplify() const;

  string to_cppstr(std::function<string(int)> write_hole) const;

  string to_cpp_bytes(vector<uint8_t>& bytes) const;

  void which_inputs(set<int>& items) const;

  // if there are no holes, return -1
  int max_hole() const;

  int num_inputs() const;

  void increment_holes(int incr);

  void remap_holes(map<int, int> const& fmap);

  void replace_at_holes(vector<node_t> const& replace_ops);
private:
  node_t simplify_once() const;
};

} // scalar_ns

struct scalarop_t {
  using op_t       = scalar_ns::op_t;
  using node_t     = scalar_ns::node_t;

  scalarop_t();

  scalarop_t(node_t const& node);

  scalar_t eval(vector<scalar_t> const& inputs) const;

  // TODO: maybe force all things to be a float type before
  //       taking the derivative. have not thought about complex
  scalarop_t derivative(int arg) const;

  scalarop_t simplify() const;

  void remap_inputs(map<int, int> const& remap);

  set<int> which_inputs() const;

  int num_inputs() const;

  bool is_constant() const;

  bool is_unary() const;

  bool is_binary() const;

  bool is_castable() const;

  bool is_mul() const;

  bool is_constant_of(scalar_t val) const;

  string to_cppstr() const;
  string to_cppstr(std::function<string(int)> write_hole) const;
  tuple<string, vector<uint8_t>> to_cpp_bytes() const;

  // Example: op = *, ops = (x0 + x1, x0 + x1), this returns
  //   (x0 + x1) * (x2 + x3)
  static scalarop_t combine(scalarop_t op, vector<scalarop_t> const& ops);

  static scalarop_t from_string(string const& str);

  static scalarop_t make_identity();

  // x0 + x1
  static scalarop_t make_add();

  // x0 * x1
  static scalarop_t make_mul();

  // x0 / x1
  static scalarop_t make_div();

  // min(x0, x1);
  static scalarop_t make_min();

  // max(x0, x1);
  static scalarop_t make_max();

  // xn * val
  static scalarop_t make_scale_which(float val, int arg);

  // x0 * val
  static scalarop_t make_scale(float val);

  // x0 - x1
  static scalarop_t make_sub();

  // x0 + val
  static scalarop_t make_increment(float val);

  static scalarop_t make_exp();

  static scalarop_t make_relu();

  static scalarop_t make_relu_deriv();

  static scalarop_t make_from_castable(castable_t castable);

  friend std::ostream& operator<<(
    std::ostream& out, scalarop_t const& op);
private:
  node_t node;
};

std::ostream& operator<<(std::ostream& out, dtype_t const& c);
std::istream& operator>>(std::istream& inn, dtype_t& c);

std::ostream& operator<<(std::ostream& out, scalar_t const& c);
std::istream& operator>>(std::istream& inn, scalar_t& c);

bool operator==(scalar_t const& lhs, scalar_t const& rhs);
bool operator!=(scalar_t const& lhs, scalar_t const& rhs);

bool operator==(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);
bool operator!=(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);

bool operator==(scalarop_t const& lhs, scalarop_t const& rhs);
bool operator!=(scalarop_t const& lhs, scalarop_t const& rhs);

std::ostream& operator<<(std::ostream& out, compare_t const& c);
std::istream& operator>>(std::istream& inn, compare_t& c);

namespace scalar_ns {
  // TODO: constants and args need datatype annotation
std::ostream& operator<<(std::ostream& out, op_t const& op);
std::istream& operator>>(std::istream& inn, op_t& op);

// TODO: when constructing node, deduce the dtype
std::ostream& operator<<(std::ostream& out, node_t const& node);
std::istream& operator>>(std::istream& inn, node_t& node);
}

std::ostream& operator<<(std::ostream& out, scalarop_t const& op);
std::istream& operator>>(std::istream& inn, scalarop_t& op);

