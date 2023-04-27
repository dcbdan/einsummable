#pragma once
#include "setup.h"

enum class compare_t { lt, gt, eq, le, ge };

bool compare(compare_t c, float lhs, float rhs);

namespace scalar_ns {

struct op_t {
  struct constant {
    float value;
  };

  struct hole {
    int arg;
  };

  struct add {};

  struct mul {};

  struct exp {};

  struct power {
    int to_the;
  };

  struct sqrt {};

  struct ite {
    compare_t compare;
  };

  bool is_constant() const;
  bool is_hole()     const;
  bool is_add()      const;
  bool is_mul()      const;
  bool is_exp()      const;
  bool is_power()    const;
  bool is_sqrt()     const;
  bool is_ite()      const;

  float get_constant() const;

  int get_which_input() const;

  int get_power() const;

  compare_t get_ite_compare() const;

  int num_inputs() const;

  float eval(vector<float> const& xs) const;

  std::variant<
    constant, hole,
    add, mul, exp,
    power, sqrt, ite> op;
};

struct node_t;

using node_ptr_t = std::shared_ptr<node_t>;

struct node_t {
  op_t op;
  vector<node_ptr_t> children;

  float eval(vector<float> const& inputs) const;

  node_t gradient(int arg) const;

  node_t simplify() const;

  void which_inputs(set<int>& items) const;

  node_t copy() const;

  int max_hole() const;

  void increment_holes(int incr);

  void remap_holes(map<int, int> const& fmap);
private:
  node_t simplify_once() const;
};

} // scalar_ns

struct scalar_op_t {
  using op_t       = scalar_ns::op_t;
  using node_t     = scalar_ns::node_t;
  using node_ptr_t = scalar_ns::node_ptr_t;

  scalar_op_t();

  scalar_op_t(node_t const& other_node);

  scalar_op_t(node_ptr_t other_node_ptr);

  scalar_op_t(scalar_op_t const& other);

  scalar_op_t& operator=(scalar_op_t const& other);

  float eval(vector<float> const& inputs) const;

  scalar_op_t gradient(int arg) const;

  scalar_op_t simplify();

  set<int> which_inputs() const;

  int num_inputs() const;

  // Example: op = *, ops = (x0 + x1, x2 + x3), this returns
  //   (x0 + x1) * (x2 + x3)
  static scalar_op_t combine(op_t op, vector<scalar_op_t> const& ops);

  // x0 + x1
  static scalar_op_t make_add();

  // x0 * x1
  static scalar_op_t make_mul();

  // xn * val
  static scalar_op_t make_scale_which(float val, int arg);

  // x0 * val
  static scalar_op_t make_scale(float val);

  // x0 - x1
  static scalar_op_t make_sub();

  // x0 + val
  static scalar_op_t make_increment(float val);

  static scalar_op_t make_relu();

  static scalar_op_t make_relu_deriv();

  friend std::ostream& operator<<(
    std::ostream& out, scalar_op_t const& op);
private:
  node_ptr_t node;
};

bool operator==(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);
bool operator!=(scalar_ns::node_t const& lhs, scalar_ns::node_t const& rhs);

bool operator==(scalar_op_t const& lhs, scalar_op_t const& rhs);
bool operator!=(scalar_op_t const& lhs, scalar_op_t const& rhs);

std::ostream& operator<<(std::ostream& out, compare_t const& c);
std::istream& operator>>(std::istream& inn, compare_t& c);

std::ostream& operator<<(std::ostream& out, scalar_ns::op_t const& op);
std::istream& operator>>(std::istream& inn, scalar_ns::op_t& op);

std::ostream& operator<<(std::ostream& out, scalar_ns::node_t const& node);
std::istream& operator>>(std::istream& inn, scalar_ns::node_t& node);

std::ostream& operator<<(std::ostream& out, scalar_op_t const& op);
std::istream& operator>>(std::istream& inn, scalar_op_t& op);


