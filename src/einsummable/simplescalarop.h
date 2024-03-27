#pragma once
#include "../base/setup.h"

#include "scalarop.h"
#include "cutensor.h"

struct simple_scalarop_t {
  // list of simple elementwise ops
  enum uop_t {
    identity,
    neg,           // negation
    sqrt,
    conj,          // conjugate
    rcp,           // reciprocol
    sigmoid,       // 1 / (1 + e^{-x})
    log,
    exp,
    relu,
    square
  };

  enum bop_t {
    add,
    mul,
    min,
    max
  };

  // bop(arg, scale)
  struct scale_t {
    bop_t bop;
    scalar_t scale;
  };

  // scale * f(arg)
  struct unary_t {
    scalar_t scale;
    uop_t op;
  };

  // op(
  //   lhs.scale * lhs.op(arg0), 
  //   rhs.scale * rhs.op(arg1))
  struct binary_t {
    bop_t op;
    unary_t lhs;
    unary_t rhs;
  };

  static cutensorOperator_t uop_to_cutensorOp(uop_t uop){
    switch (uop){
      case identity: return CUTENSOR_OP_IDENTITY;
      case neg: return CUTENSOR_OP_NEG;
      case sqrt: return CUTENSOR_OP_SQRT;
      case conj: return CUTENSOR_OP_CONJ;
      case rcp: return CUTENSOR_OP_RCP;
      case sigmoid: return CUTENSOR_OP_SIGMOID;
      case log: return CUTENSOR_OP_LOG;
      case exp: return CUTENSOR_OP_EXP;
      case relu: return CUTENSOR_OP_RELU;
      default: throw std::runtime_error("Unknown uop");
    }
  }

  static void uop_print(uop_t uop){
    switch (uop){
      case identity: std::cout << "identity"; break;
      case neg: std::cout << "neg"; break;
      case sqrt: std::cout << "sqrt"; break;
      case conj: std::cout << "conj"; break;
      case rcp: std::cout << "rcp"; break;
      case sigmoid: std::cout << "sigmoid"; break;
      case log: std::cout << "log"; break;
      case exp: std::cout << "exp"; break;
      case relu: std::cout << "relu"; break;
      default: throw std::runtime_error("Unknown uop");
    }
    std::cout << std::endl;
  }

  static void bop_print(bop_t bop){
    switch (bop){
      case add: std::cout << "add"; break;
      case mul: std::cout << "mul"; break;
      case min: std::cout << "min"; break;
      case max: std::cout << "max"; break;
      default: throw std::runtime_error("Unknown bop");
    }
    std::cout << std::endl;
  }

  static cutensorOperator_t bop_to_cutensorOp(bop_t bop){
    switch (bop){
      case add: return CUTENSOR_OP_ADD;
      case mul: return CUTENSOR_OP_MUL;
      case min: return CUTENSOR_OP_MIN;
      case max: return CUTENSOR_OP_MAX;
      default: throw std::runtime_error("Unknown bop");
    }
  }

  std::variant<scale_t, unary_t, binary_t> op;

  bool is_scale()  const { return std::holds_alternative<scale_t>(op);  }
  bool is_unary()  const { return std::holds_alternative<unary_t>(op);  }
  bool is_binary() const { return std::holds_alternative<binary_t>(op); }

  int num_inns() const;

  dtype_t get_inn_dtype(int which) const;

  scale_t  const& get_scale()  const { return std::get<scale_t>(op);  }
  unary_t  const& get_unary()  const { return std::get<unary_t>(op);  }
  binary_t const& get_binary() const { return std::get<binary_t>(op); }

  scalarop_t to_scalarop() const;

  static scalarop_t unary_to_scalarop(unary_t u);
  static scalarop_t uop_to_scalarop(uop_t uop, dtype_t dtype);
  static scalarop_t bop_to_scalarop(bop_t bop, dtype_t dtype);
};

struct list_simple_scalarop_t {
  // For n-ary op, the first n args are set
  // args < 0 are temporary and not inputs..
  // An arg of -i is the result of ops[i-1].
  struct op_t {
    simple_scalarop_t op;
    int args[2];
  };

  vector<op_t> ops;

  static
  optional<list_simple_scalarop_t>
  make(scalarop_t const& scalarop);

  scalarop_t to_scalarop() const;

  dtype_t max_dtype() const;
  
  void print(std::ostream& out) const;
};

namespace scalar_ns {
  optional<tuple<
    simple_scalarop_t,
    vector<node_t const*>>>
  pop_to_simple_scalarop(node_t const* node);
}

std::ostream& operator<<(std::ostream& out, simple_scalarop_t const& op);
std::ostream& operator<<(std::ostream& out, simple_scalarop_t::uop_t const& op);
std::ostream& operator<<(std::ostream& out, simple_scalarop_t::bop_t const& op);


