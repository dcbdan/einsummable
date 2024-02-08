#pragma once
#include "../base/setup.h"

#include "scalarop.h"

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
    pow,
    relu
  };

  enum bop_t {
    add,
    mul,
    min,
    max
  };

  // b(f(x), v)
  // scale_op(op(input), scale)
  struct arg_t {
    bop_t scale_op;
    uop_t op;
    scalar_t scale;
  };

  struct unary_t {
    arg_t arg;
  };

  struct binary_t {
    bop_t op;
    arg_t lhs;
    arg_t rhs;
  };

  std::variant<unary_t, binary_t> op;

  bool is_unary()  const { return std::holds_alternative<unary_t>(op);  }
  bool is_binary() const { return std::holds_alternative<binary_t>(op); }

  unary_t  const& get_unary()  const { return std::get<unary_t>(op);  }
  binary_t const& get_binary() const { return std::get<binary_t>(op); }

  scalarop_t to_scalarop() const;
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
};

namespace scalar_ns {
  // Node that the returned node may not have been advanced at all
  tuple<
    simple_scalarop_t::arg_t,
    node_t const*>
  _pop_arg(node_t const* node);

  optional<tuple<
    simple_scalarop_t::unary_t,
    node_t const*>>
  _pop_to_unary(node_t const* node);

  optional<tuple<
    simple_scalarop_t::binary_t,
    node_t const*,
    node_t const*>>
  _pop_to_binary(node_t const* node);

  optional<tuple<
    simple_scalarop_t,
    vector<node_t const*>>>
  pop_to_simple_scalarop(node_t const* node);
}
